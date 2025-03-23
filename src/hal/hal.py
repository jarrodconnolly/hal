"""HAL FastAPI server - streams tech query responses with RAG, history, and external context via WebSockets."""

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import toml
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.exporter import Compression
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel
from qdrant_client import QdrantClient

from . import hal_facts
from .config import (
    API_HOST,
    API_PORT,
    COLLECTOR_ENDPOINT,
    DOCS_COLLECTION,
    MODEL_NAME,
    OPENAI_API_TIMEOUT,
    QDRANT_HOST,
    QDRANT_PORT,
    VLLM_HOST,
    VLLM_PORT,
)
from .db import authenticate, get_db
from .external import analyze_query, fetch_external
from .logging_config import configure_logging
from .retrieval import (
    add_to_history,
    create_collections,
    get_history_context,
    get_rag_context,
    get_user_facts,
    store_user_facts,
)

# Logging Configuration
logger = configure_logging()

# Pull HAL version from pyproject.toml
with open("./pyproject.toml", "r") as f:
    config = toml.load(f)
    hal_version = config["project"]["version"]

# OTel setup
resource = Resource(attributes={"service.name": "hal", "service.version": hal_version})
trace.set_tracer_provider(TracerProvider(resource=resource))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=COLLECTOR_ENDPOINT, compression=Compression.Gzip, insecure=True
        )
    )
)
tracer = trace.get_tracer(__name__)

# Instrument Pymongo for OTel tracing
PymongoInstrumentor().instrument()

# httpx tracing only works with the httpx.AsyncClient
HTTPXClientInstrumentor().instrument()

VLLM_ENDPOINT = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"

qclient = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await get_db()
    yield
    # Cleanup if needed


app = FastAPI(lifespan=lifespan)

# Instrument FastAPI app for OTel tracing
# FastAPIInstrumentor.instrument_app(app)


origins = ["http://127.0.0.1:1430", "tauri://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class LogoutRequest(BaseModel):
    session_id: str | None


class QueryRequest(BaseModel):
    query: str
    session_id: str


async def fetch_contexts(query: str, session_id: str) -> str:
    """Fetch and combine context from history, RAG, and external sources for a query."""
    sources, keywords = analyze_query(query)
    fetchers = [
        (get_history_context, "history", [" ".join(keywords), session_id]),
        (get_rag_context, "qdrant", [" ".join(keywords)]),
        (fetch_external, "external", [query, 0, sources]),
    ]

    async def fetch_with_timing(func, name, args):
        start = time.time()
        result, scores = (
            await func(*args) if asyncio.iscoroutinefunction(func) else func(*args)
        )
        elapsed = time.time() - start
        logger.info("Fetch data", name=name, timing=elapsed)
        return result, scores, elapsed

    tasks = [fetch_with_timing(func, name, args) for func, name, args in fetchers]
    results = await asyncio.gather(*tasks)

    contexts = []
    top_score = 0.0
    for result, scores, _ in results:
        if result:
            contexts.append(result if isinstance(result, str) else "")
        if scores:
            top_score = max(
                top_score, max(scores) if isinstance(scores, list) else scores
            )

    combined_context = "\n\n".join(contexts)
    combined_context = re.sub(r"\n+", "\n", combined_context.strip())
    logger.info("Query complete", query=query, top_score=top_score)
    return combined_context


def build_prompt(combined_context: str, query: str, session_id: str) -> dict:
    """Build the vLLM payload with system prompt, user facts, and query context."""
    facts = get_user_facts(session_id)
    facts_str = "\n".join(facts) if facts else "No known user facts."
    system_prompt = (
        "You are HAL, a sharp AI assistant for tech queries. Answer the query below in concise, plain English. "
        "Use these user facts if relevant: {facts_str}—mention them explicitly if applicable. "
        "Focus solely on the query—use context only if it directly applies, otherwise ignore it. "
        "Do not repeat phrases or ramble—provide one clear answer."
    ).format(facts_str=facts_str)

    user_prompt = f"Context (optional, use only if directly relevant): {combined_context[:4096]}\n\nQuery: {query}"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.5,
        "stream": True,
    }
    return payload


async def stream_response(
    websocket: WebSocket,
    payload: dict,
    query: str,
    session_id: str,
    user_id: str,
    traceparent: str,
):
    """Stream vLLM's response chunks over WebSocket, log timings, and store history/facts."""
    generation_start = time.time()
    ttfb = 0.0
    answer = ""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Sending to vLLM", payload=payload)
    else:
        logger.info("Sending to vLLM: [payload omitted]")

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST", VLLM_ENDPOINT, json=payload, timeout=OPENAI_API_TIMEOUT
            ) as response:
                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        data = chunk[6:]
                        if data != "[DONE]":
                            chunk_json = json.loads(data)
                            content = chunk_json["choices"][0]["delta"].get(
                                "content", ""
                            )
                            if content:
                                if ttfb == 0.0:
                                    ttfb = time.time() - generation_start
                                answer += content
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "query_response",
                                            "content": content,
                                            "done": False,
                                        }
                                    )
                                )
                        elif data == "[DONE]":
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "query_response",
                                        "content": "",
                                        "done": True,
                                        "traceparent": traceparent,
                                    }
                                )
                            )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            error_msg = f"Error: vLLM connection failed: {e}"
            logger.error(error_msg)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "query_response",
                        "content": error_msg,
                        "done": True,
                        "traceparent": traceparent,
                    }
                )
            )

    generation_time = time.time() - generation_start
    timings = {"generation": generation_time, "ttfb": ttfb}
    logger.info("Response generated", query=query, answer=answer, timings=timings)
    add_to_history(query, answer, session_id, user_id)
    _, facts = hal_facts.extract_user_facts(query)
    if facts and facts != ["none"]:
        store_user_facts(facts, session_id, query)

    # Send stats separately
    collection_info = qclient.get_collection(DOCS_COLLECTION)
    await websocket.send_text(
        json.dumps(
            {
                "type": "stats",
                "chunk_count": collection_info.points_count,
                "ttfb": timings["ttfb"],
                "generation": timings["generation"],
            }
        )
    )


@app.websocket("/ws/hal")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for HAL login and query streaming."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info("WS Message", message=message)
            msg_type = message.get("type")

            # Extract traceparent
            carrier = {"traceparent": message.get("traceparent", "")}
            ctx = TraceContextTextMapPropagator().extract(carrier)
            with tracer.start_as_current_span(f"{msg_type}", context=ctx) as span:
                if msg_type == "logout":
                    try:
                        logout_data = LogoutRequest(**message)
                    except ValueError as e:
                        logger.error("Invalid logout format", error=str(e))
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "logout_response",
                                    "error": "Invalid logout data",
                                    "traceparent": carrier["traceparent"],
                                }
                            )
                        )
                        continue

                    if not logout_data.session_id:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "logout_response",
                                    "error": "Not logged in",
                                    "traceparent": carrier["traceparent"],
                                }
                            )
                        )
                        continue
                    await websocket.send_text(json.dumps({"type": "logout_response"}))
                    continue
                elif msg_type == "login":
                    try:
                        login_data = LoginRequest(**message)
                    except ValueError as e:
                        logger.error("Invalid login format", error=str(e))
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "login_response",
                                    "error": "Invalid login data",
                                    "traceparent": carrier["traceparent"],
                                }
                            )
                        )
                        continue

                    user = await authenticate(login_data.username, login_data.password)
                    if user:
                        session_id = uuid4().hex
                        user_id = user["username"]
                        logger.info("Login successful", username=user_id)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "login_response",
                                    "session_id": session_id,
                                    "user_id": user_id,
                                    "message": "Login successful",
                                    "traceparent": carrier["traceparent"],
                                }
                            )
                        )
                        # Send initial stats
                        collection_info = qclient.get_collection(DOCS_COLLECTION)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "stats",
                                    "chunk_count": collection_info.points_count,
                                }
                            )
                        )
                    else:
                        logger.info("Login failed", username=login_data.username)
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "login_response",
                                    "error": "Invalid username or password",
                                    "traceparent": carrier["traceparent"],
                                }
                            )
                        )

                elif msg_type == "query":
                    query = message.get("query", "").strip()
                    session_id = message.get("session_id", "")
                    user_id = message.get("user_id", "")
                    if not query:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "query_response",
                                    "content": "No question provided.",
                                    "done": True,
                                    "traceparent": carrier["traceparent"],
                                }
                            )
                        )
                        continue

                    combined_context = await fetch_contexts(query, session_id)
                    payload = build_prompt(combined_context, query, session_id)
                    await stream_response(
                        websocket,
                        payload,
                        query,
                        session_id,
                        user_id,
                        carrier["traceparent"],
                    )

    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        await websocket.close()


def main():
    create_collections()
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_config=None, log_level="info")


if __name__ == "__main__":
    main()
