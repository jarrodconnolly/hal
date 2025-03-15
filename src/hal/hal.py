"""HAL FastAPI server - streams tech query responses with RAG, history, and external context."""

import asyncio
import hashlib
import json
import logging
import re
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from . import hal_facts
from .config import (
    API_HOST,
    API_PORT,
    DOCS_COLLECTION,
    HISTORY_COLLECTION,
    MODEL_NAME,
    OPENAI_API_TIMEOUT,
    QDRANT_HOST,
    QDRANT_PORT,
    VLLM_HOST,
    VLLM_PORT,
)
from .external import analyze_query, fetch_external
from .retrieval import (
    add_to_history,
    create_collections,
    get_history_context,
    get_rag_context,
    get_user_facts,
    store_user_facts,
)

# Logging Configuration
logging.basicConfig(filename="hal_api.log", level=logging.INFO, format="%(message)s")

VLLM_ENDPOINT = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
app = FastAPI()


class LoginRequest(BaseModel):
    username: str


class QueryRequest(BaseModel):
    query: str
    session_id: str


@app.post("/login")
async def login(request: LoginRequest):
    """Login and generate a session ID, clearing prior history."""
    # Derive deterministic session_id from username (hash for simplicity)
    session_id = hashlib.sha256(request.username.encode()).hexdigest()
    # Clear history for this session_id
    client.delete(
        collection_name=HISTORY_COLLECTION,
        points_selector=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="session_id", match=rest.MatchValue(value=session_id)
                )
            ]
        ),
    )
    # Get chunk count from hal_docs
    collection_info = client.get_collection(DOCS_COLLECTION)
    chunk_count = collection_info.points_count
    return {"session_id": session_id, "chunk_count": chunk_count}


async def fetch_contexts(query: str, session_id: str) -> str:
    """Fetch and combine context from history, RAG, and external sources for a query.

    Args:
        query: The user's input string to analyze and fetch context for.
        session_id: Identifier for the user's session, used to filter history.

    Returns:
        A string of combined context, cleaned of excess newlines.
    """
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
        logging.info(f"Fetch {name} timing: {elapsed:.2f}s")
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
    # Normalize newlines to avoid bloating prompt with empty lines.
    combined_context = re.sub(r"\n+", "\n", combined_context.strip())

    logging.info(f"Query: {query} | Similarity Score: {top_score}")
    return combined_context


def build_prompt(combined_context: str, query: str, session_id: str) -> dict:
    """Build the vLLM payload with system prompt, user facts, and query context.
    
    Args:
        combined_context: Pre-fetched context from history, RAG, and external sources.
        query: The user's input string to answer.
        session_id: Identifier for the user's session, used to fetch facts.
    
    Returns:
        A dict payload for vLLM's chat completions API, with system and user messages.
    """
    facts = get_user_facts(session_id)
    facts_str = "\n".join(facts) if facts else "No known user facts."
    system_prompt = (
        "You are HAL, a sharp AI assistant for tech queries. Answer the query below in concise, plain English. "
        "Use these user facts if relevant: {facts_str}—mention them explicitly if applicable. "
        "Focus solely on the query—use context only if it directly applies, otherwise ignore it. "
        "Do not repeat phrases or ramble—provide one clear answer."
    ).format(facts_str=facts_str)

    # Limit context to 4096 chars to balance relevance and vLLM's 16k token capacity.
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


async def stream_response(payload: dict, query: str, session_id: str):
    """Stream vLLM's response chunks, log timings, and store history/facts.
    
    Args:
        payload: The vLLM chat completions payload with system and user messages.
        query: The user's input string, logged and stored in history.
        session_id: Identifier for the user's session, used for history and facts.
    
    Yields:
        Response chunks from vLLM, followed by timing metadata.
    """
    generation_start = time.time()
    ttfb = 0.0
    answer = ""
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(f"Sending to vLLM: {json.dumps(payload)}")
    else:
        logging.info("Sending to vLLM: [payload omitted]")

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST", VLLM_ENDPOINT, json=payload, timeout=OPENAI_API_TIMEOUT
            ) as response:
                async for chunk in response.aiter_lines():
                    # Strip "data: " prefix from vLLM's SSE stream, process JSON chunks.
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
                                yield content
        # Catch vLLM connection issues, yield error but keep partial answer for history.
        except (httpx.RequestError, httpx.TimeoutException) as e:
            error_msg = "Error: vLLM connection failed"
            logging.error(error_msg)
            yield error_msg

    generation_time = time.time() - generation_start
    timings = {"generation": generation_time, "ttfb": ttfb}
    logging.info(f"Query: {query} | Response: {answer} | Timings: {timings}")
    add_to_history(query, answer, session_id)
    _, facts = hal_facts.extract_user_facts(query)
    if facts and facts != ["none"]:
        store_user_facts(facts, session_id, query)
    yield f"\n\nTIMINGS:{json.dumps(timings)}"


# Updated endpoint
@app.post("/query_stream")
async def query_hal_stream(request: QueryRequest):
    """Handle query streaming, combining context and vLLM response.
    
    Args:
        request: QueryRequest with the user's query and session_id.
    
    Returns:
        StreamingResponse with vLLM chunks and timing metadata.
    """
    start = time.time()
    query = request.query.strip()
    session_id = request.session_id
    if not query:
        return StreamingResponse(
            iter(["No question provided."]), media_type="text/plain"
        )

    combined_context = await fetch_contexts(query, session_id)
    payload = build_prompt(combined_context, query, session_id)
    total_time = time.time() - start  # Rough total, generation adds more

    async def stream_with_total():
        async for chunk in stream_response(payload, query, session_id):
            yield chunk
        timings_update = {
            "total": total_time + payload.get("timings", {}).get("generation", 0)
        }
        logging.info(f"Updated timings: {timings_update}")
        yield f"\n\nUPDATED_TIMINGS:{json.dumps(timings_update)}"

    return StreamingResponse(stream_with_total(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    create_collections()
    uvicorn.run(app, host=API_HOST, port=API_PORT)
