from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import time
from config import VLLM_ENDPOINT, MODEL_NAME, TIMEOUT, logging, SIMILARITY_THRESHOLD
from retrieval import get_history_context, get_rag_context, add_to_history
from external import fetch_external

app = FastAPI()

system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id>
You are HAL, a sharp AI with edge. Answer questions straight—keep it real. Use history and external sources—drop them in smooth—then context if needed. No fluff, full sentence always. Plain text only.
<|end_header_id>"""

user_template = """<|start_header_id|>user<|end_header_id>
{context}

{question}
<|eot_id>"""

class QueryRequest(BaseModel):
    query: str

@app.post("/query_stream")
async def query_hal_stream(request: QueryRequest):
    start = time.time()
    query = request.query.strip()
    if not query:
        return StreamingResponse(iter(["No question provided."]), media_type="text/plain")

    history_start = time.time()
    history_context = get_history_context(query)
    history_time = time.time() - history_start

    qdrant_start = time.time()
    rag_context, top_score, chunk_ids, chunk_scores = get_rag_context(query)  # Added chunk_scores
    qdrant_time = time.time() - qdrant_start

    external_context = fetch_external(query, top_score) if top_score < SIMILARITY_THRESHOLD else ""
    combined_context = f"{history_context}\n\n{rag_context}\n\n{external_context}" if history_context or external_context else rag_context
    final_prompt = user_template.format(context=combined_context, question=query)

    async def stream_response():
        generation_start = time.time()
        ttfb = 0.0
        answer = ""
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": True
        }
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", VLLM_ENDPOINT, json=payload, timeout=TIMEOUT) as response:
                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        data = chunk[6:]
                        if data != "[DONE]":
                            chunk_json = json.loads(data)
                            content = chunk_json["choices"][0]["delta"].get("content", "")
                            if content:
                                if ttfb == 0.0:
                                    ttfb = time.time() - start
                                answer += content
                                yield content
        generation_time = time.time() - generation_start
        total_time = time.time() - start
        timings = {
            "total": total_time,
            "history": history_time,
            "qdrant": qdrant_time,
            "generation": generation_time,
            "ttfb": ttfb,
            "top_score": top_score
        }
        # Log chunk IDs with their scores
        chunk_info = {str(id): score for id, score in zip(chunk_ids, chunk_scores)}
        logging.info(f"Query: {query} | Response: {answer} | Chunks: {json.dumps(chunk_info)} | Timings: {timings}")
        add_to_history(query, answer)
        yield f"\n\nTIMINGS:{json.dumps(timings)}"

    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)