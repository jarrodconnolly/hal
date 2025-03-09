from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import time
from config import VLLM_ENDPOINT, MODEL_NAME, TIMEOUT, logging
from retrieval import get_history_context, get_rag_context, add_to_history

app = FastAPI()

system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id>
You are HAL, a sharp AI with edge. Answer questions straight—keep it real. Use history for names—drop it in smooth—then context if needed. No fluff, full sentence always. Plain text only.
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

    # Fetch contexts
    history_start = time.time()
    history_context = get_history_context(query)
    history_time = time.time() - history_start

    qdrant_start = time.time()
    rag_context = get_rag_context(query)
    qdrant_time = time.time() - qdrant_start

    combined_context = f"{history_context}\n\n{rag_context}" if history_context else rag_context
    final_prompt = user_template.format(context=combined_context, question=query)

    async def stream_response():
        generation_start = time.time()
        ttfb = None
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
                                if ttfb is None:
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
            "ttfb": ttfb
        }
        logging.info(f"Query: {query} | System Prompt: {system_prompt} | User Prompt: {final_prompt} | Response: {answer} | Timings: {timings}")
        add_to_history(query, answer)
        yield f"\n\nTIMINGS:{json.dumps(timings)}"

    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)