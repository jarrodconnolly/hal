from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
import logging
import httpx
import json
import time

logging.basicConfig(filename="hal_api.log", level=logging.INFO, format="%(message)s")
app = FastAPI()

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
client = QdrantClient("localhost", port=6333)
collection_name = "hal_docs"
history_store = FAISS.from_texts([""], embeddings)

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

    # History context
    history_start = time.time()
    docs = history_store.similarity_search(query, k=2)
    history_context = "\n".join([doc.page_content for doc in docs])
    history_time = time.time() - history_start

    # Rag context
    qdrant_start = time.time()
    query_embedding = embeddings.embed_query(query)
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=200)
    ).points
    rag_context = "\n".join([result.payload.get("content", "") for result in search_results])
    qdrant_time = time.time() - qdrant_start

    combined_context = f"{history_context}\n\n{rag_context}" if history_context else rag_context
    final_prompt = user_template.format(context=combined_context, question=query)

    async def stream_response():
        generation_start = time.time()
        ttfb = None
        answer = ""
        payload = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": True
        }
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", "http://localhost:8000/v1/chat/completions", json=payload, timeout=30) as response:
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
        history_store.add_texts([f"Q: {query}\nA: {answer}"])
        yield f"\n\nTIMINGS:{json.dumps(timings)}"

    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)