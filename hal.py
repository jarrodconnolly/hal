from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from custom_vllm import CustomVLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
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
llm = CustomVLLM()

template = """[INST] <<SYS>>
You are HAL, a precise AI assistant. For questions, answer directly; for statements, give a short, plain acknowledgment (no extra detail), using session history only if relevant. Keep it under 100 tokens, ending in a full sentence.
<</SYS>>

Context: {context}

Question: {question} [/INST]"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

class QdrantRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_embedding = embeddings.embed_query(query)
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=5,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=200)
        ).points
        return [
            Document(
                page_content=result.payload.get("content", ""),
                metadata={"source": result.payload.get("source", ""), "chunk_id": result.payload.get("chunk_id", 0)}
            )
            for result in search_results
        ]

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=QdrantRetriever(),
    chain_type_kwargs={"prompt": prompt},
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_hal(request: QueryRequest):
    start = time.time()
    query = request.query.strip()
    if not query:
        return {"response": "No question provided.", "timings": {}}

    history_start = time.time()
    docs = history_store.similarity_search(query, k=2)
    history_context = "\n".join([doc.page_content for doc in docs])
    history_time = time.time() - history_start

    qdrant_start = time.time()
    rag_docs = qa_chain.retriever._get_relevant_documents(query)
    qdrant_time = time.time() - qdrant_start

    rag_context = "\n".join([doc.page_content for doc in rag_docs])
    combined_context = f"Session History:\n{history_context}\n\nDocument Context:\n{rag_context}" if history_context else rag_context
    final_prompt = qa_chain.combine_documents_chain.llm_chain.prompt.format(context=combined_context, question=query)

    generation_start = time.time()
    answer = llm._call(final_prompt)
    generation_time = time.time() - generation_start

    total_time = time.time() - start
    timings = {"total": total_time, "history": history_time, "qdrant": qdrant_time, "generation": generation_time}
    logging.info(f"Query: {query} | Response: {answer} | Timings: {timings}")
    history_store.add_texts([f"Q: {query}\nA: {answer}"])
    return {"response": answer, "timings": timings}

@app.post("/query_stream")
async def query_hal_stream(request: QueryRequest):
    start = time.time()
    query = request.query.strip()
    if not query:
        return StreamingResponse(iter(["No question provided."]), media_type="text/plain")

    history_start = time.time()
    docs = history_store.similarity_search(query, k=2)
    history_context = "\n".join([doc.page_content for doc in docs])
    history_time = time.time() - history_start

    qdrant_start = time.time()
    rag_docs = qa_chain.retriever._get_relevant_documents(query)
    qdrant_time = time.time() - qdrant_start

    rag_context = "\n".join([doc.page_content for doc in rag_docs])
    combined_context = f"Session History:\n{history_context}\n\nDocument Context:\n{rag_context}" if history_context else rag_context
    final_prompt = qa_chain.combine_documents_chain.llm_chain.prompt.format(context=combined_context, question=query)

    async def stream_response():
        generation_start = time.time()
        ttfb = None
        answer = ""
        payload = {
            "model": llm.model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": True
        }
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{llm.server_endpoint}/chat/completions", json=payload, timeout=llm.timeout) as response:
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
        timings = {"total": total_time, "history": history_time, "qdrant": qdrant_time, "generation": generation_time, "ttfb": ttfb}
        logging.info(f"Query: {query} | Response: {answer} | Timings: {timings}")
        history_store.add_texts([f"Q: {query}\nA: {answer}"])
        yield f"\n\nTIMINGS:{json.dumps(timings)}"  # Clear separator

    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)