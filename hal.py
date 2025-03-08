from custom_vllm import CustomVLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import requests
import torch.distributed as dist
import atexit
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams
from typing import List

load_dotenv()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
atexit.register(cleanup)

def check_vllm_server(endpoint="http://localhost:8000/v1"):
    try:
        response = requests.get(f"{endpoint}/models", timeout=5)
        if response.status_code == 200:
            print("Connected to vLLM server successfully.")
            return True
        else:
            raise Exception(f"Server responded with status {response.status_code}")
    except Exception as e:
        print(f"Error: Could not connect to vLLM server at {endpoint} - {str(e)}")
        print("Please start the server with: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B-Instruct ...")
        return False

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
client = QdrantClient("localhost", port=6333)
collection_name = "hal_docs"

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
                metadata={
                    "source": result.payload.get("source", ""),
                    "chunk_id": result.payload.get("chunk_id", 0)
                }
            )
            for result in search_results
        ]

history_store = FAISS.from_texts([""], embeddings)

server_endpoint = "http://localhost:8000/v1"
if not check_vllm_server(server_endpoint):
    exit(1)

llm = CustomVLLM()

template = """[INST] <<SYS>>
You are HAL, a precise AI assistant. For questions, answer directly; for statements, give a short, plain acknowledgment (no extra detail), using session history only if relevant. Keep it under 100 tokens, ending in a full sentence.
<</SYS>>

Context: {context}

Question: {question} [/INST]"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=QdrantRetriever(),
    chain_type_kwargs={"prompt": prompt},
)

def query_hal(qa_chain, query, history_store):
    import time
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", legacy=False)
    start = time.time()
    if not query.strip():
        print("No question provided.")
        return None
    query = str(query)
    
    # History retrieval
    history_start = time.time()
    docs = history_store.similarity_search(query, k=2)
    history_context = "\n".join([doc.page_content for doc in docs])
    history_time = time.time() - history_start
    
    # Qdrant retrieval (once)
    qdrant_start = time.time()
    rag_docs = qa_chain.retriever._get_relevant_documents(query)
    qdrant_time = time.time() - qdrant_start
    
    # Combine context
    rag_context = "\n".join([doc.page_content for doc in rag_docs])
    combined_context = f"Session History:\n{history_context}\n\nDocument Context:\n{rag_context}" if history_context and rag_context else rag_context or history_context
    
    # Generation with pre-fetched docs, bypassing retriever
    generation_start = time.time()
    final_prompt = qa_chain.combine_documents_chain.llm_chain.prompt.format(
        context=combined_context,
        question=query
    )
    answer = llm._call(final_prompt)  # Fix: Use llm directly, not qa_chain.llm
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start
    print(f"HAL [total: {total_time:.2f} sec, history: {history_time:.2f} sec, qdrant: {qdrant_time:.2f} sec, generation: {generation_time:.2f} sec]: {answer}")
    history_store.add_texts([f"Q: {query}\nA: {answer}"])
    return answer

if __name__ == "__main__":
    print("Welcome to HAL - Type 'exit' to quit")
    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        query_hal(qa_chain, query, history_store)