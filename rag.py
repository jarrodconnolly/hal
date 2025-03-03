from langchain_community.llms import VLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import torch.distributed as dist
import atexit

load_dotenv()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
atexit.register(cleanup)

# Load Faiss index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("vector_db", embeddings, "faiss_index.bin", allow_dangerous_deserialization=True)

MODEL_CONFIGS = {
  "meta-llama/Llama-2-13b-chat-hf": {"quantization": "bitsandbytes", "load_format": "bitsandbytes"},
  "mistralai/Mistral-7B-Instruct-v0.3": {"quantization": "bitsandbytes", "load_format": "bitsandbytes"},
}

MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"  # Swap here

# Load vLLM with Llama2-13B
llm = VLLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.90,  # Lower to 85% for KV cache
    max_model_len=1024,           # Lower to save VRAM
    vllm_kwargs=MODEL_CONFIGS[MODEL_NAME],
    enforce_eager=False,          # Optimize memory
    trust_remote_code=True,
)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",           # Simple context stuffing
    retriever=vector_store.as_retriever(k=5),  # Top 5 chunks
    # return_source_documents=True  # For debugging
)

def query_hal(query):
    result = qa_chain.invoke({"query": query})
    print(f"HAL's Answer: {result['result']}")
    # print("\nSources:")
    # for i, doc in enumerate(result['source_documents']):
    #     print(f"Chunk {i+1}: {doc.page_content[:100]}... (Distance: {doc.metadata.get('distance', 'N/A')})")

if __name__ == "__main__":
    query_hal("What skills does a lead developer need?")