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

# Load Faiss indices
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("vector_db", embeddings, "faiss_index.bin", allow_dangerous_deserialization=True)
history_store = FAISS.from_texts([""], embeddings)

# Load vLLM with Llama2-13B
llm = VLLM(
    model="meta-llama/Llama-2-13b-chat-hf",
    gpu_memory_utilization=0.90,
    max_model_len=1024,
    max_num_seqs=512,
    vllm_kwargs={
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
    },
    trust_remote_code=True,
)

# Simple RetrievalQA—no custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(k=5)
)

def query_hal(qa_chain, query, history_store):
    import time
    start = time.time()
    docs = history_store.similarity_search(query, k=10)
    context = "\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
    full_query = f"Previous context:\n{context}\n\nCurrent query: {query}" if context else query
    print(f"Before invoke: {time.time() - start:.2f} sec")
    result = qa_chain.invoke({"query": full_query})
    print(f"After invoke: {time.time() - start:.2f} sec")
    answer = result['result'].strip()
    print(f"HAL Over9000's Answer: {answer}")
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