from custom_vllm import CustomVLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import torch.distributed as dist
import atexit
import os

# Load environment variables (optional)
load_dotenv()

# Cleanup for distributed processes
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
atexit.register(cleanup)

# Check server availability
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
        print("Please start the server with: python -m vllm.entrypoints.openai.api_server --model TheBloke/Llama-2-13B-chat-GPTQ ...")
        return False

# Setup embeddings and vector stores
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")  # Updated from all-MiniLM-L6-v2
vector_store = FAISS.load_local(
    "vector_db",
    embeddings,
    "faiss_index.bin",
    allow_dangerous_deserialization=True
)
history_store = FAISS.from_texts([""], embeddings)

# Verify server before proceeding
server_endpoint = "http://localhost:8000/v1"
if not check_vllm_server(server_endpoint):
    exit(1)

# Initialize custom LLM
llm = CustomVLLM()

# Define HAL's prompt template
template = """[INST] <<SYS>>
You are HAL, a precise AI assistant. For questions, answer directly; for statements, give a short, plain acknowledgment (no extra detail), using session history only if relevant. Keep it under 100 tokens, ending in a full sentence.
<</SYS>>

Context: {context}

Question: {question} [/INST]"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Setup RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(k=5),
    chain_type_kwargs={"prompt": prompt},
)

# Query function with history
def query_hal(qa_chain, query, history_store):
    import time
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", legacy=False)
    start = time.time()
    if not query.strip():
        print("No question provided.")
        return None
    # Ensure query is a plain string
    query = str(query)
    # print(f"Query type: {type(query)}, value: {query}")  # Debug
    docs = history_store.similarity_search(query, k=2)
    history_context = "\n".join([doc.page_content for doc in docs])
    rag_docs = vector_store.similarity_search(query, k=5)
    rag_context = "\n".join([doc.page_content for doc in rag_docs])
    combined_context = f"Session History:\n{history_context}\n\nDocument Context:\n{rag_context}" if history_context and rag_context else rag_context or history_context
    full_query = f"Use this info to answer:\n{combined_context}\n\nQuestion: {query}" if combined_context else query
    result = qa_chain.invoke({"query": full_query})
    answer = result["result"].strip()
    print(f"HAL [{time.time() - start:.2f} sec]: {answer}")
    history_store.add_texts([f"Q: {query}\nA: {answer}"])
    return answer

# Main chat REPL
if __name__ == "__main__":
    print("Welcome to HAL - Type 'exit' to quit")
    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        query_hal(qa_chain, query, history_store)