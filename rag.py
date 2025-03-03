from langchain_community.llms import VLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from vllm import SamplingParams
from dotenv import load_dotenv
import torch.distributed as dist
import atexit
import os

load_dotenv()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
atexit.register(cleanup)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("vector_db", embeddings, "faiss_index.bin", allow_dangerous_deserialization=True)
history_store = FAISS.from_texts([""], embeddings)

sampling_params = SamplingParams(n=3, best_of=3, stop=["\n\n", "."])  # Generate 3, pick best

llm = VLLM(
    model="TheBloke/Llama-2-13B-chat-GPTQ",
    revision="gptq-4bit-128g-actorder_True",  # Lock this in
    gpu_memory_utilization=0.85,
    max_model_len=2048,
    max_num_seqs=128,
    quantization="gptq",
    dtype="float16",
    trust_remote_code=True,
    sampling_params=sampling_params,
)

template = """[INST] <<SYS>>
You are HAL, a precise AI assistant inspired by HAL-9000. Answer only the question asked, using the context (especially session history) if relevant. Keep your response concise, natural, and under 100 tokens. Do not guess names, repeat prior answers unless requested, or add buttons, options, instructions, or closers.
<</SYS>>

Context: {context}

Question: {question} [/INST]"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(k=5),
    chain_type_kwargs={"prompt": prompt},
)

def query_hal(qa_chain, query, history_store):
    import time
    start = time.time()
    docs = history_store.similarity_search(query, k=2)  # Broaden for name recall
    context = "\n".join([doc.page_content for doc in docs])
    full_query = f"Previous context:\n{context}\n\nCurrent query: {query}" if context else query
    print(f"Full query: {full_query}")
    print(f"Before invoke: {time.time() - start:.2f} sec")
    result = qa_chain.invoke({"query": full_query})
    print(f"After invoke: {time.time() - start:.2f} sec")
    answer = result["result"].strip()
    print(f"HAL's Answer: {answer}")
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