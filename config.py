import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from qdrant_client import QdrantClient

logging.basicConfig(filename="hal_api.log", level=logging.INFO, format="%(message)s")

VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "hal_docs"
SIMILARITY_THRESHOLD = 0.7  # Trigger external calls below this score
TIMEOUT = 30

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
history_store = FAISS.from_texts([""], embeddings)