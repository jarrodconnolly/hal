"""Configuration settings for HAL's FastAPI server, vLLM inference, and Qdrant vector store."""

# Model Settings
MODEL_NAME = "microsoft/Phi-4-mini-instruct"  # LLM model for inference
#MODEL_NAME = "/home/totally/.cache/huggingface/hub/models--local--Llama-3.1-8B-Instruct-GPTQ-4bit/snapshots/4bit-128g-20250312"  # LLM model for inference
OPENAI_API_TIMEOUT = 30  # Timeout for API requests in seconds

# Embedding Settings
EMBEDDING_MODEL = "thenlper/gte-large"  # Model for generating embeddings
#EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Qdrant Vector Store Settings
QDRANT_HOST = "localhost"  # Host for Qdrant server
QDRANT_PORT = 6333  # Port for Qdrant server
DOCS_COLLECTION = "hal_docs"  # Collection for document chunks
HISTORY_COLLECTION = "hal_history"  # Collection for session history
FACTS_COLLECTION = "hal_facts"  # Collection for facts
HNSW_M = 32  # HNSW index parameter for Qdrant

# HAL API Settings
API_HOST = "localhost"  # Host for HAL FastAPI server
API_PORT = 8001  # Port for HAL FastAPI server

# vLLM Inference Settings
VLLM_HOST = "localhost"  # Host for vLLM inference server
VLLM_PORT = 8000  # Port for vLLM inference server
VLLM_ENDPOINT = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"  # Full endpoint for vLLM

# MongoDB Settings
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "hal"
USERS_COLLECTION = "users"


# Open Telegram API Settings
COLLECTOR_ENDPOINT = "localhost:4317"