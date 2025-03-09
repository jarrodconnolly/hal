from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# Connect to your Qdrant instance
client = QdrantClient("localhost", port=6333)
collection_name = "hal_docs"

# Same embeddings HAL uses
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

# Your query—mimics "I am so mad, what is my name"
query = "I am so mad, what is my name"
query_embedding = embeddings.embed_query(query)

# Manual similarity search—matches QdrantRetriever in hal.py
search_results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=5,  # Same as HAL’s retriever
    with_payload=True  # Gets content + metadata
).points

# Dump the results
for i, result in enumerate(search_results):
    print(f"Result {i+1}:")
    print(f"Score: {result.score}")
    print(f"Content: {result.payload.get('content', 'No content')}")
    print(f"Metadata: {result.payload.get('source', 'No source')} | Chunk ID: {result.payload.get('chunk_id', 'No ID')}")
    print("---")