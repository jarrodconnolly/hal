from config import embeddings, client, history_store, COLLECTION_NAME
from qdrant_client.http.models import SearchParams

def get_history_context(query: str) -> str:
    """Fetch history context from FAISS."""
    docs = history_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

def get_rag_context(query: str) -> str:
    """Fetch RAG context from Qdrant."""
    query_embedding = embeddings.embed_query(query)
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=200)
    ).points
    return "\n".join([result.payload.get("content", "") for result in search_results])

def add_to_history(query: str, answer: str):
    """Add query-answer pair to FAISS history."""
    history_store.add_texts([f"Q: {query}\nA: {answer}"])