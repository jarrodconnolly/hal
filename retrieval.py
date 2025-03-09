from config import embeddings, client, history_store, COLLECTION_NAME
from qdrant_client.http.models import SearchParams

def get_history_context(query: str) -> str:
    """Fetch history context from FAISS."""
    docs = history_store.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

def get_rag_context(query: str) -> tuple[str, float, list, list]:
    """Fetch RAG context from Qdrant, return context, top score, chunk IDs, and scores."""
    query_embedding = embeddings.embed_query(query)
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=200)
    ).points
    context = "\n".join([result.payload.get("content", "") for result in search_results])
    top_score = max([result.score for result in search_results], default=0.0) if search_results else 0.0
    chunk_ids = [result.id for result in search_results]
    scores = [result.score for result in search_results]  # All scores
    return context, top_score, chunk_ids, scores

def add_to_history(query: str, answer: str):
    """Add query-answer pair to FAISS history."""
    history_store.add_texts([f"Q: {query}\nA: {answer}"])