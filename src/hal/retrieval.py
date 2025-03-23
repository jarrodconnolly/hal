"""Qdrant vector store operations for HAL's RAG, history, and user facts."""

import time

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import SearchParams
from sentence_transformers import SentenceTransformer

from .config import (
    DOCS_COLLECTION,
    EMBEDDING_MODEL,
    FACTS_COLLECTION,
    HISTORY_COLLECTION,
    HNSW_M,
    QDRANT_HOST,
    QDRANT_PORT,
)
from .logging_config import configure_logging

# Logging Configuration
logger = configure_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Move client init up top
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


def create_collections() -> None:
    # Create collections if they don't exist
    if not client.collection_exists(FACTS_COLLECTION):
        logger.info(f"Creating collection {FACTS_COLLECTION}")
        client.create_collection(
            collection_name=FACTS_COLLECTION,
            vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            hnsw_config=rest.HnswConfigDiff(m=HNSW_M),
        )

    if not client.collection_exists(HISTORY_COLLECTION):
        logger.info(f"Creating collection {HISTORY_COLLECTION}")
        client.create_collection(
            collection_name=HISTORY_COLLECTION,
            vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            hnsw_config=rest.HnswConfigDiff(m=HNSW_M),
        )


def get_history_context(query: str, session_id: str) -> tuple[str, list[float]]:
    """Fetch prior query-answer context from Qdrant history for a session.

    Args:
        query: The user's input string to match against history.
        session_id: Identifier for the user's session to filter history.

    Returns:
        A tuple of (content, scores), where content is concatenated history chunks
        and scores are similarity values.
    """
    query_embedding = embeddings.encode(query, show_progress_bar=False)
    search_results = client.query_points(
        collection_name=HISTORY_COLLECTION,
        query=query_embedding,
        query_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="session_id", match=rest.MatchValue(value=session_id)
                )
            ]
        ),
        limit=5,  # Fetch 5 chunks
    ).points
    chunks = [result.payload.get("content", "") for result in search_results]
    scores = [result.score for result in search_results]
    return "\n".join(chunks), scores


def get_rag_context(query: str) -> tuple[str, list[float]]:
    """Fetch RAG context from Qdrant, return content and scores."""
    query_embedding = embeddings.encode(query, show_progress_bar=False)
    search_results = client.query_points(
        collection_name=DOCS_COLLECTION,
        query=query_embedding,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=50),
    ).points
    context = "\n".join(
        [result.payload.get("content", "") for result in search_results]
    )
    scores = [result.score for result in search_results]
    for result in search_results:
        logger.info(
            f"Qdrant chunk ID {result.id}: '{result.payload.get('content', '')[:50]}...' | Score: {result.score:.3f}"
        )
    return context, scores


def add_to_history(query: str, answer: str, session_id: str, user_id: str) -> None:
    content = f"Q: {query}\nA: {answer}"
    embedding = embeddings.encode(content, show_progress_bar=False)
    point_id = int(time.time() * 1000)
    client.upsert(
        collection_name=HISTORY_COLLECTION,
        points=[
            rest.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "content": content,
                    "session_id": session_id,
                    "user_id": user_id,
                    "timestamp": time.time(),
                },
            )
        ],
    )


def store_user_facts(facts: list[str], session_id: str, source_query: str) -> None:
    """Store extracted user facts in the hal_facts collection."""
    for fact in facts:
        embedding = embeddings.encode(fact, show_progress_bar=False)
        point_id = int(time.time() * 1000)
        try:
            client.upsert(
                collection_name=FACTS_COLLECTION,
                points=[
                    rest.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "fact": fact,
                            "user_id": session_id,  # Still store as "user_id" in payload for now
                            "timestamp": int(time.time()),
                            "source_query": source_query,
                        },
                    )
                ],
            )
            logger.info(f"Stored fact: {fact} for session_id: {session_id}")
        except Exception as e:
            logger.error(f"Error storing fact '{fact}': {e}")
            raise


def get_user_facts(user_id: str, limit: int = 5) -> list[str]:
    """Retrieve user facts from hal_facts for the given user_id."""
    try:
        scroll_result = client.scroll(
            collection_name=FACTS_COLLECTION,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="user_id",
                        match=rest.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        facts = [point.payload["fact"] for point in scroll_result[0]]
        logger.info(f"Retrieved {len(facts)} facts for user_id: {user_id}")
        return facts
    except Exception as e:
        logger.error(f"Error retrieving facts for user_id {user_id}: {e}")
        return []
