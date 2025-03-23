"""Check specific points in Qdrant's hal_docs collection by ID.

Connects to a local Qdrant instance and retrieves points from the hal_docs collection
using a predefined list of point IDs. Prints each point's content, source, and chunk ID
(if present) for manual inspection. Used to debug or verify chunk data post-ingestion.
"""
from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "hal_docs"

# Points to check—paste your array here
point_ids = [5773, 10418, 4280]

# Fetch points by ID
points = client.retrieve(
    collection_name=collection_name, ids=point_ids, with_payload=True
)

# Print results
print(f"Checking {len(points)} points from {collection_name}:")
print("----------------------------------------")
for i, point in enumerate(points):
    print(f"Point {i + 1} (ID: {point.id}):")
    print(f"Content: {point.payload.get('content', 'No content')}")
    print(
        f"Source: {point.payload.get('source', 'No source')} | Chunk ID: {point.payload.get('chunk_id', 'No ID')}"
    )
    print("----------------------------------------")
