from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)
collection_name = "hal_docs"

# Points to check—paste your array here
point_ids = [4505, 4555, 4543, 9628, 7585]  # From bread query—swap as needed

# Fetch points by ID
points = client.retrieve(
    collection_name=collection_name,
    ids=point_ids,
    with_payload=True  # Gets content + metadata
)

# Print results
print(f"Checking {len(points)} points from {collection_name}:")
print("----------------------------------------")
for i, point in enumerate(points):
    print(f"Point {i+1} (ID: {point.id}):")
    print(f"Content: {point.payload.get('content', 'No content')}")
    print(f"Source: {point.payload.get('source', 'No source')} | Chunk ID: {point.payload.get('chunk_id', 'No ID')}")
    print("----------------------------------------")