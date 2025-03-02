import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import time
import os

def load_file_chunks(file_path):
    """Load pre-extracted chunks from a specific file."""
    chunk_file = os.path.join("vector_db", "chunks", 
                             "".join(c if c.isalnum() or c in "._-" else "_" 
                                     for c in os.path.basename(file_path)) + ".txt")
    if os.path.exists(chunk_file):
        with open(chunk_file, "r", encoding="utf-8") as f:
            return f.read().split("\n---\n")[:-1]  # Last is empty
    return []

def query_index(query, k=5, show_text=True):
    total_start = time.time()

    # Load the Faiss index
    start_time = time.time()
    index = faiss.read_index("vector_db/faiss_index.bin")
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    load_elapsed = time.time() - start_time
    print(f"Loading Faiss index took {load_elapsed:.2f} seconds")

    # Load the embedding model and generate query embedding
    start_time = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2", 
                               device="cuda" if torch.cuda.is_available() else "cpu")
    query_embedding = model.encode([query], batch_size=1)
    embed_elapsed = time.time() - start_time
    print(f"Embedding generation took {embed_elapsed:.2f} seconds")

    # Search the index
    start_time = time.time()
    D, I = index.search(np.array(query_embedding).astype("float32"), k=k)
    search_elapsed = time.time() - start_time
    print(f"Index search took {search_elapsed:.2f} seconds")

    # Load metadata and chunks for matched files
    start_time = time.time()
    with open("vector_db/metadata.txt", "r") as f:
        metadata = [line.strip().split(",") for line in f]
    
    # Load only relevant chunk files
    file_chunks = {}
    if show_text:
        for idx in I[0]:
            file_path = metadata[idx][0]
            if file_path not in file_chunks:
                file_chunks[file_path] = load_file_chunks(file_path)
    
    print(f"\nTop {k} matches for query: '{query}'")
    for idx, dist in zip(I[0], D[0]):
        file_path, chunk_id = metadata[idx]
        print(f"Match {idx}: {file_path}, chunk {chunk_id}, distance {dist}")
        if show_text and file_path in file_chunks:
            chunk_text = file_chunks[file_path][int(chunk_id)]
            print(f"Text: {chunk_text[:100]}...")
            print("-" * 80)
    result_elapsed = time.time() - start_time
    print(f"Processing results took {result_elapsed:.2f} seconds")

    total_elapsed = time.time() - total_start
    print(f"Total query process took {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    query_index("What skills does a lead developer need?")