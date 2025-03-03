import os
import shutil
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import time
from multiprocessing import Pool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = os.path.expanduser("~/data")
OUTPUT_DIR = "vector_db"
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def split_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_file(file_path):
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        text = extract_text_from_txt(file_path)
    else:
        return [], []
    if text:
        file_chunks = split_text(text)
        return file_chunks, [(file_path, i) for i in range(len(file_chunks))]
    return [], []

def process_files(data_dir):
    start_time = time.time()
    chunks = []
    metadata = []
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
    with Pool() as pool:
        results = pool.map(process_file, file_paths)
    for file_chunks, file_metadata in results:
        if file_chunks:
            chunks.extend(file_chunks)
            metadata.extend(file_metadata)
            print(f"Processed {os.path.basename(file_metadata[0][0])}: {len(file_chunks)} chunks")
    elapsed = time.time() - start_time
    print(f"Text extraction and chunking took {elapsed:.2f} seconds")
    return chunks, metadata

def generate_embeddings(chunks, model_name=EMBEDDING_MODEL):
    start_time = time.time()
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(chunks, batch_size=128, show_progress_bar=True)
    elapsed = time.time() - start_time
    print(f"Embedding generation took {elapsed:.2f} seconds")
    return embeddings

def store_in_faiss(embeddings, metadata, chunks, output_dir):
    start_time = time.time()
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embeddings)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Use LangChain FAISS to save index and metadata
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(chunks, embedding_model, metadatas=[{"source": m[0], "chunk_id": m[1]} for m in metadata])
    vector_store.save_local(output_dir, "faiss_index.bin")
    # faiss.write_index(index if not torch.cuda.is_available() else faiss.index_gpu_to_cpu(index), 
                      # os.path.join(output_dir, "faiss_index.bin.faiss"))
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        for meta in metadata:
            f.write(f"{meta[0]},{meta[1]}\n")
    
    # Save chunks per file
    chunks_dir = os.path.join(output_dir, "chunks")
    Path(chunks_dir).mkdir(exist_ok=True)
    chunk_idx = 0
    for file_path, _ in metadata:
        # Use a sanitized filename (replace special chars)
        file_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(file_path))
        chunk_file = os.path.join(chunks_dir, f"{file_name}.txt")
        if not os.path.exists(chunk_file):  # Write only once per file
            file_chunks = chunks[chunk_idx:chunk_idx + sum(1 for m in metadata if m[0] == file_path)]
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write("\n---\n".join(file_chunks) + "\n---\n")
        chunk_idx += 1
    
    elapsed = time.time() - start_time
    print(f"Faiss storage took {elapsed:.2f} seconds")
    print(f"Stored {len(embeddings)} embeddings in {output_dir}")

def wipe_and_reload(data_dir, output_dir):
    total_start = time.time()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleared existing database at {output_dir}")
    chunks, metadata = process_files(data_dir)
    if not chunks:
        print("No data to process. Check your data directory.")
        return
    embeddings = generate_embeddings(chunks)
    store_in_faiss(embeddings, metadata, chunks, output_dir)
    total_elapsed = time.time() - total_start
    print(f"Total process took {total_elapsed:.2f} seconds")

def main():
    print("Starting text extraction, embedding, and storage process...")
    wipe_and_reload(DATA_DIR, OUTPUT_DIR)
    print("Process complete!")

if __name__ == "__main__":
    main()