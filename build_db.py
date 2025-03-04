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
from docx import Document
import json

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

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def split_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_file(file_path):
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith((".txt", ".md")):
        text = extract_text_from_txt(file_path)
    elif file_path.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        return [], []
    if text:
        file_chunks = split_text(text)
        return file_chunks, [(file_path, i) for i in range(len(file_chunks))]
    return [], []

def process_files(file_paths):
    start_time = time.time()
    chunks = []
    metadata = []
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Use IndexIDMap for removal support
    vector_store = FAISS.from_texts(
        chunks, 
        embedding_model, 
        metadatas=[{"source": m[0], "chunk_id": m[1]} for m in metadata]
    )
    vector_store.save_local(output_dir, "faiss_index.bin")
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        for meta in metadata:
            f.write(f"{meta[0]},{meta[1]}\n")
    
    # Save chunks per file
    chunks_dir = os.path.join(output_dir, "chunks")
    Path(chunks_dir).mkdir(exist_ok=True)
    chunk_idx = 0
    for file_path, _ in metadata:
        file_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(file_path))
        chunk_file = os.path.join(chunks_dir, f"{file_name}.txt")
        if not os.path.exists(chunk_file):
            file_chunks = chunks[chunk_idx:chunk_idx + sum(1 for m in metadata if m[0] == file_path)]
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write("\n---\n".join(file_chunks) + "\n---\n")
        chunk_idx += 1
    
    elapsed = time.time() - start_time
    print(f"Faiss storage took {elapsed:.2f} seconds")
    print(f"Stored {len(embeddings)} embeddings in {output_dir}")

def update_vector_store(data_dir, output_dir):
    total_start = time.time()
    state_file = os.path.join(output_dir, "state.json")
    
    # Load previous state or init
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            prev_state = json.load(f)
    else:
        prev_state = {}

    # Get current file states
    current_state = {}
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
    for fp in file_paths:
        if fp.lower().endswith((".pdf", ".txt", ".md", ".docx")):
            current_state[fp] = os.path.getmtime(fp)

    # Detect changes
    to_process = [fp for fp in file_paths if fp not in prev_state or prev_state[fp] != current_state[fp]]
    if not os.path.exists(output_dir) or not file_paths:
        print("Full rebuild required.")
        chunks, metadata = process_files(file_paths)
        if not chunks:
            print("No data to process.")
            return
        embeddings = generate_embeddings(chunks)
        store_in_faiss(embeddings, metadata, chunks, output_dir)
        with open(state_file, "w") as f:
            json.dump(current_state, f)
        print(f"Full build stored {len(chunks)} chunks in {time.time() - total_start:.2f} seconds")
        return

    # Process only changed/new files
    if not to_process:
        print("No changes detected.")
        return
    chunks, metadata = process_files(to_process)
    if not chunks:
        print("No new/changed data to process.")
        return
    embeddings = generate_embeddings(chunks)

    # Load existing FAISS
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(output_dir, embedding_model, "faiss_index.bin", allow_dangerous_deserialization=True)
    
    # Remove old embeddings for changed files
    changed_files = [fp for fp in to_process if fp in prev_state]
    if changed_files:
        ids_to_remove = []
        for i, doc in enumerate(vector_store.docstore._dict.values()):
            if doc.metadata["source"] in changed_files:
                ids_to_remove.append(i)
        if ids_to_remove:
            vector_store.index.remove_ids(np.array(ids_to_remove))
            print(f"Removed {len(ids_to_remove)} stale chunks for {len(changed_files)} changed files")

    # Add new embeddings directly (no merge_from)
    vector_store.add_texts(
        texts=chunks,
        metadatas=[{"source": m[0], "chunk_id": m[1]} for m in metadata]
    )
    vector_store.save_local(output_dir, "faiss_index.bin")

    # Update state
    with open(state_file, "w") as f:
        json.dump(current_state, f)
    
    print(f"Updated with {len(chunks)} new/changed chunks in {time.time() - total_start:.2f} seconds")

def main():
    print("Starting text extraction, embedding, and storage process...")
    update_vector_store(DATA_DIR, OUTPUT_DIR)
    print("Process complete!")

if __name__ == "__main__":
    main()