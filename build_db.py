import os
from pathlib import Path
import pypdfium2 as pdfium
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import time
from multiprocessing import Pool
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from docx import Document as DocxDocument
import json

DATA_DIR = os.path.expanduser("~/data")
OUTPUT_DIR = "vector_db"
CHUNK_SIZE = 2000
EMBEDDING_MODEL = "thenlper/gte-large"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def extract_text_from_pdf(file_path):
    try:
        pdf = pdfium.PdfDocument(file_path)
        text = ""
        for i in range(len(pdf)):
            text += pdf[i].get_textpage().get_text_bounded() or ""
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
        doc = DocxDocument(file_path)
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
    with Pool(processes=20) as pool:
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
    embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True)  # Safe VRAM
    elapsed = time.time() - start_time
    print(f"Embedding generation took {elapsed:.2f} seconds")
    return embeddings

def store_in_faiss(embeddings, metadata, chunks, output_dir):
    start_time = time.time()
    embeddings = np.array(embeddings).astype("float32")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build HNSW index
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings)
    
    # Use LangChain Document
    docs = [Document(page_content=chunk, metadata={"source": m[0], "chunk_id": m[1]}) for chunk, m in zip(chunks, metadata)]
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
    
    # Wrap in LangChain FAISS
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: i for i in range(len(embeddings))},
    )
    
    vector_store.save_local(output_dir, "faiss_index.bin")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(output_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f)
    elapsed = time.time() - start_time
    print(f"Faiss storage took {elapsed:.2f} seconds")
    print(f"Stored {len(embeddings)} embeddings in {output_dir}")

def update_vector_store(data_dir, output_dir):
    total_start = time.time()
    state_file = os.path.join(output_dir, "state.json")
    chunks_file = os.path.join(output_dir, "chunks.json")
    metadata_file = os.path.join(output_dir, "metadata.json")
    
    prev_state = {}
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            prev_state = json.load(f)

    current_state = {}
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
    for fp in file_paths:
        if fp.lower().endswith((".pdf", ".txt", ".md", ".docx")):
            current_state[fp] = os.path.getmtime(fp)

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

    to_process = [fp for fp in file_paths if fp not in prev_state or prev_state[fp] != current_state[fp]]
    if not to_process:
        print("No changes detected.")
        return

    if os.path.exists(chunks_file) and os.path.exists(metadata_file):
        with open(chunks_file, "r") as f:
            old_chunks = json.load(f)
        with open(metadata_file, "r") as f:
            old_metadata = json.load(f)
    else:
        old_chunks, old_metadata = [], []

    new_chunks, new_metadata = process_files(to_process)
    if not new_chunks:
        print("No new/changed data to process.")
        return

    changed_files = set(fp for fp in to_process)
    keep_chunks = [c for c, m in zip(old_chunks, old_metadata) if m[0] not in changed_files]
    keep_metadata = [m for m in old_metadata if m[0] not in changed_files]
    
    all_chunks = keep_chunks + new_chunks
    all_metadata = keep_metadata + new_metadata
    embeddings = generate_embeddings(all_chunks)
    store_in_faiss(embeddings, all_metadata, all_chunks, output_dir)
    
    with open(state_file, "w") as f:
        json.dump(current_state, f)
    
    print(f"Updated with {len(new_chunks)} new/changed chunks, total {len(all_chunks)} in {time.time() - total_start:.2f} seconds")

def main():
    print("Starting text extraction, embedding, and storage process...")
    update_vector_store(DATA_DIR, OUTPUT_DIR)
    print("Process complete!")

if __name__ == "__main__":
    main()