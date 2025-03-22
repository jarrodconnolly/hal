import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mistune
import mistune.renderers
import mistune.renderers.markdown
import pymupdf4llm
import spacy
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

from .config import (
    DOCS_COLLECTION,
    EMBEDDING_MODEL,
    HNSW_M,
    QDRANT_HOST,
    QDRANT_PORT,
)
from .logging_config import configure_logging

# Logging Configuration
logger = configure_logging()

DATA_DIR = os.path.expanduser("~/data")
OUTPUT_DIR = "vector_db"
CACHE_DIR = "cache"
MIN_LENGTH = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info("Encoding device", device)
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000


# Mistune Renderer for prose-only extraction
class ProseRenderer(mistune.renderers.markdown.MarkdownRenderer):
    def block_code(self, code, lang=None):
        return ""  # Skip code blocks

    def table(self, header, body):
        return ""  # Skip tables

    def paragraph(self, token, state):
        return self.render_children(token, state) + "\n\n"  # Keep prose


mistune_renderer = ProseRenderer()
mistune_md = mistune.Markdown(renderer=mistune_renderer)


def generate_embeddings(chunks):
    """Generate embeddings for a list of chunk texts."""
    start_time = time.time()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(
        texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    elapsed = time.time() - start_time
    logger.info(f"Embedding generation took {elapsed:.2f} seconds")
    return embeddings


def process_section(args):
    """Process a Markdown section into prose chunks.

    Args:
        args: A tuple of (section, file_path, skip_titles), where:
            - section: The raw Markdown section text to chunk.
            - file_path: The source PDF file path for metadata.
            - skip_titles: Set of section titles to skip (e.g., 'contents').

    Returns:
        A list of chunk dicts with 'text', 'section', and 'source' keys.
    """
    section, file_path, skip_titles = args
    chunks = []

    section_title = section.split("\n")[0].strip()
    if section_title.lower() in skip_titles:
        return chunks  # Skip silently

    prose_text = mistune_md(section).strip()
    if not prose_text:
        return chunks

    paras = prose_text.split("\n\n")
    for para in paras:
        para = para.strip()
        if len(para) >= MIN_LENGTH and not para.startswith("- "):
            while len(para) >= MIN_LENGTH:
                if len(para) > 1000:
                    mid = para[:1000].rfind(" ") or 1000
                    chunks.append(
                        {
                            "text": para[:mid],
                            "section": f"# {section_title}",
                            "source": file_path,
                        }
                    )
                    para = para[mid:].strip()
                else:
                    if chunks and len(chunks[-1]["text"]) < 600:
                        new_text = chunks[-1]["text"] + " " + para
                        if len(new_text) > 1000:
                            chunks.append(
                                {
                                    "text": para,
                                    "section": f"# {section_title}",
                                    "source": file_path,
                                }
                            )
                        else:
                            chunks[-1]["text"] = new_text
                    else:
                        chunks.append(
                            {
                                "text": para,
                                "section": f"# {section_title}",
                                "source": file_path,
                            }
                        )
                    break
    return chunks


def extract_markdown_paragraphs_parallel(file_path, stats, max_workers=20):
    """Extract prose chunks from a PDF's cached Markdown using parallel processing.

    Args:
        file_path: Path to the PDF file to process.
        stats: Dict to update with chunk statistics ('total_size', 'count', 'min_size', 'max_size').
        max_workers: Maximum number of parallel processes (default: 20).

    Returns:
        A list of chunk dicts with 'text', 'section', and 'source' keys, extracted from
        Markdown sections using Mistune for prose-only content.
    """
    logger.info(f"Processing {file_path} (parallel sections)...")
    start_time = time.time()

    # Cache setup
    os.makedirs(CACHE_DIR, exist_ok=True)
    pdf_name = os.path.basename(file_path).replace(".pdf", ".md")
    md_file = os.path.join(CACHE_DIR, pdf_name)

    if os.path.exists(md_file):
        logger.info(f"Loading cached Markdown from {md_file}...")
        with open(md_file, "r") as f:
            md_text = f.read()
    else:
        logger.info(f"Converting {file_path} to Markdown...")
        md_text = pymupdf4llm.to_markdown(file_path)
        with open(md_file, "w") as f:
            f.write(md_text)

    # Strip page breaks
    md_text = md_text.replace("\n-----\n", " ")
    logger.info(f"Loaded/stripped Markdown in {time.time() - start_time:.2f} seconds")

    # Split into sections—still using headings as split points
    start_time = time.time()
    sections = re.split(r"\n#{1,2} ?", md_text)[1:]  # Skip preamble
    skip_titles = {
        "contents",
        "preface",
        "index",
        "appendix",
        "acknowledgments",
        "bibliography",
        "about the authors",
        "chapter notes",
    }

    # Parallel chunking with Mistune
    all_chunks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [(section, file_path, skip_titles) for section in sections]
        results = executor.map(process_section, args)
        all_chunks = [
            chunk for chunk_list in results for chunk in chunk_list
        ]  # Flatten

    # Compute stats after chunking
    if all_chunks:
        chunk_sizes = [len(chunk["text"]) for chunk in all_chunks]
        stats["total_size"] += sum(chunk_sizes)
        stats["count"] += len(chunk_sizes)
        stats["min_size"] = min(stats["min_size"], min(chunk_sizes))
        stats["max_size"] = max(stats["max_size"], max(chunk_sizes))

    avg_size = stats["total_size"] / stats["count"] if stats["count"] else 0
    logger.info(
        f"Split {file_path} into {len(all_chunks)} chunks in {time.time() - start_time:.2f} seconds"
    )
    logger.info(
        f"Current Stats - Count: {stats['count']}, Avg Size: {avg_size:.1f} chars, "
        f"Min Size: {stats['min_size']} chars, Max Size: {stats['max_size']} chars"
    )
    return all_chunks


def score_chunk(chunk_text):
    """Score a chunk's quality for RAG based on linguistic features.

    Args:
        chunk_text: The text content of the chunk to evaluate.

    Returns:
        A float between 0.0 and 1.0, where higher scores indicate richer, more
        meaningful content suitable for retrieval-augmented generation.
    """
    try:
        doc = nlp(chunk_text)
    except Exception as e:
        logger.warning(f"spaCy failed on chunk: {str(e)} - returning 0.0 score")
        return 0.0
    score = 0.0

    # Sentences—any prose gets a base
    sentences = list(doc.sents)
    if len(sentences) >= 3:
        score += 0.3  # Full paragraph
    elif len(sentences) == 2:
        score += 0.2  # Decent thought
    elif len(sentences) == 1:
        score += 0.1  # At least something

    # Words—content density
    words = len([token for token in doc if token.is_alpha])
    if words >= 50:
        score += 0.3  # Meaty
    elif words >= 20:
        score += 0.2  # Solid
    elif words >= 10:
        score += 0.1  # Minimal but okay

    # Nouns/Verbs—semantic richness
    nouns = len([token for token in doc if token.pos_ in ["NOUN", "PROPN"]])
    verbs = len([token for token in doc if token.pos_ == "VERB"])
    if nouns >= 10 and verbs >= 5:
        score += 0.4  # Rich narrative
    elif nouns >= 5 and verbs >= 2:
        score += 0.3  # Good action
    elif nouns >= 2:
        score += 0.2  # Basic meaning

    # Length boost—reward fuller chunks
    if len(chunk_text) > 600:
        score += 0.1  # RAG loves ~620 avg

    # Penalties—catch junk
    punct = len([token for token in doc if token.is_punct])
    if punct / max(words, 1) > 0.3:  # Over-punctuated mess
        score -= 0.1
    if "|" in chunk_text and words < 20:  # Small pipe-heavy = table-ish
        score = min(score, 0.3)  # Cap, not kill
    if len(chunk_text) < 100 and words < 10:  # Tiny and sparse
        score = min(score, 0.2)  # True junk

    return min(max(score, 0.0), 1.0)


def process_files(file_paths, stats):
    start_time = time.time()
    all_chunks = []
    for file_path in file_paths:
        if file_path.lower().endswith(".pdf"):
            chunks = extract_markdown_paragraphs_parallel(file_path, stats)
            all_chunks.extend(chunks)
            logger.info(f"Processed {os.path.basename(file_path)}: {len(chunks)} chunks")

    # Filter low-score chunks
    filtered_chunks = [
        chunk for chunk in all_chunks if score_chunk(chunk["text"]) >= 0.3
    ]
    logger.info(f"Filtered {len(all_chunks) - len(filtered_chunks)} chunks scoring <0.3")
    all_chunks = filtered_chunks

    elapsed = time.time() - start_time
    logger.info(f"Text extraction and chunking took {elapsed:.2f} seconds")
    return all_chunks


def update_vector_store(data_dir, output_dir):
    """Update the Qdrant vector store with embeddings from PDF files.

    Args:
        data_dir: Directory path containing PDF files to process.
        output_dir: Directory path to store state.json and cache.

    Processes new or modified PDFs, extracts prose chunks, generates embeddings,
    and upserts them into Qdrant, maintaining an incremental state.
    """
    total_start = time.time()
    state_file = os.path.join(output_dir, "state.json")

    # Load previous state
    prev_state = {}
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            prev_state = json.load(f)
    else:
        client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, timeout=300)
        if client.collection_exists(DOCS_COLLECTION):
            logger.info("No state.json found—resetting collection.")
            client.delete_collection(DOCS_COLLECTION)

    # Build current state
    current_state = {}
    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(data_dir)
        for file in files
        if file.lower().endswith(".pdf")
    ]
    for fp in file_paths:
        current_state[fp] = {"mtime": os.path.getmtime(fp), "chunk_ids": []}

    # Connect to Qdrant
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, timeout=300)

    # Create collection if missing
    if not client.collection_exists(DOCS_COLLECTION):
        logger.info("Creating new collection.")
        client.create_collection(
            collection_name=DOCS_COLLECTION,
            vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            hnsw_config=rest.HnswConfigDiff(m=HNSW_M),
        )

    # Handle deletions
    deleted_files = [fp for fp in prev_state if fp not in current_state]
    if deleted_files:
        for fp in deleted_files:
            chunk_ids = prev_state[fp].get("chunk_ids", [])
            if chunk_ids:
                logger.info(f"Deleting {len(chunk_ids)} chunks for {fp}")
                client.delete(
                    collection_name=DOCS_COLLECTION,
                    points_selector=rest.PointIdsList(points=chunk_ids),
                )

    if not file_paths:
        logger.info("No files to process.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(current_state, f)
        return

    # Process new/changed files with stats
    stats = {"total_size": 0, "count": 0, "min_size": float("inf"), "max_size": 0}
    to_process = [
        fp
        for fp in file_paths
        if fp not in prev_state or prev_state[fp]["mtime"] != current_state[fp]["mtime"]
    ]
    if not to_process:
        logger.info("No changes detected.")
        return

    new_chunks = process_files(to_process, stats)
    if not new_chunks:
        logger.info("No new/changed data to process.")
        return

    # Final stats (post-filter)
    stats["count"] = len(new_chunks)
    stats["total_size"] = sum(len(chunk["text"]) for chunk in new_chunks)
    stats["min_size"] = (
        min(len(chunk["text"]) for chunk in new_chunks) if new_chunks else float("inf")
    )
    stats["max_size"] = (
        max(len(chunk["text"]) for chunk in new_chunks) if new_chunks else 0
    )
    final_avg = stats["total_size"] / stats["count"] if stats["count"] else 0
    logger.info(
        f"Final Stats - Count: {stats['count']}, Avg Size: {final_avg:.1f} chars, "
        f"Min Size: {stats['min_size']} chars, Max Size: {stats['max_size']} chars"
    )

    embeddings = generate_embeddings(new_chunks)
    # Upsert to Qdrant
    points = []
    for idx, (embedding, chunk) in enumerate(zip(embeddings, new_chunks)):
        point_id = str(uuid.uuid4())  # Unique ID, no race risk
        points.append(
            rest.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "content": chunk["text"],
                },
            )
        )
        current_state[chunk["source"]]["chunk_ids"].append(point_id)

        batch_size = 1000
        failed_batches = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            logger.info(
                f"Upserting batch {i // batch_size + 1} of {(len(points) - 1) // batch_size + 1} ({len(batch)} points)"
            )
            try:
                client.upsert(collection_name=DOCS_COLLECTION, points=batch)
            except Exception as e:
                failed_batches += 1
                logger.error(
                    f"Upsert failed for batch {i // batch_size + 1}: {str(e)}"
                )
                if hasattr(e, "response") and hasattr(e.response, "content"):
                    logger.error(f"Response content: {e.response.content}")
                continue  # Skip to next batch
        if failed_batches:
            logger.warning(
                f"Completed with {failed_batches} failed batches out of {(len(points) - 1) // batch_size + 1}"
            )
        else:
            logger.info("All batches upserted successfully")

    # Save state
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(current_state, f)
    logger.info(
        f"Updated with {len(new_chunks)} new/changed chunks in {time.time() - total_start:.2f} seconds"
    )


def main():
    logger.info("Starting text extraction, embedding, and storage process...")
    update_vector_store(DATA_DIR, OUTPUT_DIR)
    logger.info("Process complete!")


if __name__ == "__main__":
    main()
