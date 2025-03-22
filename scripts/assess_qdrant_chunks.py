"""Assess the quality of chunks stored in Qdrant for HAL's vector database.

Fetches all chunks from the hal_docs collection, scores them using spaCy-based heuristics
(from build_db.py), and generates statistics on score distribution. Logs low-scoring chunks
(<0.3) to a file and samples mid (0.3-0.7) and high (>0.7) chunks for inspection. Used to
debug chunk quality post-ingestion and tune scoring thresholds.
"""
import os
import spacy
from qdrant_client import QdrantClient
from tqdm import tqdm
import time

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DOCS_COLLECTION = "hal_docs"  # Matches your run
BATCH_SIZE = 1000

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000

def score_chunk(chunk_text):
    doc = nlp(chunk_text)
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

def assess_qdrant_collection():
    start_time = time.time()
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, timeout=300)
    
    collection_info = client.get_collection(DOCS_COLLECTION)
    total_points = collection_info.points_count
    print(f"Assessing {total_points} chunks in {DOCS_COLLECTION}")

    scores = []
    low_chunks = []  # Full log of <0.3
    samples = {"mid": [], "high": []}  # Keep 10 each for mid/high
    offset = None
    total_processed = 0
    
    with tqdm(total=total_points, desc="Scoring chunks", disable=True) as pbar:
        while True:
            scroll_result = client.scroll(
                collection_name=DOCS_COLLECTION,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = scroll_result
            if not points:
                break
            
            for point in points:
                chunk_text = point.payload["content"]
                source = point.payload["source"]
                score = score_chunk(chunk_text)
                scores.append(score)
                
                if score < 0.3:
                    low_chunks.append((chunk_text, score, source))
                elif 0.3 <= score <= 0.7 and len(samples["mid"]) < 10:
                    samples["mid"].append((chunk_text[:50], score, source))
                elif score > 0.7 and len(samples["high"]) < 10:
                    samples["high"].append((chunk_text[:50], score, source))
                
                total_processed += 1
                pbar.update(1)
            
            offset = next_offset
            if total_processed >= total_points:
                break

    # Stats
    avg_score = sum(scores) / len(scores)
    low = len([s for s in scores if s < 0.3])
    mid = len([s for s in scores if 0.3 <= s <= 0.7])
    high = len([s for s in scores if s > 0.7])
    print("\nStats:")
    print(f"  Total Chunks: {total_points}")
    print(f"  Processed: {total_processed}")
    print(f"  Avg Score: {avg_score:.2f}")
    print(f"  Low (<0.3): {low} ({low/total_points*100:.1f}%)")
    print(f"  Mid (0.3-0.7): {mid} ({mid/total_points*100:.1f}%)")
    print(f"  High (>0.7): {high} ({high/total_points*100:.1f}%)")

    # Mid/High Samples
    print("\nMid Scoring Samples (0.3-0.7):")
    for text, score, source in samples["mid"]:
        print(f"  {score:.2f} - {os.path.basename(source)}: {text}")
    print("\nHigh Scoring Samples (>0.7):")
    for text, score, source in samples["high"]:
        print(f"  {score:.2f} - {os.path.basename(source)}: {text}")

    # Log all low chunks
    print(f"\nWriting {len(low_chunks)} low-scoring chunks to low_score_chunks.txt...")
    with open("low_score_chunks.txt", "w") as f:
        for chunk_text, score, source in low_chunks:
            f.write(f"Score: {score:.2f}\nSource: {os.path.basename(source)}\nText ({len(chunk_text)} chars):\n{chunk_text}\n{'-'*50}\n")

    elapsed = time.time() - start_time
    print(f"Assessment took {elapsed:.2f} seconds")

def main():
    print("Starting Qdrant chunk assessment...")
    assess_qdrant_collection()
    print("Assessment complete!")

if __name__ == "__main__":
    main()