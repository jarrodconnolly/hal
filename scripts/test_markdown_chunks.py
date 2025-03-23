"""Test Markdown chunk extraction from PDFs for HAL's ingestion pipeline.

Converts PDFs in a directory to Markdown using pymupdf4llm, splits them into prose chunks
(similar to build_db.py), and analyzes chunk sizes and skipped sections. Processes all PDFs
in BASE_DIR recursively, caching results, and prints stats per file plus a summary. Used to
debug chunking logic and section filtering before full vector store ingestion.
"""
import os
import re
import time
from statistics import mean

import pymupdf4llm

# Config
BASE_DIR = os.path.expanduser("~/data")  # Adjust if PDFs are elsewhere
MIN_LENGTH = 100
CACHE_DIR = "cache"


def extract_markdown_paragraphs(pdf_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    pdf_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    md_file = os.path.join(CACHE_DIR, pdf_name)

    if os.path.exists(md_file):
        print(f"Loading cached Markdown from {md_file}...")
        start_time = time.time()
        with open(md_file, "r") as f:
            md_text = f.read()
        md_text = md_text.replace("\n-----\n", " ")
        print(f"Loaded cache in {time.time() - start_time:.2f} seconds")
    else:
        print(f"Converting {pdf_path} to Markdown...")
        start_time = time.time()
        md_text = pymupdf4llm.to_markdown(pdf_path)
        print(f"pymupdf4llm.to_markdown {time.time() - start_time:.2f} seconds")
        with open(md_file, "w") as f:
            f.write(md_text)

    chunks = []
    skips = []
    start_time = time.time()
    sections = re.split(r"\n#{1,2} ?", md_text)[1:]  # # or ## only
    print(
        f"First 5 sections for {pdf_path}: {[s.split('\n')[0][:50] for s in sections[:5]]}"
    )
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
    for section in sections:
        section_title = section.split("\n")[0].strip()
        #if any(keyword in section_title.lower() for keyword in skip_titles):
        if section_title.lower() in skip_titles:
            print(f"Skipping section: #{section_title}")
            skips.append(section_title)
            continue

        paras = section.split("\n\n")
        for para in paras:
            para = para.strip()
            if len(para) >= MIN_LENGTH and not para.startswith("- "):
                while len(para) >= MIN_LENGTH:
                    if len(para) > 1000:
                        mid = para[:1000].rfind(" ") or 1000
                        chunks.append(
                            {"text": para[:mid], "section": f"# {section_title}"}
                        )
                        para = para[mid:].strip()
                    else:
                        if chunks and len(chunks[-1]["text"]) < 600:
                            new_text = chunks[-1]["text"] + " " + para
                            if len(new_text) > 1000:
                                chunks.append(
                                    {"text": para, "section": f"# {section_title}"}
                                )
                            else:
                                chunks[-1]["text"] = new_text
                        else:
                            chunks.append(
                                {"text": para, "section": f"# {section_title}"}
                            )
                        break

    print(
        f"Split {pdf_path} into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds"
    )
    return chunks, skips


def analyze_chunks(chunks, skips, pdf_path):
    print(f"\nProcessed {pdf_path}:")
    if not chunks:
        print("No chunks found!")
    else:
        chunk_sizes = [len(chunk["text"]) for chunk in chunks]
        print(f"Chunk Count: {len(chunks)}")
        print(f"Avg Size: {mean(chunk_sizes):.1f} chars")
        print(f"Min Size: {min(chunk_sizes)} chars")
        print(f"Max Size: {max(chunk_sizes)} chars")
    if skips:
        print(f"Skipped Sections ({len(skips)}):")
        for skip in skips[:10]:  # Limit to 10 for brevity
            print(f"  - {skip}")
        if len(skips) > 10:
            print(f"  ...and {len(skips) - 10} more")
    print()


def main():
    print(f"Scanning PDFs in {BASE_DIR}...")
    total_chunks = 0
    all_skips = {}

    # Walk through BASE_DIR and subdirs like build_db.py
    for root, _, files in os.walk(BASE_DIR):
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(root, pdf_file)
            print(f"\nLoading PDF: {pdf_path}")
            chunks, skips = extract_markdown_paragraphs(pdf_path)
            total_chunks += len(chunks)
            if skips:
                all_skips[pdf_file] = skips
            analyze_chunks(chunks, skips, pdf_path)

    print("\nSummary:")
    print(f"Total Chunks: {total_chunks}")
    print(f"Files with Skips: {len(all_skips)}")
    if all_skips:
        print("Files with Skipped Sections:")
        for pdf, skips in all_skips.items():
            print(f"  {pdf}: {len(skips)} skips")


if __name__ == "__main__":
    main()
