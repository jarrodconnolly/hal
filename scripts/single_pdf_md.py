"""Convert a single PDF to Markdown using pymupdf4llm for HAL's ingestion testing.

Takes a hardcoded Node.js documentation PDF, extracts its content as Markdown with code blocks
skipped, and writes the result to a local file (Node.js.text.md). Used to test or debug PDF
conversion before full processing in build_db.py's pipeline.
"""
import pymupdf4llm
import pathlib
md_text = pymupdf4llm.to_markdown("/home/totally/data/Node.js/Node.js v22.14.0 Documentation.pdf", ignore_code=True)
pathlib.Path("Node.js.text.md").write_bytes(md_text.encode())
