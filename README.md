# HAL: Highly Adaptable Learning AI

HAL is your precision-crafted AI wingman, built to sling quality answers (50-100 tokens) from a mountain of tech docs—PDFs, DOCX, TXT—in a flash. Powered by an RTX 4080 (16GB VRAM, 14.1GB peak), i9-13900KF (20-core shredder), and 128GB RAM, HAL devours GBs (421MB in 232s) and retrieves in <0.1s. Packed with 14,658 chunks of compiler, algorithm, and data system gold, it’s your go-to for razor-sharp dev insights—fast.

## Features
- **Blazing Ingestion**: 421MB (28 PDFs) in 232s—3.42s extraction, 227s embeddings, 1.63s HNSW FAISS build.
- **Snappy Retrieval**: <0.1s via HNSW—queries land in ~0.7-1.3s total.
- **Session Memory**: Tracks your name or last ask—resets on restart, keeps it lean.
- **Scalable AF**: From 444 chunks (8s) to 14k (232s)—GBs? Bring it on.

## Tech Stack
- **`build_db.py`**: Ingests with `thenlper/gte-large` (768D embeddings), HNSW FAISS—14,658 chunks in 232s.
- **`hal.py`**: Runs Llama-3.2-3B via vLLM, RAG with in-run history—answers in ~1s.
- **Hardware**: RTX 4080 (14.1GB peak), i9-13900KF, 128GB RAM—speed demons unleashed.

## Current State
- **Data**: 421MB, 28 PDFs
- **Perf**: 232s build (14k chunks), ~1s queries—scales to GBs with <0.1s retrieval.
- **Next**: Crank it to 1GB+—VRAM’s near 16GB max, might shard for bigger hauls.

## Hot Deets
- **GPU Flex**: 4080 peaked at 14.1GB—no sweat, ready for more.
- **CPU Shred**: i9’s 20 cores ripped 14k chunks in 3.42s—extraction’s a breeze.
- **Books**: 28 dev bibles—HAL’s a walking PhD in compilers and data systems.

## License
Proprietary - see [LICENSE](LICENSE) for details. All rights reserved by Jarrod Connolly.

Thanks to Grok (xAI) for code collaboration and ideas.
