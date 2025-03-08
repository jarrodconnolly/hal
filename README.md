# HAL: Highly Adaptable Learning AI

HAL is your precision-crafted AI wingman, built to deliver sharp answers (50-100 tokens) from a tech doc empire—PDFs, DOCX, TXT—in a heartbeat. Powered by an RTX 4080 (16GB VRAM, 14.1GB peak), i9-13900KF (20-core shredder), and 128GB RAM, HAL ingests 421MB in 242s and retrieves in 0.05s. With 14,658 chunks of compiler, algorithm, and software architecture gold in Qdrant, it’s your go-to for dev insights—fast and fierce.

## Features
- **Blazing Ingestion**: 421MB (28 PDFs) in 242s—4.14s extraction, 227s embeddings, ~3s Qdrant upsert.
- **Snappy Retrieval**: 0.05s via Qdrant HNSW—queries land in ~0.5-1.3s total with Llama-3.2-3B.
- **Session Memory**: Tracks history in-run via FAISS—resets on restart, keeps it lean.
- **Scalable AF**: From 444 chunks (8s) to 14,658 (242s)—GBs? HAL’s ready to roll.

## Tech Stack
- **`build_db.py`**: Ingests with `thenlper/gte-large` (1024D embeddings), Qdrant HNSW—14,658 chunks in 242s.
- **`hal.py`**: Runs Llama-3.2-3B via vLLM, RAG with Qdrant + FAISS history—answers in ~1s.
- **Hardware**: RTX 4080 (14.1GB peak), i9-13900KF, 128GB RAM—speed demons unleashed.

## Current State
- **Data**: 421MB, 28 PDFs—compilers, algorithms, software arch.
- **Perf**: 242s build (14,658 chunks), 0.05s retrieval, ~1.32s total queries—scales to GBs with room to spare.
- **Next**: Push to 1GB+—VRAM’s near max, might shard or tweak for bigger hauls.

## Hot Deets
- **GPU Flex**: 4080 peaks at 14.1GB—handles `gte-large` embeddings like a champ.
- **CPU Shred**: i9’s 20 cores tear through extraction in 4.14s—multiprocessing FTW.
- **Books**: 28 tech bibles—HAL’s your PhD-level compiler and systems guru.

## Optional Tweaks
- **Cache Embeddings**: Pre-compute `gte-large` embeddings for common queries—drop `qdrant_time` from 0.05s to ~0.01s (Qdrant’s 3-4ms raw).
- **Lite Embedder**: Swap to `all-MiniLM-L6-v2` (384D)—faster embedding (~10-20ms vs. 40-50ms), less precision.
- **Sharding**: For 1GB+, shard Qdrant collections—keeps VRAM and RAM comfy.

## License
Proprietary - see [LICENSE](LICENSE) for details. All rights reserved by Jarrod Connolly.

Thanks to Grok (xAI) for code collaboration and ideas.

## Dependancies

### General Text Embeddings (GTE) model (thenlper/gte-large)
```
@article{li2023towards,
  title={Towards general text embeddings with multi-stage contrastive learning},
  author={Li, Zehan and Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Pengjun and Zhang, Meishan},
  journal={arXiv preprint arXiv:2308.03281},
  year={2023}
}
```

### Llama 3.2 (meta-llama/Llama-3.2-3B-Instruct)

Built with Llama - see [LICENSE_LLAMA](LICENSE_LLAMA)
