# HAL: Highly Adaptable Learning AI

HAL is your precision-crafted AI wingman, delivering sharp answers (under 100 tokens) from a tech doc empire—PDFs, DOCX, TXT, MD—in a flash, styled like *WarGames*’ IMSAI 8080 screen. Powered by an RTX 4080 (16GB VRAM, 14.1GB peak), i9-13900KF (20-core shredder), and 128GB RAM, HAL ingests 421MB in 242s, retrieves in 0.05s, and streams replies live in ~1s. With 14,658 chunks of compiler, algorithm, and software architecture gold in Qdrant, it’s your go-to for dev insights—fast, fierce, and retro-cool.

## Features
- **Blazing Ingestion**: 421MB (28 files) in 242s—4.14s extraction, 227s embeddings, ~3s Qdrant upsert.
- **Snappy Retrieval**: 0.05s via Qdrant HNSW—queries stream in ~0.5-1.3s total with Llama-3.2-3B.
- **Live Streaming**: Answers trickle in real-time—no wait, just *WarGames*-style text flow.
- **Session Memory**: FAISS tracks history in-run—resets on restart, keeps it lean.
- **Scalable AF**: From 444 chunks (8s) to 14,658 (242s)—GBs? HAL’s got room to grow.

## Tech Stack
- **`build_db.py`**: Extracts with `pypdfium2`, `python-docx`, embeds via `thenlper/gte-large` (1024D), stores in Qdrant HNSW—14,658 chunks in 242s.
- **`hal.py`**: FastAPI API, talks to vLLM at `localhost:8000` (Llama-3.2-3B), RAG with Qdrant + FAISS—serves `/query` and `/query_stream`.
- **`hal_ui.py`**: `Textual` UI—whiter-cyan (#E0FFFF) text, `█` cursor, no border, streams replies, shows live Qdrant chunk count + timings.
- **`custom_vllm.py`**: Thin client for vLLM endpoint—keeps `hal.py` lean.
- **`start_vllm.sh`**: Fires vLLM server—`float16`, 85% GPU, 8192 max len.
- **Hardware**: RTX 4080 (14.1GB peak), i9-13900KF, 128GB RAM—speed demons unleashed.
- **Env**: Python 3.12.9, WSL Ubuntu 22.04, `uv` deps (`fastapi`, `uvicorn`, `httpx`, etc.).

## Current State
- **Data**: 421MB, 28 files—compilers, algorithms, software arch.
- **Perf**: 242s build (14,658 chunks), 0.05s retrieval, ~1s total queries—streaming feels instant.
- **UI**: *WarGames* look—Q\nA\n\n format, no “HAL:”, live chunk count, timings from API.
- **Setup**: `start_vllm.sh` → `hal.py` (API) → `hal_ui.py` (UI)—plug and play.

## How to Run
1. **Setup**: `uv sync` in `~/code/hal/.venv`—grabs deps from `pyproject.toml`.
2. **vLLM**: `./start_vllm.sh`—runs Llama-3.2-3B at `localhost:8000`.
3. **DB**: `python build_db.py`—loads docs into Qdrant (242s first run, seconds if unchanged).
4. **API**: `python hal.py`—FastAPI at `localhost:8001`.
5. **UI**: `python hal_ui.py`—type queries, watch HAL stream answers.

## Hot Deets
- **GPU Flex**: 4080 peaks at 14.1GB—handles `gte-large` embeddings like a champ.
- **CPU Shred**: i9’s 20 cores tear through extraction in 4.14s—multiprocessing FTW.
- **Books**: 28 tech bibles—HAL’s your PhD-level compiler and systems guru.
- **UI Glow**: Whiter-cyan text, `█` cursor, no border—pure 80s terminal soul.

## Optional Tweaks
- **Cache Embeddings**: Pre-compute `gte-large` for hot queries—drop retrieval to ~0.01s.
- **Lite Embedder**: `all-MiniLM-L6-v2` (384D)—faster (~10-20ms vs. 40-50ms), less precision.
- **Sharding**: For 1GB+, shard Qdrant—keeps VRAM and RAM comfy.

## License
Proprietary - see [LICENSE](LICENSE) for details. All rights reserved by Jarrod Connolly.

Thanks to Grok (xAI) for code collaboration and vibes.

## Other License Attribution

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
