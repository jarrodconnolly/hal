# HAL: Highly Adaptable Learning AI

HAL is a precision-engineered AI assistant designed to deliver concise, accurate answers from a corpus of technical documents—PDFs, DOCX, TXT, and Markdown files. Built for speed and efficiency, HAL leverages Retrieval-Augmented Generation (RAG) to provide real-time responses, styled with a retro *WarGames*-inspired terminal aesthetic. It’s optimized for developers seeking fast, reliable insights from complex technical literature.

## Features

- **Rapid Document Ingestion**: Processes technical documents into a searchable knowledge base with high efficiency.
- **Real-Time Query Streaming**: Delivers answers incrementally as they’re generated, minimizing wait times.
- **Session-Based Memory**: Retains conversation history within a session using FAISS, resetting on restart for lightweight operation.
- **Scalable Architecture**: Handles growing datasets, from small collections to gigabytes of content, with consistent performance.
- **Retro Terminal Interface**: Features a minimalist, *WarGames*-style UI with whiter-cyan text and a live cursor.

## Technology Stack

- **`build_db.py`**: Extracts text using `pypdfium2` and `python-docx`, generates embeddings with `thenlper/gte-large` (1024-dimensional), and stores them in Qdrant with HNSW indexing.
- **`hal.py`**: A FastAPI-based API serving a single `/query_stream` endpoint, interfacing with vLLM (`meta-llama/Llama-3.2-3B-Instruct`) for generation and integrating RAG via Qdrant and FAISS.
- **`hal_ui.py`**: A `Textual`-powered UI displaying responses in whiter-cyan (#E0FFFF) text, with a `█` cursor and live metrics (chunk count, timings).
- **`start_vllm.sh`**: Launches the vLLM server in `float16` mode, utilizing up to 85% of GPU capacity with an 8192-token maximum length.
- **Hardware**: Powered by an NVIDIA RTX 4080 (16GB VRAM), Intel i9-13900KF (20 cores), and 128GB RAM.
- **Environment**: Runs on Python 3.12.9 under WSL Ubuntu 22.04, with dependencies managed via `uv`.

## Current State

- **Data**: Comprises 28 technical documents totaling 421MB, covering compilers, algorithms, and software architecture.
- **Operation**: Built as a single-user, public GitHub project, with a streamlined workflow from document ingestion to query response.
- **Interface**: Presents a clean, question-and-answer format without prefixes, displaying live Qdrant chunk counts and performance timings.

## Getting Started

1. **Install Dependencies**: Run `uv sync` in `~/code/hal/.venv` to install requirements from `pyproject.toml`.
2. **Start vLLM Server**: Execute `./start_vllm.sh` to launch Llama-3.2-3B at `localhost:8000`.
3. **Build Database**: Use `python build_db.py` to ingest documents into Qdrant (initial run takes ~242 seconds; subsequent runs are faster if unchanged).
4. **Launch API**: Start `python hal.py` to serve the FastAPI endpoint at `localhost:8001`.
5. **Open UI**: Run `python hal_ui.py` to interact with HAL via the terminal interface.

## Performance Highlights

- **Ingestion Speed**: Processes 421MB (14,658 chunks) in 242 seconds—4.14s extraction, 227s embeddings, ~3s Qdrant upsert.
- **Retrieval Latency**: Achieves 0.05s retrieval using Qdrant’s HNSW index.
- **Query Response**: Streams answers in ~0.5-1.3s total, with generation peaking at 14.1GB VRAM on an RTX 4080.
- **Scalability**: Scales from 444 chunks (8s) to 14,658 chunks (242s), with capacity for larger datasets.

## License

Proprietary - see [LICENSE](LICENSE) for details. All rights reserved by Jarrod Connolly.

Thanks to Grok (xAI) for code collaboration and development support.

## Attribution

### General Text Embeddings (GTE) Model (`thenlper/gte-large`)
```
@article{li2023towards,
  title={Towards general text embeddings with multi-stage contrastive learning},
  author={Li, Zehan and Zhang, Xin and Zhang, Yanzhao and Long, Dingkun and Xie, Pengjun and Zhang, Meishan},
  journal={arXiv preprint arXiv:2308.03281},
  year={2023}
}
```

### Llama 3.2 (`meta-llama/Llama-3.2-3B-Instruct`)
Built with Llama - see [LICENSE_LLAMA](LICENSE_LLAMA)