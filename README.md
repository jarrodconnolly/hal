  # HAL - Highly Adaptable Learning AI

  HAL is a Python-based AI project for processing text data (PDFs, TXT) into a Faiss vector database and querying it with embeddings. Built for scale (1-5 GB+), it’s ready for LLM integration and future speech capabilities. Inspired by *2001: A Space Odyssey*.

  ## Setup

  Run HAL on Ubuntu (WSL2 or native) with Python 3.10, leveraging a GPU (e.g., RTX 4080).

  1. **Clone the Repo**
     ```bash
     git clone https://github.com/jarrodconnolly/hal.git
     cd hal
     ```

  1. **Install UV (Package Manager)**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
     source ~/.bashrc
     ```

  1. **Create Virtual Environment**
     ```bash
     uv venv --python 3.10
     source .venv/bin/activate
     ```

  1. **Set Hugging Face Token**  
     - Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  
     ```bash
     export HF_TOKEN="your-token-here"
     ```

  1. **Install Dependencies**
     ```bash
     uv pip install .
     ```

  1. **Add Data**
     - Place PDFs or TXT files in `~/data/` (not tracked by Git).

  1. **Build the Database**
     ```bash
     python build_db.py
     ```
     - Outputs to `vector_db/` (ignored by Git).

  1. **Query the Database**
     ```bash
     python query_db.py
     ```
     - Try: "What skills does a lead developer need?"

  ## Requirements
  - Python 3.10
  - GPU with CUDA (e.g., NVIDIA RTX 4080)
  - Ubuntu (WSL2 or native)
  - ~1-5 GB disk space for data and vectors

## License
Proprietary - see [LICENSE](LICENSE) for details. All rights reserved by Jarrod Connolly.

Thanks to Grok (xAI) for code collaboration and ideas.
