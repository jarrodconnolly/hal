"""vLLM server for HAL project - runs Llama-3.2-3B-Instruct model with OpenAI-compatible API."""

import asyncio
import signal

import torch
import torch.distributed as dist
from vllm.entrypoints.openai.api_server import run_server as vllm_run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

from .logging_config import configure_logging

# Logging Configuration
logger = configure_logging()


def start_vllm_server():
    """Start the vLLM server with Llama-3.2-3B-Instruct for HAL's API.

    Configures the server with bfloat16 precision for efficiency, a 16k token limit for long contexts,
    and 80% GPU memory utilization on an RTX 4080. Handles SIGINT/SIGTERM for graceful shutdown.
    """
    args_list = [
        "--model", "microsoft/Phi-4-mini-instruct",  # Swap model
        "--dtype", "float16",                       # Drop to FP16—Phi-4—faster, less VRAM
        "--gpu-memory-utilization", "0.9",         # Push to 90%—Phi-4—needs room
        "--max-model-len", "4096",                 # Cut context—4k—HAL’s queries—fine
        "--max-num-seqs", "8",                     # Halve seqs—8—VRAM relief
        "--trust-remote-code",                     # Keep—Phi-4—Hugging Face—safe
        "--chat-template", "src/hal/prompt_template.jinja",  # Same—hope it vibes
        "--task", "generate",                      # No change—chat gen
        #"--quantization", "bitsandbytes",          # Add—4-bit—~8GB VRAM—safety net
        #"--load-format", "bitsandbytes"            # Match quant—BnB—Phi-4—fits
    ]
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(args_list)
    validate_parsed_serve_args(args)

    def shutdown_handler(signum, frame):
        logger.info("Caught signal, shutting down...")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit

    # Set our handler—main process
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info("Starting vLLM server...")
    try:
        asyncio.run(vllm_run_server(args))
    except (SystemExit, KeyboardInterrupt):
        logger.info("Shutting down vLLM server...")
    except Exception as e:
        logger.error(f"vLLM server failed: {e}")
    finally:
        logger.info("Shutdown complete.")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    start_vllm_server()
