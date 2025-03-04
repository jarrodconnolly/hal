#!/bin/bash

# start_vllm.sh
# Launches vLLM server with predefined settings

echo "Starting vLLM server..."
# python -m vllm.entrypoints.openai.api_server \
#   --model TheBloke/Llama-2-13B-chat-GPTQ \
#   --revision gptq-4bit-128g-actorder_True \
#   --quantization gptq \
#   --dtype float16 \
#   --gpu-memory-utilization 0.85 \
#   --max-model-len 2048 \
#   --max-num-seqs 128 \
#   --trust-remote-code

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --trust-remote-code


echo "Server stopped."