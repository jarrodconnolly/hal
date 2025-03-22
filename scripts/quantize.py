"""Quantize a pretrained Llama model for HAL using GPTQ and evaluate perplexity.

Loads a pretrained Llama-3.1-8B-Instruct model from Hugging Face cache, quantizes it to 4-bit
with GPTQ using WikiText-2 samples (from quant-fetch.py style), and saves the result to a
local HF cache path. Tests the quantized model's perplexity on WikiText-2 to assess quality.
Used to optimize HAL's model efficiency for inference (e.g., vLLM integration).
"""
import os

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import Perplexity
from transformers import AutoTokenizer

# Pretrained model—use HF repo ID (cached already by vLLM)
pretrained_model_dir = "/home/totally/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Quantized model—HF cache path
quantized_model_dir = os.path.expanduser(
    "/home/totally/.cache/huggingface/hub/models--local--Llama-3.1-8B-Instruct-GPTQ-4bit/snapshots/4bit-128g-20250312"
)
os.makedirs(quantized_model_dir, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)


# Dataset prep (from repo sample)
def get_wikitext2(tokenizer, nsamples=256, seqlen=1024):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )
    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


traindataset = get_wikitext2(tokenizer)

# Quant config
quant_config = QuantizeConfig(bits=4, group_size=128, damp_percent=0.1, desc_act=False)

# Load and quantize on GPU
model = GPTQModel.from_pretrained(
    pretrained_model_dir, quantize_config=quant_config, torch_dtype=torch.float16
)
model.quantize(traindataset)

# Save to HF cache
model.save_quantized(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir)

ppl = Perplexity(model, tokenizer, "wikitext", "wikitext-2-raw-v1", "train", "text")
avg_ppl = sum(ppl.calculate(n_ctx=512, n_batch=512)) / 256
print(f"Quantized Model {quantized_model_dir} avg PPL: {avg_ppl}")
