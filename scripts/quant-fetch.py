"""Fetch a sample of long texts from WikiText for HAL's quantization experiments.

Loads the wikitext-2-raw-v1 dataset (train split) using Hugging Face's datasets library,
filters for texts over 512 characters, and grabs the first 256 samples. Used to gather
raw text data for testing model quantization or preprocessing steps in HAL's pipeline.
"""
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [ex["text"] for ex in dataset if len(ex["text"]) > 512][:256]  # 256 samples, 512+ chars