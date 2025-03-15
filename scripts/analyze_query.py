"""Benchmark NLTK and spaCy for analyzing HAL user queries.

Measures the performance of tokenization, POS tagging, and (for spaCy) dependency parsing
on a sample query to inform HAL's query processing pipeline (e.g., external.py's analyze_query).
Runs warm-up iterations to stabilize timings, then averages test runs, printing tokens, tags,
and timing results for comparison.
"""
import time

import nltk
import spacy
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download NLTK data (run once)
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

# Load spaCy model (run `python -m spacy download en_core_web_sm` once)
nlp = spacy.load("en_core_web_sm")

# Constants—tweak these
WARMUP_RUNS = 10  # Runs to warm up (discarded)
TEST_RUNS = 1000  # Runs to average

# Query to analyze—swap this out
# query = "what is a good recipe to bake bread?"
query = "What's alien bread flux?"

# NLTK Analysis
print("=== NLTK Analysis ===")
nltk_times = []
for i in range(WARMUP_RUNS + TEST_RUNS):
    start_time = time.time()
    nltk_tokens = word_tokenize(query)
    nltk_pos = pos_tag(nltk_tokens)
    nltk_times.append(time.time() - start_time)
# Average test runs after warm-up
nltk_avg = sum(nltk_times[WARMUP_RUNS:]) / TEST_RUNS
print("Tokens:", nltk_tokens)
print("POS Tags:", nltk_pos)
print(
    f"Average Time ({TEST_RUNS} runs after {WARMUP_RUNS} warm-up): {nltk_avg:.4f} seconds"
)
print()

# spaCy Analysis
print("=== spaCy Analysis ===")
spacy_times = []
for i in range(WARMUP_RUNS + TEST_RUNS):
    start_time = time.time()
    doc = nlp(query)
    spacy_times.append(time.time() - start_time)
# Average test runs after warm-up
spacy_avg = sum(spacy_times[WARMUP_RUNS:]) / TEST_RUNS
print("Tokens:", [token.text for token in doc])
print("POS Tags:", [(token.text, token.pos_) for token in doc])
print("Dependencies:", [(token.text, token.dep_, token.head.text) for token in doc])
print("Root:", [token.text for token in doc if token.dep_ == "ROOT"][0])
print(
    f"Average Time ({TEST_RUNS} runs after {WARMUP_RUNS} warm-up): {spacy_avg:.4f} seconds"
)
