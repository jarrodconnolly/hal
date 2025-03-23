"""Test fact extraction from user queries using Flan-T5 for HAL.

Loads google/flan-t5-base, processes a list of sample queries to extract 1-2 concise facts
per query as English sentences, and times the process. Prints results and timings to evaluate
T5's performance for parsing user context (e.g., names, preferences) in HAL's pipeline.
"""
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to("cuda")

QUERIES = [
    "Hello HAL, I have been working hard on this project. Lets work on it together and find some great changes and fixes. You can refer to me as Jarrod.",
    "My name is Jarrod",
    "I'm from Seattle",
    "I like Python",
    "I'm learning Go",
    "My favorite tool is Git",
    "I ate pizza",
    "It's raining outside",
    "I'm tired today",
    "The sky is blue",
    "I saw a movie",
]

def extract_fact(text):
    input_text = (
        "Extract 1-2 short facts about the user from the text. "
        "Use only explicit details, no assumptions. "
        "Write each fact as a concise English sentence under 10 words. "
        "Examples: 'My name is Bob' → 'Users name is Bob', 'I like coding' → 'User likes coding'. "
        "Separate facts with a period. "
        "Text: " + text
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end = time.time()
    return f"fact: {result}", end - start

print("Fact Extraction Test Harness\n" + "="*20)
print("Queries:")
for i, q in enumerate(QUERIES, 1):
    print(f"{i}. {q}")
print("="*20)

print("\ngoogle/flan-t5-base (Temperature 0.7)")
print("-"*40)
for query in QUERIES:
    result, timing = extract_fact(query)
    print(f"{query}: {result}, time: {timing:.4f}s")