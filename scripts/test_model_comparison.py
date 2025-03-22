"""Compare sentence transformer models for HAL's query similarity tuning.

Tests gte-large and multi-qa-MiniLM on related and noise query pairs, combining their cosine
similarities with weighted averages and veto thresholds. Iterates over weight ratios (0.10-0.50)
and veto cutoffs (0.15-0.25), printing averages and keep counts to evaluate model synergy.
Used to optimize HAL's retrieval or context-matching logic (e.g., in retrieval.py).
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Related query pairs
RELATED_PAIRS = [
    ("My name is Jarrod", "What is my name?"),
    ("I'm working on compilers", "What am I working on?"),
    ("I like Python", "What language do I like?"),
    ("I'm debugging a parser", "What am I debugging?"),
    ("Call me Jarrod", "What should you call me?"),
    ("I'm in timezone EST", "What timezone am I in?"),
    ("I build parsers", "What do I build?"),
    ("I'm into Rust", "What am I into?"),
    ("I fix bugs", "What do I fix?"),
    ("I'm from Seattle", "Where am I from?"),
    ("I enjoy coding", "What do I enjoy?"),
    ("I'm coding in Python", "What am I coding in?"),
    ("My favorite tool is Git", "What's my favorite tool?"),
    ("I hate bugs", "What do I hate?"),
    ("I use Vim", "What do I use?"),
    ("I'm learning Go", "What am I learning?"),
    ("I don't like Java", "What don't I like?"),
    ("I wrote a script", "What did I write?"),
    ("I'm Jarrod", "Who am I?"),
    ("I code daily", "How often do I code?"),
    ("I prefer C++", "What do I prefer?"),
    ("I'm coding now", "What am I doing now?"),
    ("I switched to Linux", "What did I switch to?"),
]

# Noise query pairs
NOISE_PAIRS = [
    ("I like Python", "What's an AST?"),
    ("I'm working on compilers", "What's my name?"),
    ("I fix bugs", "Where am I from?"),
    ("I use Vim", "What's my name?"),
    ("I'm learning Go", "Where am I from?"),
    ("I eat pizza", "What's my name?"),
    ("I'm tired", "What's my timezone?"),
    ("I love coffee", "What's my name?"),
    ("I run fast", "What do I code in?"),
]

QUERY_PAIRS = RELATED_PAIRS + NOISE_PAIRS

# Models
MODELS = {
    "gte-large": SentenceTransformer("thenlper/gte-large"),
    "multi-qa-MiniLM": SentenceTransformer("multi-qa-MiniLM-L6-cos-v1"),
}

# Weight range: 0.10 to 0.50, step 0.05
WEIGHT_RANGE = [(w1/20, 1 - w1/20) for w1 in range(2, 11)]

# Veto thresholds: 0.15 to 0.25, step 0.01
VETO_THRESHOLDS = [round(0.15 + i * 0.01, 2) for i in range(11)]

def test_combo(gte_model, toss_model, query_pairs, w1, w2, veto_threshold):
    related_results = []
    noise_results = []
    for q1, q2 in RELATED_PAIRS:
        gte_emb1 = gte_model.encode(q1)
        gte_emb2 = gte_model.encode(q2)
        toss_emb1 = toss_model.encode(q1)
        toss_emb2 = toss_model.encode(q2)
        gte_sim = cosine_similarity([gte_emb1], [gte_emb2])[0][0]
        toss_sim = cosine_similarity([toss_emb1], [toss_emb2])[0][0]
        weighted = (w1 * gte_sim) + (w2 * toss_sim) if toss_sim >= veto_threshold else 0.2
        veto = 1 if toss_sim >= veto_threshold else 0
        related_results.append((weighted, veto))
    for q1, q2 in NOISE_PAIRS:
        gte_emb1 = gte_model.encode(q1)
        gte_emb2 = gte_model.encode(q2)
        toss_emb1 = toss_model.encode(q1)
        toss_emb2 = toss_model.encode(q2)
        gte_sim = cosine_similarity([gte_emb1], [gte_emb2])[0][0]
        toss_sim = cosine_similarity([toss_emb1], [toss_emb2])[0][0]
        weighted = (w1 * gte_sim) + (w2 * toss_sim) if toss_sim >= veto_threshold else 0.2
        veto = 1 if toss_sim >= veto_threshold else 0
        noise_results.append((weighted, veto))
    return related_results, noise_results

def main():
    print("Related Pairs Reference\n" + "="*20)
    for i, (q1, q2) in enumerate(RELATED_PAIRS, 1):
        print(f"{i}. Q1: {q1} | Q2: {q2}")
    print("\nNoise Pairs Reference\n" + "="*20)
    for i, (q1, q2) in enumerate(NOISE_PAIRS, 1):
        print(f"{i}. Q1: {q1} | Q2: {q2}")
    print("="*20)

    print("\nTesting Model Combination (gte-large + multi-qa-MiniLM)\n" + "="*20)
    
    for veto_threshold in VETO_THRESHOLDS:
        print(f"\nVeto Threshold: {veto_threshold}")
        print("Weights | Related Avg | Noise Avg | Related Keep | Noise Keep")
        print("-"*50)
        for w1, w2 in WEIGHT_RANGE:
            related_results, noise_results = test_combo(
                MODELS["gte-large"], MODELS["multi-qa-MiniLM"], QUERY_PAIRS, w1, w2, veto_threshold
            )
            related_weighted_avg = sum(r[0] for r in related_results) / len(related_results)
            noise_weighted_avg = sum(n[0] for n in noise_results) / len(noise_results)
            related_veto_keep = sum(r[1] for r in related_results)
            noise_veto_keep = sum(n[1] for n in noise_results)
            print(f"{w1:.2f}+{w2:.2f} | "
                  f"{related_weighted_avg:.4f} | {noise_weighted_avg:.4f} | "
                  f"{related_veto_keep}/{len(RELATED_PAIRS)} | {noise_veto_keep}/{len(NOISE_PAIRS)}")

if __name__ == "__main__":
    main()