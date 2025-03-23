"""Test hybrid spaCy fact extraction from queries for HAL's context system.

Uses spaCy's en_core_web_lg model to extract user facts from queries via a three-step approach:
NER for names/timezones, intent mapping from root verbs (e.g., ‘like' to preferences), and noun
chunk fallbacks. Prints tokens, POS tags, dependencies, and facts for test queries. Used to debug
and refine HAL's user context parsing (e.g., for external.py).
"""
import spacy

# Load spaCy large model for better NER
nlp = spacy.load("en_core_web_lg")

# Test queries (expanded)
QUERIES = [
    "My name is Jarrod",
    "I'm working on compilers",
    "I like Python",
    "Call me Jarrod",
    "I'm in timezone EST",
    "I'm debugging a parser",
    "I build parsers",
    "I'm into Rust",
    "I fix bugs",
    "I'm from Seattle",
]

def extract_facts(query):
    doc = nlp(query)
    facts = []

    # Step 1: NER - Prioritize PERSON, force PROPN after "in" to timezone
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            facts.append({"type": "name", "value": ent.text})
    for token in doc:
        if token.dep_ == "pobj" and token.head.text == "in" and token.pos_ == "PROPN":
            facts.append({"type": "timezone", "value": token.text})

    # Step 2: Intent via root verb + dependencies (use lemma)
    root = next(token for token in doc if token.dep_ == "ROOT")
    verb = root.lemma_.lower()

    # If root is "be", shift to first prep (e.g., "into") or VERB (not AUX)
    if verb == "be":
        for child in root.children:
            if child.pos_ in ["ADP"] and child.dep_ == "prep":  # "into", "in"
                verb = child.lemma_.lower()
                root = child
                break
        else:  # No prep found, try VERB
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ != "aux":
                    verb = token.lemma_.lower()
                    root = token
                    break

    # Map verb intent to fact type
    intent_map = {
        "be": "name",       # "My name is Jarrod"
        "like": "preference",  # "I like Python"
        "work": "project",  # "I'm working on compilers"
        "call": "name",     # "Call me Jarrod"
        "debug": "debugging",  # "I'm debugging a parser"
        "build": "project",  # "I build parsers"
        "into": "preference",  # "I'm into Rust"
        "fix": "debugging",  # "I fix bugs"
        "from": "location",  # "I'm from Seattle"
    }

    if verb in intent_map:
        # Collect candidates from root and its prep children
        candidates = []
        for child in root.children:
            if child.pos_ != "PRON" and child.dep_ != "nsubj":
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    candidates.append({"type": intent_map[verb], "value": child.text})
                # Check prep children (e.g., "on" -> "compilers")
                elif child.dep_ == "prep":
                    for grandkid in child.children:
                        if grandkid.pos_ != "PRON" and grandkid.dep_ in ["pobj"]:
                            candidates.append({"type": intent_map[verb], "value": grandkid.text})
        # Debug: Log candidates
        print(f"Candidates for {query}: {candidates}")
        # Take last candidate if any
        if candidates:
            facts.append(candidates[-1])

    # Step 3: Fallback - Noun chunk after VERB/ADP
    if not facts:
        for chunk in doc.noun_chunks:
            head = chunk.root.head
            if head.pos_ in ["VERB", "ADP"] and head.dep_ != "aux" and chunk.root.pos_ != "PRON":
                verb = head.lemma_.lower()
                type_guess = intent_map.get(verb, "fact")
                facts.append({"type": type_guess, "value": chunk.text})

    # Dedupe facts
    seen = set()
    unique_facts = [
        f for f in facts if not (f["value"] in seen or seen.add(f["value"]))
    ]

    return doc, unique_facts

def main():
    print("Testing SpaCy Flagging (Hybrid Approach)\n" + "=" * 30)
    for query in QUERIES:
        doc, facts = extract_facts(query)
        print(f"Query: {query}")
        print("Tokens:", [token.text for token in doc])
        print("POS Tags:", [(token.text, token.pos_) for token in doc])
        print(
            "Dependencies:",
            [(token.text, token.dep_, token.head.text) for token in doc],
        )
        print("Root:", [token.text for token in doc if token.dep_ == "ROOT"][0])
        print(f"Facts: {facts}")
        print("-" * 30)

if __name__ == "__main__":
    main()