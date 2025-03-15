"""Test spaCy-based rule extraction of user facts for HAL's context system.

Uses spaCy's en_core_web_lg model to preprocess queries (expanding contractions) and extract
user-specific facts via dependency parsing rules (e.g., verbs, objects, prepositions). Prints
tokens, POS tags, dependencies, and extracted facts for a set of test queries. Used to debug
and refine fact extraction for HAL's user context pipeline (e.g., external.py).
"""
import spacy
import re

nlp = spacy.load("en_core_web_lg")

def preprocess_text(text):
    contractions = {
        r"I'm": "I am",
        r"It's": "It is",
        r"Let's": "Let us",
        r"'s": " is"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return text

def extract_user_facts(text):
    text = preprocess_text(text)
    doc = nlp(text)
    facts = []

    for sent in doc.sents:
        root = next((token for token in sent if token.dep_ == "ROOT"), None)
        if not root:
            continue

        has_user = any(token.text.lower() in ["i", "me", "my"] and token.dep_ in ["nsubj", "dobj", "pobj", "poss"] for token in sent)
        if not has_user:
            continue

        # Verb handling
        verb_token = root
        verb_parts = []
        if root.lemma_ == "be" or root.text == "am":
            verb_parts.append("is")
            has_favorite = "favorite" in [t.text.lower() for t in sent]
            for child in root.children:
                if child.dep_ == "attr" and child.pos_ != "PRON":
                    verb_parts.append(child.text)
                    prefix = "likes" if has_favorite else "is"
                    facts.append(f"User {prefix} {' '.join(verb_parts[1:])}")
                    verb_parts = ["is"]
                elif child.dep_ == "acomp" and child.pos_ == "ADJ":
                    verb_parts.append(child.text)
                    facts.append(f"User {' '.join(verb_parts)}")
                    verb_parts = ["is"]
                elif child.pos_ == "VERB" and child.dep_ == "xcomp":
                    verb_parts.append(child.text)
                    for grandkid in child.children:
                        if grandkid.dep_ in ["dobj", "pobj"]:
                            verb_parts.append(grandkid.text)
                    facts.append(f"User {' '.join(verb_parts)}")
                    verb_parts = ["is"]
                elif child.dep_ == "prep" and child.text in ["from", "into", "in"]:
                    verb_parts.append(child.text)
                    for grandkid in child.children:
                        if grandkid.dep_ == "pobj":
                            verb_parts.append(grandkid.text)
                    facts.append(f"User {' '.join(verb_parts)}")
                    verb_parts = ["is"]
        else:
            has_aux_be = verb_token.tag_ == "VBG" and any(t.pos_ == "AUX" for t in sent)
            if has_aux_be:
                verb_parts.append("is")
            verb = verb_token.text
            if verb_token.pos_ == "VERB" and verb_token.tag_ in ["VB", "VBP"]:
                lemma = verb_token.lemma_
                if lemma.endswith(("x", "ch", "sh", "s", "z")):
                    verb = lemma + "es"
                else:
                    verb = lemma + "s"
            verb_parts.append(verb)
            for child in verb_token.children:
                if child.dep_ in ["dobj", "pobj"] and child.pos_ != "PRON":
                    verb_parts.append(child.text)
                elif child.dep_ == "prep" and child.text in ["on", "as"]:
                    for grandkid in child.children:
                        if grandkid.dep_ == "pobj":
                            verb_parts.append(f"{child.text} {grandkid.text}")
            if len(verb_parts) > 1 or (len(verb_parts) == 1 and has_aux_be):
                facts.append(f"User {' '.join(verb_parts)}")

        # "me" special case
        if "me" in [t.text.lower() for t in sent]:
            for token in sent:
                if token.text.lower() == "me" and token.dep_ in ["dobj", "pobj"]:
                    for child in sent:
                        if child.dep_ in ["pobj", "oprd"] and child.pos_ == "PROPN":
                            facts.append(f"User is {child.text}")
                            break

    # Dedupe with priority
    seen = set()
    unique_facts = []
    has_is_name = any("is" in f.split() and any(t.pos_ == "PROPN" for t in nlp(f)) for f in facts)
    for fact in facts:
        if has_is_name and "refers as" in fact and any(t.pos_ == "PROPN" for t in nlp(fact)):
            continue
        if fact not in seen:
            seen.add(fact)
            unique_facts.append(fact)
    if not unique_facts:
        unique_facts = ["none"]

    return doc, unique_facts

def main():
    print("SpaCy Rule-Based Fact Extraction Test\n" + "="*40)
    print("Queries:")
    for i, q in enumerate(QUERIES, 1):
        print(f"{i}. {q}")
    print("="*40)

    for query in QUERIES:
        doc, facts = extract_user_facts(query)
        print(f"Query: {query}")
        print("Tokens:", [token.text for token in doc])
        print("POS Tags:", [(token.text, token.pos_) for token in doc])
        print("Dependencies:", [(token.text, token.dep_, token.head.text) for token in doc])
        print("Root:", [token.text for token in doc if token.dep_ == "ROOT"][0])
        print(f"Facts: {facts}")
        print("-"*40)

QUERIES = [
    "Hello HAL, I have been working hard on this project. Let's work on it together. You can refer to me as Jarrod.",
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
    "I'm debugging a parser right now",
    "Call me Jarrod",
    "I'm into Rust and compilers",
    "I fix bugs every day",
    "I'm in timezone EST",
]

if __name__ == "__main__":
    main()