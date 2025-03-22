"""User fact extraction for HAL's personalization using spaCy NLP."""

import re

import spacy

nlp = spacy.load("en_core_web_lg")


def preprocess_text(text: str) -> str:
    """Expand contractions in the text."""
    contractions = {
        r"I'm": "I am",
        r"it's": "it is",
        r"let's": "let us",
        r"'s": " is",
        r"we're": "we are",
        r"can't": "cannot",
        r"won't": "will not",
        r"aren't": "are not",
        r"didn't": "did not",
        r"you're": "you are",
        r"they're": "they are"
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return text


def extract_user_facts(text: str) -> tuple[spacy.tokens.Doc, list[str]]:
    """Extract user-specific facts from text using spaCy dependency parsing.

    Args:
        text: The user's input string to analyze for facts (e.g., "I like coding").

    Returns:
        A tuple of (doc, facts), where doc is the spaCy-parsed text and facts is a list
        of extracted user facts (e.g., ["User likes coding"]). Returns ["none"] if no facts.

    Notes:
        Focuses on sentences with user references ("I", "me", "my") and verb structures.
        Handles "be" verbs, favorites, and prepositional phrases for fact variety.
    """
    text = preprocess_text(text)
    doc = nlp(text)
    facts = []

    for sent in doc.sents:
        root = next((token for token in sent if token.dep_ == "ROOT"), None)
        if not root:
            continue

        has_user = any(
            token.text.lower() in ["i", "me", "my"]
            and token.dep_ in ["nsubj", "dobj", "pobj", "poss"]
            for token in sent
        )
        if not has_user:
            continue

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

        if "me" in [t.text.lower() for t in sent]:
            for token in sent:
                if token.text.lower() == "me" and token.dep_ in ["dobj", "pobj"]:
                    for child in sent:
                        if child.dep_ in ["pobj", "oprd"] and child.pos_ == "PROPN":
                            facts.append(f"User is {child.text}")
                            break

    seen = set()
    unique_facts = []
    has_is_name = any(
        "is" in f.split() and any(t.pos_ == "PROPN" for t in nlp(f)) for f in facts
    )
    for fact in facts:
        if (
            has_is_name
            and "refers as" in fact
            and any(t.pos_ == "PROPN" for t in nlp(fact))
        ):
            continue
        if fact not in seen:
            seen.add(fact)
            unique_facts.append(fact)
    if not unique_facts:
        unique_facts = ["none"]

    return doc, unique_facts
