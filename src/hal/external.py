"""External content fetching and query analysis for HAL's context enrichment."""

import spacy

from .logging_config import configure_logging

# Logging Configuration
logger = configure_logging()

# Initialize on load
nlp = spacy.load("en_core_web_sm")

# Extensible source config—add new sources here
SOURCES = {
    "arXiv": {"triggers": ["latest", "paper", "research"], "type": "research"},
    "MDN": {"triggers": ["doc", "syntax", "explain"], "type": "docs"},
    "GitHub": {"triggers": ["code", "example", "build"], "type": "code"},
}




def analyze_query(query: str) -> tuple[list[str], list[str]]:
    """Analyze a query with spaCy to extract sources and keywords for retrieval.

    Args:
        query: The user's input string to parse.

    Returns:
        A tuple of (selected_sources, keywords), where sources are external data origins
        (e.g., 'GitHub') and keywords are key terms for fetching relevant content.
    """
    doc = nlp(query.lower())

    # Skip VERB ROOT, grab key terms directly
    keywords = []
    for token in doc:
        if token.dep_ in [
            "amod",
            "nsubj",
            "dobj",
            "pobj",
            "compound",
            "npadvmod",
        ] and token.pos_ not in ["PRON", "AUX", "DET", "ADP", "PUNCT", "VERB"]:
            if not (token.dep_ == "compound" and token.head.text not in keywords):
                keywords.append(token.text)
    keywords = list(dict.fromkeys(keywords))  # Dedupe

    # Source selection (unchanged)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    adjs = [token.text for token in doc if token.pos_ == "ADJ"]
    selected_sources = []
    for source, config in SOURCES.items():
        triggers = config["triggers"]
        if any(token in triggers for token in nouns + verbs + adjs):
            selected_sources.append(source)
    if not selected_sources:
        selected_sources.append("GitHub")

    return selected_sources, keywords


def fetch_external(
    query: str, similarity_score: float, sources: list[str] = None
) -> tuple[str, list[float]]:
    """Fetch mock external content based on query analysis (placeholder for real APIs).

    Args:
        query: The user's input string to fetch content for.
        similarity_score: A baseline score (currently unused, for future weighting).
        sources: Optional list of sources to fetch from; if None, derived from query.

    Returns:
        A tuple of (content, scores), where content is a string of mock external data
        and scores are simulated similarity values.
    """
    sources, keywords = analyze_query(query)
    logger.info("analyze_query", query, sources, keywords, similarity_score)
    # Mock external fetch - replace with real logic if you have it
    mock_chunks = [
        f"External content from {', '.join(sources)} for {kw}" for kw in keywords[:5]
    ]
    mock_scores = [
        similarity_score - 0.1 * i for i in range(min(5, len(keywords)))
    ]  # Fake scores
    context = "\n".join(mock_chunks)
    return context, mock_scores
