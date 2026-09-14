"""Text preparation for Word2Vec vocabulary lookups."""


def preprocess_text(text):
    """Trim surrounding whitespace, preserving case and phrase vocabulary keys."""
    return text.strip()
