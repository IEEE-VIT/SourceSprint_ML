import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz


def preprocess_text(text):
    """
    Tokenizes text, lowercases it, strips punctuation,
    and removes stopwords.

    Returns:
        list[str]: Cleaned tokens.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [
        t for t in tokens
        if t not in string.punctuation and t not in stop_words
    ]
    return tokens


def fuzzy_match(user_input, candidates, threshold=80):
    """
    Compares user_input against a list of candidate words
    using fuzzy string matching.

    Returns:
        list[tuple[str, int]]: Matches at/above the confidence threshold.
    """
    matches = []
    for word in candidates:
        confidence = fuzz.partial_ratio(user_input.lower(), word.lower())
        if confidence >= threshold:
            matches.append((word, confidence))
    return matches
