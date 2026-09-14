import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_text(raw_text):
    """Tokenize text and remove punctuation and English stopwords."""
    text_without_punctuation = raw_text.translate(
        str.maketrans("", "", string.punctuation)
    )
    tokens = word_tokenize(text_without_punctuation.lower())
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]
