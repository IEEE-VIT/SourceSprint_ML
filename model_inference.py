from gensim.models import Word2Vec


def load_word2vec_model(model_path):
    """Load a trained Word2Vec model from disk."""
    return Word2Vec.load(model_path)


def find_similar_words(model, query, topn=10):
    """Return the words most similar to a vocabulary query."""
    return model.wv.most_similar(query, topn=topn)


def print_similar_words(similar_words):
    """Print similar words and their similarity scores."""
    print("\nMost similar words to your input:")
    for word, similarity in similar_words:
        print(f"{word} - similarity: {similarity:.2f}")
