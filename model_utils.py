from gensim.models import Word2Vec


def load_model(model_path="word2vec_model.model"):
    """
    Loads a trained Word2Vec model from disk.

    Returns:
        Word2Vec: The loaded model.
    """
    return Word2Vec.load(model_path)


def find_similar_words(model, user_input, topn=10):
    """
    Finds words most similar to user_input using the given
    Word2Vec model.

    Returns:
        list[tuple[str, float]] | None: Similar words and scores,
        or None if user_input is not in the vocabulary.
    """
    try:
        return model.wv.most_similar(user_input, topn=topn)
    except KeyError:
        return None
