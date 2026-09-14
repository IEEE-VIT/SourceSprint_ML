"""CLIP inference and Word2Vec model operations.

Model dependencies and weights are loaded only when explicitly requested.
"""

CLIP_MODEL_NAME = "ViT-L-14/openai"
CAPTION_MODEL_NAME = "blip-large"


def load_clip_model():
    """Create the CLIP Interrogator with the original model configuration."""
    from clip_interrogator import Config, Interrogator

    return Interrogator(Config(
        clip_model_name=CLIP_MODEL_NAME,
        caption_model_name=CAPTION_MODEL_NAME,
    ))


def image_to_prompt(image, interrogator):
    """Extract a text description from a PIL image using CLIP."""
    return str(interrogator.interrogate_fast(image.convert("RGB")))


def load_word2vec_model(model_path):
    """Load an existing trained Word2Vec model."""
    from gensim.models import Word2Vec

    return Word2Vec.load(model_path)


def find_similar_words(model, text, topn=10):
    """Return similar words; propagate KeyError for unknown vocabulary entries."""
    return model.wv.most_similar(text, topn=topn)


def print_similar_words(similar_words):
    """Print similar words and their similarity scores."""
    print("\nMost similar words to your input:")
    for word, similarity in similar_words:
        print(f"{word} - similarity: {similarity:.2f}")
