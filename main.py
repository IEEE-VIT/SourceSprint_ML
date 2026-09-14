
import os
import string
import sys
import argparse

try:
    from sklearn import svm
except ImportError:  # pragma: no cover - optional dependency guard
    svm = None

try:
    from joblib import dump, load
except ImportError:  # pragma: no cover - optional dependency guard
    dump = None
    load = None

from PIL import Image

try:
    import nltk
except ImportError:  # pragma: no cover - optional dependency guard
    nltk = None

try:
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover - optional dependency guard
    stopwords = None

try:
    from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover - optional dependency guard
    word_tokenize = None

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional dependency guard
    Word2Vec = None

try:
    from fuzzywuzzy import fuzz
except ImportError:  # pragma: no cover - optional dependency guard
    fuzz = None

# -----------------------
# CLIP libraries
# -----------------------
try:
    from clip_interrogator import Config, Interrogator
except ImportError:  # pragma: no cover - optional dependency guard
    Config = None
    Interrogator = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None


def validate_image_path(image_path):
    """Validate that the image path is present, exists, and is an image file."""
    if image_path is None:
        raise ValueError("Image path is required.")
    if not isinstance(image_path, str):
        raise TypeError("Image path must be a string.")

    image_path = image_path.strip()
    if not image_path:
        raise ValueError("Image path cannot be empty.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.isfile(image_path):
        raise ValueError(f"Path is not a file: {image_path}")

    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
    _, extension = os.path.splitext(image_path)
    if extension.lower() not in allowed_extensions:
        raise ValueError("Unsupported image format. Please provide a common image file.")

    return image_path


def validate_search_text(user_input):
    """Validate user-entered text before using it in the NLP pipeline."""
    if user_input is None:
        raise ValueError("Search text is required.")
    if not isinstance(user_input, str):
        raise TypeError("Search text must be a string.")

    cleaned = user_input.strip()
    if not cleaned:
        raise ValueError("Search text cannot be empty.")
    if cleaned.isdigit():
        raise ValueError("Search text must contain letters or words, not only numbers.")
    if all(char in string.punctuation or char.isspace() for char in cleaned):
        raise ValueError("Search text must include at least one letter or number.")

    return cleaned


# -----------------------
# CLIP configuration
# -----------------------
clip_model_name = "ViT-L-14/openai"
caption_model_name = "blip-large"

ci_config = None
ci = None
if Config is not None and Interrogator is not None:
    ci_config = Config(
        clip_model_name=clip_model_name,
        caption_model_name=caption_model_name,
    )
    try:
        ci = Interrogator(ci_config)
    except Exception:
        ci = None


# -----------------------
# Image to Prompt
# -----------------------
def image_to_prompt(image):
    """
    Takes a PIL image as input and uses the CLIP model
    to extract a text description from the image.

    Returns:
        str: Extracted text description.
    """
    if image is None:
        raise ValueError("Image object is required.")
    if ci is None:
        raise RuntimeError(
            "CLIP Interrogator is unavailable. Please install the required dependencies."
        )

    try:
        image = image.convert("RGB")
        text = ci.interrogate_fast(image)
        return str(text)
    except Exception as exc:
        raise ValueError(f"Could not extract text from the image: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Image search using CLIP and Word2Vec"
    )
    parser.add_argument(
        "image_path",
        help="Path to the image file",
    )
    args = parser.parse_args()

    try:
        image_path = validate_image_path(args.image_path)
        image = Image.open(image_path)
        try:
            image.show()
        except Exception:
            print("Image preview skipped because no display is available.")

        user_input = input("Enter the text you want to search for: ")
        user_input = validate_search_text(user_input)
        print(f"You entered: {user_input}")

        try:
            extracted_text = image_to_prompt(image)
            print("\nText extracted from image using CLIP:")
            print(extracted_text)
        except RuntimeError as exc:
            print(f"CLIP image extraction skipped: {exc}", file=sys.stderr)

        if Word2Vec is not None:
            model_path = "word2vec_model.model"
            try:
                model = Word2Vec.load(model_path)
                try:
                    similar_words = model.wv.most_similar(user_input, topn=10)
                    print("\nMost similar words to your input:")
                    for word, similarity in similar_words:
                        print(f"{word} - similarity: {similarity:.2f}")
                except KeyError:
                    print(f"The word '{user_input}' is not in the Word2Vec vocabulary.")
            except FileNotFoundError as exc:
                print(f"Model lookup skipped: {exc}", file=sys.stderr)
        else:
            print("Word2Vec similarity lookup skipped because gensim is not installed.", file=sys.stderr)

    except (FileNotFoundError, ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

