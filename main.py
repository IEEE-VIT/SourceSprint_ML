
import argparse
import os
import string
import sys

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency guard
    Image = None

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover - optional dependency guard
    Word2Vec = None

try:
    from clip_interrogator import Config, Interrogator
except ImportError:  # pragma: no cover - optional dependency guard
    Config = None
    Interrogator = None


def validate_image_path(image_path):
    """Validate and normalize the user-provided image path."""
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

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".webp",
        ".tif",
        ".tiff",
    }
    _, extension = os.path.splitext(image_path)
    if extension.lower() not in allowed_extensions:
        raise ValueError(
            "Unsupported image format. Please provide a common image file."
        )

    return image_path


def validate_search_text(user_input):
    """Validate user-provided search text."""
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

ci = None
if Config is not None and Interrogator is not None:
    try:
        ci = Interrogator(
            Config(
                clip_model_name=clip_model_name,
                caption_model_name=caption_model_name,
            )
        )
    except Exception:
        ci = None


# -----------------------
# Image to Prompt
# -----------------------
def image_to_prompt(image):
    """Takes a PIL image and converts it to a descriptive text prompt."""
    if image is None:
        raise ValueError("Image object is required.")

    if Image is None:
        raise ImportError("Pillow is required to process images.")

    if ci is None:
        raise RuntimeError(
            "CLIP Interrogator is unavailable. Please make sure the dependencies are installed."
        )

    try:
        rgb_image = image.convert("RGB")
        text = ci.interrogate_fast(rgb_image)
        if text is None:
            return ""
        return str(text).strip()
    except Exception as exc:
        raise ValueError(f"Could not extract text from the image: {exc}") from exc


def display_image(image):
    """Display the image when possible, without crashing on headless systems."""
    if Image is None:
        return

    try:
        image.show()
    except Exception:
        print("Image preview skipped because a display is unavailable.")


def find_similar_words(user_input, topn=10):
    """Return similar words using the local Word2Vec model if available."""
    if Word2Vec is None:
        raise ImportError("gensim is required for Word2Vec similarity search.")

    model_path = "word2vec_model.model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Word2Vec model not found: {model_path}")

    model = Word2Vec.load(model_path)
    user_input = validate_search_text(user_input)

    try:
        return model.wv.most_similar(user_input, topn=topn)
    except KeyError:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Image search using CLIP and Word2Vec"
    )
    parser.add_argument("image_path", nargs="?", help="Path to the image file")
    parser.add_argument(
        "--text",
        dest="search_text",
        help="Text to search for in the Word2Vec model",
    )
    args = parser.parse_args()

    if not args.image_path:
        parser.error("An image path is required.")

    try:
        image_path = validate_image_path(args.image_path)
        image = Image.open(image_path)
        display_image(image)

        user_input = args.search_text
        if user_input is None:
            user_input = input("Enter the text you want to search for: ")

        user_input = validate_search_text(user_input)
        print(f"You entered: {user_input}")

        extracted_text = image_to_prompt(image)
        print("\nText extracted from image using CLIP:")
        print(extracted_text)

        try:
            similar_words = find_similar_words(user_input)
            if similar_words:
                print("\nMost similar words to your input:")
                for word, similarity in similar_words:
                    print(f"{word} - similarity: {similarity:.2f}")
            else:
                print(f"The word '{user_input}' is not in the Word2Vec vocabulary.")
        except (FileNotFoundError, ImportError):
            print("Similarity lookup skipped because the Word2Vec model or dependency is unavailable.")

    except (FileNotFoundError, ValueError, TypeError, ImportError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

