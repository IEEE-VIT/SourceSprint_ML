import argparse
import string
import nltk

from PIL import Image
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from clip_interrogator import Config, Interrogator


# -----------------------
# Download NLTK resources
# -----------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -----------------------
# CLIP configuration
# -----------------------
clip_model_name = "ViT-L-14/openai"
caption_model_name = "blip-large"

ci_config = Config(
    clip_model_name=clip_model_name,
    caption_model_name=caption_model_name
)

# Load the CLIP Interrogator model
ci = Interrogator(ci_config)


# -----------------------
# Text preprocessing
# -----------------------
def preprocess_text(text):
    """
    Takes raw text as input and returns a list
    of preprocessed tokens.

    Processing includes:
    1. Lowercase conversion
    2. Punctuation removal
    3. Tokenization using NLTK
    4. Stopword removal
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation using string module
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize text using NLTK
    tokens = word_tokenize(text)

    # Get English stopwords
    stop_words = set(stopwords.words("english"))

    # Remove stopwords
    processed_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return processed_tokens


# -----------------------
# Image to Prompt
# -----------------------
def image_to_prompt(image):
    """
    Takes a PIL image and generates a text
    description using CLIP Interrogator.
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Generate text description
    text = ci.interrogate_fast(image)

    return str(text)


# -----------------------
# Command-line arguments
# -----------------------
parser = argparse.ArgumentParser(
    description="Image search using CLIP and Word2Vec"
)

parser.add_argument(
    "image_path",
    help="Path to the image file"
)

args = parser.parse_args()

image_path = args.image_path


# -----------------------
# Open and display image
# -----------------------
image = Image.open(image_path)

image.show()


# -----------------------
# Prompt user for search text
# -----------------------
user_input = input(
    "Enter the text you want to search for: "
)

print(f"\nYou entered: {user_input}")


# -----------------------
# Convert image to text
# -----------------------
extracted_text = image_to_prompt(image)

print("\nText extracted from image using CLIP:")
print(extracted_text)


# -----------------------
# Preprocess extracted text
# -----------------------
processed_tokens = preprocess_text(extracted_text)

print("\nPreprocessed image text:")
print(processed_tokens)


# -----------------------
# Preprocess user input
# -----------------------
processed_user_input = preprocess_text(user_input)

print("\nPreprocessed user input:")
print(processed_user_input)


# -----------------------
# Load trained Word2Vec model
# -----------------------
model_path = "word2vec_model.model"

model = Word2Vec.load(model_path)


# -----------------------
# Find similar words
# -----------------------
if len(processed_user_input) == 0:

    print("\nNo valid words found in the search input.")

else:

    # Use the first preprocessed word
    search_word = processed_user_input[0]

    try:

        similar_words = model.wv.most_similar(
            search_word,
            topn=10
        )

        print(
            f"\nMost similar words to '{search_word}':"
        )

        for word, similarity in similar_words:

            print(
                f"{word} - similarity: {similarity:.2f}"
            )

    except KeyError:

        print(
            f"\nThe word '{search_word}' is not "
            "in the Word2Vec vocabulary."
        )


# -----------------------
# Optional fuzzy matching
# -----------------------
# for word, similarity in similar_words:
#
#     confidence = fuzz.partial_ratio(
#         user_input.lower(),
#         word.lower()
#     )
#
#     if confidence >= 80:
#
#         print(
#             f"Fuzzy match: {word} "
#             f"(Confidence: {confidence}%)"
#         )