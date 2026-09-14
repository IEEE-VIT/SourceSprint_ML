
import os, subprocess
from sklearn import svm
from joblib import dump, load
from PIL import Image
import nltk
import string
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz

# -----------------------
# CLIP libraries
# -----------------------
from clip_interrogator import Config, Interrogator
import torch

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
# Image to Prompt
# -----------------------
def image_to_prompt(image):
    """
    Takes a PIL image as input and uses the CLIP model
    to extract a text description from the image.

    Returns:
        str: Extracted text description.
    """

    # Make sure the image is in RGB format
    image = image.convert("RGB")

    # Use CLIP Interrogator to generate a text description
    text = ci.interrogate_fast(image)

    # Return the extracted text as a string
    return str(text)


# -----------------------
# Take image path from command line and display the image
# -----------------------
import argparse

parser = argparse.ArgumentParser(
    description="Image search using CLIP and Word2Vec"
)

parser.add_argument(
    "image_path",
    help="Path to the image file"
)

args = parser.parse_args()

image_path = args.image_path

# Open and display the image using PIL
image = Image.open(image_path)
image.show()

# -----------------------
# Prompt user for search text
# -----------------------
user_input = input("Enter the text you want to search for: ")
print(f"You entered: {user_input}")


# -----------------------
# Convert image to text using CLIP
# -----------------------
extracted_text = image_to_prompt(image)

print("\nText extracted from image using CLIP:")
print(extracted_text)


# -----------------------
# Find similar words using trained Word2Vec model
# -----------------------

# Load the trained Word2Vec model
model_path = "word2vec_model.model"
model = Word2Vec.load(model_path)

# Find most similar words
try:
    similar_words = model.wv.most_similar(user_input, topn=10)

    print("\nMost similar words to your input:")

    for word, similarity in similar_words:
        print(f"{word} - similarity: {similarity:.2f}")

except KeyError:
    print(
        f"The word '{user_input}' is not in the Word2Vec vocabulary."
    )


# -----------------------
# Optional fuzzy matching
# -----------------------
# for word, similarity in similar_words:
#     confidence = fuzz.partial_ratio(
#         user_input.lower(),
#         word.lower()
#     )
#
#     if confidence >= 80:
#         print(
#             f"Fuzzy match: {word} "
#             f"(Confidence: {confidence}%)"
#         )

