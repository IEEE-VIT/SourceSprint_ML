import os, subprocess
from sklearn import svm
from joblib import dump, load
from PIL import Image
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz
# Note: Extra modules may have to be imported
# TO-DO: Clip configuration

from clip_interrogator import Config, Interrogator
import torch

# -----------------------
# CLIP configuration (assume already done)
# -----------------------
clip_model_name = "ViT-L-14/openai"
caption_model_name = "blip-large"
ci_config = Config(clip_model_name=clip_model_name,
                   caption_model_name=caption_model_name)
ci = Interrogator(ci_config)

def image_to_prompt(image):
    """
    Takes a PIL image and returns extracted text using CLIP.
    """
    image = image.convert('RGB')  # Ensure RGB format
    text = ci.interrogate_fast(image)
    return text

# -----------------------
# TO-DO implemented: take user input and display the image
# -----------------------
image_path = "image.png"  # Replace with your test image path
image = Image.open(image_path)
image.show()

user_input = input("Enter the text you want to search for: ")
print(f"You entered: {user_input}")

# -----------------------
# New TO-DO: Find similar words using trained Word2Vec model
# -----------------------
# Load the trained Word2Vec model
model_path = "word2vec_model.model"  # Replace with your actual model path
model = Word2Vec.load(model_path)

# Find most similar words
try:
    similar_words = model.wv.most_similar(user_input, topn=10)
    print("\nMost similar words to your input:")
    for word, similarity in similar_words:
        print(f"{word} - similarity: {similarity:.2f}")
except KeyError:
    print(f"The word '{user_input}' is not in the Word2Vec vocabulary.")

# Optionally, use fuzz.partial_ratio to check similarity with user input
# for word, similarity in similar_words:
#     confidence = fuzz.partial_ratio(user_input.lower(), word.lower())
#     if confidence >= 80:
#         print(f"Fuzzy match: {word} (Confidence: {confidence}%)")
