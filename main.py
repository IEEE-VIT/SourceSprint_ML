import os, subprocess
import argparse
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

from clip_interrogator import Config, Interrogator
import torch

nltk.download('punkt')
nltk.download('stopwords')

parser = argparse.ArgumentParser(description="Image search using text input")
parser.add_argument("image_path", help="Path to the image")
args = parser.parse_args()

image = Image.open(args.image_path)
image.show()

user_input = input("Enter the text you want to search for: ")
print(f"You entered: {user_input}")

def preprocess_text(raw_text):
    raw_text = raw_text.lower()
    raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(raw_text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def image_to_prompt(image):
    image = image.convert('RGB')
    text = ci.interrogate_fast(image)
    return text

image_path = args.image_path
image = Image.open(image_path)
image.show()

user_input = input("Enter the text you want to search for: ")
print(f"You entered: {user_input}")

processed_tokens = preprocess_text(user_input)

print("\nPreprocessed tokens:")
print(processed_tokens)

model_path = "word2vec_model.model"
model = Word2Vec.load(model_path)

for word in processed_tokens:
    try:
        similar_words = model.wv.most_similar(word, topn=10)

        print(f"\nMost similar words to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"{similar_word} - similarity: {similarity:.2f}")

    except KeyError:
        print(f"The word '{word}' is not in the Word2Vec vocabulary.")

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
