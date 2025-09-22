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
# Note: Extra modules may have to be imported
# TO-DO: Clip configuration

from clip_interrogator import Config, Interrogator
import torch

# Initialize CLIP once
ci_config = Config(clip_model_name="ViT-L-14/openai")
ci = Interrogator(ci_config)

def image_to_prompt(image):
    """
    Takes a PIL image and returns extracted text using CLIP.
    """
    image = image.convert('RGB')  # Ensure RGB format
    text = ci.interrogate_fast(image)
    return text


# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(raw_text):
    """
    Takes raw text as input and returns a list of preprocessed tokens.
    - Tokenizes text
    - Converts to lowercase
    - Removes punctuation
    - Removes stopwords
    """
    # Tokenize
    tokens = word_tokenize(raw_text.lower())

    # Remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    return tokens

image_path = "image.png"  # Replace with your image path
image = Image.open(image_path)
image.show()

user_input = input("Enter the text you want to search for: ")
print(f"You entered: {user_input}")

# -----------------------
# Rest of your original code can continue here
# -----------------------
