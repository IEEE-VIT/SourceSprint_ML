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

def image_to_prompt(image):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    a = ci.interrogate_fast(image)
    return a

# -----------------------
# TO-DO implemented: Take user input and display the image
# -----------------------
image_path = "image.png"  # Replace with your test image path
image = Image.open(image_path)
image.show()

user_input = input("Enter the text you want to search for: ")
print(f"You entered: {user_input}")

# -----------------------
# Future code can continue here for text extraction, Word2Vec, fuzzy search, etc.
# -----------------------
