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
#Note: Extra modules may have to be imported
#TO-DO: Clip configuration
def image_to_prompt(image):
    ci.config.chunk_size = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    ci.config.flavor_intermediate_count = 2048 if ci.config.clip_model_name == "ViT-L-14/openai" else 1024
    image = image.convert('RGB')
    a= ci.interrogate_fast(image)
    return a
#TO-DO: take user input and display the image that youre using to test and 


#Search for similar words using fuzz.partial_ratio and a confidence threshold and print if its similar to user input or not
import fuzzywuzzy
from fuzzywuzzy import fuzz

user_input = "apple"
similar_words = ["apples", "banana", "orange", "grape"]
threshold = 80  # Adjust the threshold as needed
similar_words_filtered = [word for word in similar_words if fuzz.partial_ratio(user_input, word) >= threshold]

if similar_words_filtered:
    print(f"Similar words to '{user_input}': {', '.join(similar_words_filtered)}")
else:
    print(f"No similar words found for '{user_input}'")