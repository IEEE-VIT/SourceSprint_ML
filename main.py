import argparse
import string
import nltk
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

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