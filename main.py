from PIL import Image
import pytesseract
import nltk
import string
import re

# Make sure the tokenizer resources are available
nltk.download('punkt')

# Step 1: Load the image
image_path = "/mnt/data/acd90d71-000f-4917-a066-0e8f9d161af9.png"
image = Image.open(image_path)

# Step 2: Extract text using pytesseract
extracted_text = pytesseract.image_to_string(image)

print("Extracted Text:")
print(extracted_text)

# Step 3: Tokenize the text
tokens = nltk.word_tokenize(extracted_text)

# Step 4: Clean the tokens
cleaned_tokens = [
    re.sub(r'[^a-zA-Z0-9]', '', token).lower()
    for token in tokens
    if re.sub(r'[^a-zA-Z0-9]', '', token) != ''
]

print("\nCleaned Tokens:")
print(cleaned_tokens)
