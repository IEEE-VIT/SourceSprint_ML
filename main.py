import argparse

from image_utils import load_and_show_image, image_to_prompt
from nlp_utils import preprocess_text, fuzzy_match
from model_utils import load_model, find_similar_words


def main():
    parser = argparse.ArgumentParser(
        description="Image search using CLIP and Word2Vec"
    )
    parser.add_argument(
        "image_path",
        help="Path to the image file"
    )
    args = parser.parse_args()

    # Load and display the image
    image = load_and_show_image(args.image_path)

    # Prompt user for search text
    user_input = input("Enter the text you want to search for: ")
    print(f"You entered: {user_input}")

    # Convert image to text using CLIP
    extracted_text = image_to_prompt(image)
    print("\nText extracted from image using CLIP:")
    print(extracted_text)

    # Preprocess extracted text
    tokens = preprocess_text(extracted_text)

    # Load Word2Vec model and find similar words
    model = load_model("word2vec_model.model")
    similar_words = find_similar_words(model, user_input, topn=10)

    if similar_words is not None:
        print("\nMost similar words to your input:")
        for word, similarity in similar_words:
            print(f"{word} - similarity: {similarity:.2f}")

        # Optional fuzzy matching against similar words
        matches = fuzzy_match(user_input, [w for w, _ in similar_words])
        for word, confidence in matches:
            print(f"Fuzzy match: {word} (Confidence: {confidence}%)")
    else:
        print(f"The word '{user_input}' is not in the Word2Vec vocabulary.")

    # Fuzzy match against preprocessed extracted-text tokens too
    token_matches = fuzzy_match(user_input, tokens)
    if token_matches:
        print("\nMatch found in picture")
        for word, confidence in token_matches:
            print(f"Fuzzy match: {word} (Confidence: {confidence}%)")


if __name__ == "__main__":
    main()
