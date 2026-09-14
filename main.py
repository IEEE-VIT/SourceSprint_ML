import argparse

from image_processing import display_image, image_to_prompt, load_image
from model_inference import (
    find_similar_words,
    load_word2vec_model,
    print_similar_words,
)
from nlp_preprocessing import preprocess_text


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract text from an image and find similar words."
    )
    parser.add_argument("image_path", help="Path to the image to process")
    parser.add_argument(
        "--model-path",
        default="word2vec_model.model",
        help="Path to the trained Word2Vec model",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        image = load_image(args.image_path)
    except FileNotFoundError:
        print(f"Error: The file '{args.image_path}' was not found.")
        return 1

    display_image(image)
    extracted_text = image_to_prompt(image)
    processed_tokens = preprocess_text(extracted_text)
    print(f"Extracted tokens: {processed_tokens}")

    user_input = input("Enter the text you want to search for: ")
    print(f"You entered: {user_input}")

    try:
        model = load_word2vec_model(args.model_path)
    except FileNotFoundError:
        print(f"Error: The trained Word2Vec model '{args.model_path}' was not found.")
        return 1

    try:
        similar_words = find_similar_words(model, user_input)
        print_similar_words(similar_words)
    except KeyError:
        print(f"The word '{user_input}' is not in the Word2Vec vocabulary.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
