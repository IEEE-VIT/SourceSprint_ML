"""Command-line entry point for image description and Word2Vec search."""

import argparse

from image_processing import display_image, load_image
from model_inference import (
    find_similar_words,
    image_to_prompt,
    load_clip_model,
    load_word2vec_model,
)
from nlp_preprocessing import preprocess_text


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Image search using CLIP and Word2Vec"
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--model-path",
        default="word2vec_model.model",
        help="Path to the trained Word2Vec model",
    )
    args = parser.parse_args(argv)

    image = load_image(args.image_path)
    try:
        display_image(image)
        user_input = preprocess_text(input("Enter the text you want to search for: "))
        print(f"You entered: {user_input}")

        interrogator = load_clip_model()
        extracted_text = image_to_prompt(image, interrogator)
        print("\nText extracted from image using CLIP:")
        print(extracted_text)

        model = load_word2vec_model(args.model_path)
        try:
            similar_words = find_similar_words(model, user_input)
        except KeyError:
            print(f"The word '{user_input}' is not in the Word2Vec vocabulary.")
        else:
            print("\nMost similar words to your input:")
            for word, similarity in similar_words:
                print(f"{word} - similarity: {similarity:.2f}")
    finally:
        image.close()


if __name__ == "__main__":
    main()
