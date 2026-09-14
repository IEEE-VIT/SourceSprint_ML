import os

from PIL import Image
from clip_interrogator import Config, Interrogator


clip_model_name = "ViT-L-14/openai"
caption_model_name = "blip-large"


def load_image(image_path):
    """Open and return an image from the provided path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path)


def display_image(image):
    """Display a PIL image using the default image viewer."""
    image.show()


def image_to_prompt(image):
    """Run CLIP inference and return text extracted from the image."""
    config = Config(
        clip_model_name=clip_model_name,
        caption_model_name=caption_model_name,
    )
    try:
        interrogator = Interrogator(config)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Failed to load CLIP interrogator model files: {error}"
        ) from error
    return interrogator.interrogate_fast(image.convert("RGB"))
