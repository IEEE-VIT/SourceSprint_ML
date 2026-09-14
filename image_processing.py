from PIL import Image
from clip_interrogator import Config, Interrogator


clip_model_name = "ViT-L-14/openai"
caption_model_name = "blip-large"


def load_image(image_path):
    """Open and return an image from the provided path."""
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
    interrogator = Interrogator(config)
    return interrogator.interrogate_fast(image.convert("RGB"))
