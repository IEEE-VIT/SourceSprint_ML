from PIL import Image
from clip_interrogator import Config, Interrogator

clip_model_name = "ViT-L-14/openai"
caption_model_name = "blip-large"

ci_config = Config(
    clip_model_name=clip_model_name,
    caption_model_name=caption_model_name
)

ci = Interrogator(ci_config)


def load_and_show_image(image_path):
    """
    Opens an image from disk and displays it.

    Returns:
        PIL.Image: The opened image.
    """
    image = Image.open(image_path)
    image.show()
    return image


def image_to_prompt(image):
    """
    Takes a PIL image as input and uses the CLIP model
    to extract a text description from the image.

    Returns:
        str: Extracted text description.
    """
    image = image.convert("RGB")
    text = ci.interrogate_fast(image)
    return str(text)
