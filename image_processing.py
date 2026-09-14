"""Image loading and display helpers."""

import os


def load_image(image_path):
    """Load an image as RGB and close the underlying file."""
    from PIL import Image

    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        return image.convert("RGB")


def display_image(image):
    """Display an image with the system image viewer."""
    image.show()
