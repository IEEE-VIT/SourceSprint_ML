"""Image loading and display helpers."""


def load_image(image_path):
    """Load an image as RGB and close the underlying file."""
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


def display_image(image):
    """Display an image with the system image viewer."""
    image.show()
