import os
from PIL import Image

def save_image(image: Image.Image, path: str):
    """
    Save a PIL Image to the given path.
    Creates directories if they don't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format="PNG")

def save_text(text: str, path: str):
    """
    Save a string as a text file.
    Creates directories if they don't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)