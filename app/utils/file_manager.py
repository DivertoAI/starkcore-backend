from PIL import Image

def save_image(image: Image.Image, path: str):
    image.save(path)

def save_text(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)