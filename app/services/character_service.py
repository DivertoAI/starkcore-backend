import os
import json
from PIL import Image
from app.utils.prompt_builder import build_prompt
from app.utils.file_manager import save_image, save_text

CHARACTER_DIR = "/workspace/characters"

def create_character(data):
    os.makedirs(f"{CHARACTER_DIR}/{data['name']}", exist_ok=True)

    # Build prompt (stub)
    prompt = build_prompt(data)

    # Generate image and story (stub placeholders)
    image = Image.new("RGB", (512, 768), color="pink")
    story = f"This is {data['name']}'s story."

    # Save files
    save_image(image, f"{CHARACTER_DIR}/{data['name']}/{data['name']}.png")
    save_text(story, f"{CHARACTER_DIR}/{data['name']}/story.txt")

    # Save metadata
    metadata = {**data, "image_url": f"/characters/{data['name']}/{data['name']}.png"}
    with open(f"{CHARACTER_DIR}/{data['name']}/metadata.json", "w") as f:
        json.dump(metadata, f)

    return metadata

def list_characters():
    results = []
    for folder in os.listdir(CHARACTER_DIR):
        try:
            with open(f"{CHARACTER_DIR}/{folder}/metadata.json") as f:
                results.append(json.load(f))
        except Exception:
            continue
    return results