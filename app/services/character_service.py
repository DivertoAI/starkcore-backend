import os
import json
import uuid
from PIL import Image
from datetime import datetime
from app.utils.prompt_builder import build_prompt
from app.utils.file_manager import save_image, save_text
from app.core.config import CHARACTER_DIR  # central config path
from app.services.handler import handler as runpod_handler


def create_character(data):
    timestamp = int(datetime.now().timestamp())
    character_id = f"{data['name']}_{timestamp}"
    character_folder = os.path.join(CHARACTER_DIR, character_id)
    os.makedirs(character_folder, exist_ok=True)

    # Build prompt string
    prompt = build_prompt(data)

    # Generate image using handler
    generation_result = runpod_handler({
        "mode": "txt2img",
        "prompt": prompt,
        "guidance_scale": 7.5,
        "steps": 30
    })

    tmp_image_path = generation_result["image_paths"][0]  # Don't overwrite
    image = Image.open(tmp_image_path)

    # Save final image
    image_filename = f"{data['name']}.png"
    final_image_path = os.path.join(character_folder, image_filename)
    save_image(image, final_image_path)

    # Save simple story
    story = f"This is {data['name']}'s story."
    story_path = os.path.join(character_folder, "story.txt")
    save_text(story, story_path)

    # Metadata
    metadata = {
        **data,
        "id": character_id,
        "image_url": f"/characters/{character_id}/{image_filename}",
        "avatar": f"/characters/{character_id}/{image_filename}",
        "tagline": data.get("personalityDescription", "Let's get to know each other."),
        "category": "Boyfriend" if data.get("gender") == "Male" else "Girlfriend"
    }

    metadata_path = os.path.join(character_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return metadata

def list_characters():
    results = []
    if not os.path.exists(CHARACTER_DIR):
        return results

    for folder in os.listdir(CHARACTER_DIR):
        metadata_path = os.path.join(CHARACTER_DIR, folder, "metadata.json")
        try:
            with open(metadata_path) as f:
                results.append(json.load(f))
        except Exception:
            continue

    return results


def get_character_by_id(character_id: str):
    """
    Find and return a character by its ID.
    """
    if not os.path.exists(CHARACTER_DIR):
        return None

    for folder in os.listdir(CHARACTER_DIR):
        metadata_path = os.path.join(CHARACTER_DIR, folder, "metadata.json")
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                if metadata.get("id") == character_id:
                    return metadata
        except Exception:
            continue

    return None