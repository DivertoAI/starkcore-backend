def build_prompt(data: dict) -> str:
    """
    Build a descriptive prompt from character data
    for image and story generation.
    """
    prompt_parts = [
        f"{data.get('name', 'Character')},",
        f"{data.get('age', '')} years old,",
        f"{data.get('race', '')} race,",
        f"{data.get('gender', '')},",
        f"with {data.get('hairColor', '')} hair,",
        f"{data.get('hairStyle', '')} hairstyle,",
        f"{data.get('eyeColor', '')} eyes,",
        f"body type is {data.get('bodyType', '')},",
        f"boob size {data.get('boobSize', '')},",
        f"butt size {data.get('buttSize', '')}.",
        f"Personality: {data.get('personality', '')}.",
        f"Storyline background: {data.get('storylineBackground', '')}.",
        f"Setting: {data.get('setting', '')}.",
        f"Relationship type: {data.get('relationshipType', '')}."
    ]

    # Filter out empty strings and join nicely
    prompt = " ".join([part for part in prompt_parts if part.strip() != ""])
    return prompt