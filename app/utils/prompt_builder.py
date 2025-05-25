def build_prompt(data: dict) -> str:
    return (
        f"A beautiful {data['race']} {data['gender']} with {data['hairColor']} hair, "
        f"{data['eyeColor']} eyes, a {data['bodyType']} body type, "
        f"{data['boobSize']} bust and {data['buttSize']} hips, "
        f"styled with {data['hairStyle']} hair. Personality: {data['personality']}."
    )