from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.character_service import create_character, list_characters, get_character_by_id
import os
import json
import traceback

router = APIRouter()

CHARACTER_DIR = "/workspace/characters"  # Make sure this matches your config

class CharacterData(BaseModel):

    name: str
    age: int
    gender: str
    hairColor: str
    eyeColor: str
    bodyType: str
    boobSize: str
    buttSize: str
    hairStyle: str
    personality: str
    personalityDescription: str = ""
    storylineBackground: str
    setting: str
    relationshipType: str
    race: str
    image_url: Optional[str] = None
    id: Optional[str] = None 

@router.post("/create", response_model=CharacterData)
async def create_character_route(data: CharacterData):
    try:
        result = create_character(data.dict())
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[CharacterData])
async def list_all_characters():
    try:
        return list_characters()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{character_id}")
def get_character(character_id: str):
    try:
        character = get_character_by_id(character_id)
        if not character:
            raise HTTPException(status_code=404, detail="Character not found")
        return character
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))