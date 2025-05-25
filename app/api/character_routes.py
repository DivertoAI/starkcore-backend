from fastapi import APIRouter, HTTPException
from app.services.character_service import create_character, list_characters

router = APIRouter()

@router.post("/create")
async def create_character_route(data: dict):
    try:
        return create_character(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def list_all_characters():
    return list_characters()