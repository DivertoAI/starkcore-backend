from fastapi import APIRouter, HTTPException
from app.services.video_service import generate_video

router = APIRouter()

@router.post("/generate")
async def generate_video_route(data: dict):
    try:
        return generate_video(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))