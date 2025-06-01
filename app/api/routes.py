import httpx
from fastapi import APIRouter
from app.models.user import User
from app.services.user_service import create_user

router = APIRouter()

# Existing user registration route
@router.post("/register")
def register_user(user: User):
    return create_user(user)

# New route for video generation
@router.get("/generate_video")
async def generate_video():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://127.0.0.1:8081/generate_video")
    return response.json()

# ✅ Health check route
@router.get("/ping")
async def ping():
    return {"message": "StarkCore backend is alive ⚡"}