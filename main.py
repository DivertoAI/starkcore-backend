# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import character_routes, video_routes, payment_routes, routes

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(character_routes.router, prefix="/characters")
app.include_router(video_routes.router, prefix="/videos")
app.include_router(payment_routes.router, prefix="/payments")
app.include_router(routes.router)