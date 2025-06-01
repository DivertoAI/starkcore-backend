# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Mach Backend"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    CHARACTER_DIR: str = os.getenv("CHARACTER_DIR", "./characters_data")  # More flexible

settings = Settings()


# Set the base directory where character data is stored
CHARACTER_DIR = os.getenv("CHARACTER_DIR", "./characters_data")