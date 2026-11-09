import os
from dotenv import load_dotenv

# Tự động nạp biến môi trường từ .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password_uit_secret@localhost:5432/jd_matching_db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GITHUB_PERSONAL_TOKEN = os.getenv("GITHUB_PERSONAL_TOKEN", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
AI_ENGINE_PORT = int(os.getenv("AI_ENGINE_PORT", "8001"))
