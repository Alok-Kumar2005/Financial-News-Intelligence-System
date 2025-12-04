import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set in environment variables")

    DATABASE_URL = os.getenv("DATABASE_URL")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD"))
    ENTITY_TYPES = ["COMPANY", "SECTOR", "REGULATOR", "PERSON", "EVENT"]
    CONFIDENCE_LEVELS = {
        "direct": 1.0,
        "sector_high": 0.8,
        "sector_medium": 0.6,
        "regulatory": 0.5,
        "indirect": 0.3
    }

    MAX_RESULTS = 10
    CONTEXT_EXPANSION_ENABLED = True

config = Config()

# 