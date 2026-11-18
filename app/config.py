import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "docLens"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/chroma")
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Parametri RF-01
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".txt"]

settings = Settings()