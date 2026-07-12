from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chat_model: str = os.getenv("CHAT_MODEL", "llama3.2")
    chroma_persist_dir: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", PROJECT_ROOT / "chroma_db")
    )
    collection_name: str = os.getenv("COLLECTION_NAME", "pdf_documents")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    fetch_k: int = int(os.getenv("FETCH_K", "20"))


settings = Settings()
