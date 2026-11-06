import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")  # silently does nothing if .env doesn't exist

#path helper function to read paths from the environment, resolving relative paths from the project root
def _path(env_name: str, default: str) -> Path:
    """Read a path from the environment, resolving relative paths from the project root."""
    p = Path(os.getenv(env_name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p

#contains all the configuration settings for the application, read from environment variables or .env file
@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    upload_dir: Path = _path("UPLOAD_DIR", "data/uploads")
    chroma_dir: Path = _path("CHROMA_DIR", "data/chroma")


settings = Settings()