"""Application settings and environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for document ingestion and question answering."""

    anthropic_api_key: str | None
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    documents_dir: Path
    chroma_dir: Path

    def __post_init__(self) -> None:
        """Normalize and validate runtime settings."""
        object.__setattr__(self, "documents_dir", Path(self.documents_dir))
        object.__setattr__(self, "chroma_dir", Path(self.chroma_dir))
        if self.anthropic_api_key is not None:
            cleaned = str(self.anthropic_api_key).strip()
            object.__setattr__(self, "anthropic_api_key", cleaned or None)

        if not self.model_name or not str(self.model_name).strip():
            raise ValueError("Model name must not be empty.")
        if not self.embedding_model or not str(self.embedding_model).strip():
            raise ValueError("Embedding model must not be empty.")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
            )
        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than zero.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from the project environment and `.env` file."""
    load_dotenv(PROJECT_ROOT / ".env")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k = int(os.getenv("TOP_K", "4"))
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
    if top_k <= 0:
        raise ValueError("TOP_K must be greater than zero.")

    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        documents_dir=Path(
            os.getenv("DOCUMENTS_DIR", str(PROJECT_ROOT / "data" / "documents"))
        ).resolve(),
        chroma_dir=Path(
            os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))
        ).resolve(),
    )
