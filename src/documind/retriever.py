"""Persistent Chroma vector-store access and similarity retrieval."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from documind.config import Settings, get_settings


class SearchStore(Protocol):
    """Minimal interface required for similarity retrieval."""

    def similarity_search(self, query: str, k: int) -> list[Document]: ...


@lru_cache(maxsize=2)
def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Create and cache the local embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=4)
def _get_persistent_store(
    directory: str, embedding_model: str
) -> Chroma:
    """Open a persistent Chroma collection."""
    return Chroma(
        collection_name="documind",
        embedding_function=_get_embeddings(embedding_model),
        persist_directory=directory,
    )


def get_vector_store(settings: Settings | None = None) -> Chroma:
    """Return the persistent Chroma vector store for the configured paths."""
    current_settings = settings or get_settings()
    current_settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return _get_persistent_store(
        str(current_settings.chroma_dir), current_settings.embedding_model
    )


def search_documents(
    query: str,
    settings: Settings | None = None,
    *,
    vector_store: SearchStore | None = None,
) -> list[Document]:
    """Return the most relevant document chunks for a standalone query."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")
    current_settings = settings or get_settings()
    store = vector_store or get_vector_store(current_settings)
    return store.similarity_search(query, k=current_settings.top_k)


def list_indexed_files(settings: Settings | None = None) -> list[str]:
    """Return sorted unique PDF names represented in the vector store."""
    store = get_vector_store(settings)
    records = store.get(include=["metadatas"])
    metadatas = records.get("metadatas") or []
    names = {
        Path(str(metadata["source"])).name
        for metadata in metadatas
        if metadata and metadata.get("source")
    }
    return sorted(names, key=str.casefold)


def delete_indexed_file(
    filename: str, settings: Settings | None = None
) -> bool:
    """Remove a PDF's chunks and its local copy; return whether it was indexed."""
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise ValueError("Expected a PDF filename, not a path.")
    current_settings = settings or get_settings()
    store = get_vector_store(current_settings)
    records = store.get(
        where={"source": filename},
        include=["metadatas"],
    )
    ids = records["ids"]
    if ids:
        store.delete(ids=ids)

    pdf_path = current_settings.documents_dir / filename
    if pdf_path.exists():
        pdf_path.unlink()
    return bool(ids)
