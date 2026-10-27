"""Tests for similarity retrieval without loading external models."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from documind.config import Settings
from documind.retriever import search_documents


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.query: str | None = None
        self.k: int | None = None

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.query = query
        self.k = k
        return self.documents[:k]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key=None,
        model_name="test-model",
        embedding_model="test-embedding",
        chunk_size=100,
        chunk_overlap=10,
        top_k=2,
        documents_dir=tmp_path / "documents",
        chroma_dir=tmp_path / "chroma",
    )


def test_search_documents_uses_query_and_configured_top_k(tmp_path: Path) -> None:
    documents = [
        Document(page_content="One", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="Two", metadata={"source": "b.pdf", "page": 1}),
        Document(page_content="Three", metadata={"source": "c.pdf", "page": 2}),
    ]
    store = FakeVectorStore(documents)

    results = search_documents(
        "What is relevant?", make_settings(tmp_path), vector_store=store
    )

    assert store.query == "What is relevant?"
    assert store.k == 2
    assert results == documents[:2]


def test_search_documents_rejects_blank_query(
    tmp_path: Path,
) -> None:
    store = FakeVectorStore([])

    with pytest.raises(ValueError, match="must not be empty"):
        search_documents("  ", make_settings(tmp_path), vector_store=store)
