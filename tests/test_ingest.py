"""Tests for document chunking."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from documind.chain import _citation
from documind.config import Settings
from documind.ingest import split_documents


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key=None,
        model_name="test-model",
        embedding_model="test-embedding",
        chunk_size=40,
        chunk_overlap=10,
        top_k=2,
        documents_dir=tmp_path / "documents",
        chroma_dir=tmp_path / "chroma",
    )


def test_split_documents_preserves_metadata_and_page_text(tmp_path: Path) -> None:
    source = Document(
        page_content="0123456789 " * 12,
        metadata={"source": "sample.pdf", "page": 2},
    )

    chunks = split_documents([source], make_settings(tmp_path))

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert all(chunk.metadata["start_index"] >= 0 for chunk in chunks)


def test_split_documents_returns_no_chunks_for_empty_page(tmp_path: Path) -> None:
    assert split_documents(
        [Document(page_content="   ", metadata={"page": 0})],
        make_settings(tmp_path),
    ) == []


def test_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(
            anthropic_api_key=None,
            model_name="test-model",
            embedding_model="test-embedding",
            chunk_size=10,
            chunk_overlap=10,
            top_k=2,
            documents_dir="documents",
            chroma_dir="chroma",
        )


def test_citation_falls_back_when_page_metadata_is_invalid() -> None:
    doc = Document(
        page_content="Example text",
        metadata={"source": "sample.pdf", "page": "not-a-page"},
    )

    assert _citation(doc) == "[sample.pdf, p.1]"
