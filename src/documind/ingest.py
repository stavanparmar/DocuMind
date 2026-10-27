"""PDF loading, chunking, and persistent vector-store ingestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documind.config import Settings, get_settings
from documind.retriever import get_vector_store, list_indexed_files

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Summary of documents processed by an ingestion operation."""

    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class DocumentIngestionError(Exception):
    """Raised when a PDF cannot be loaded or contains no extractable text."""


def split_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    """Split loaded pages into overlapping chunks while preserving metadata."""
    current_settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=current_settings.chunk_size,
        chunk_overlap=current_settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _load_pdf(pdf_path: Path) -> list[Document]:
    """Load a PDF and reject documents that have no extractable page text."""
    pages = PyPDFLoader(str(pdf_path)).load()
    pages_with_text = [
        page for page in pages if page.page_content and page.page_content.strip()
    ]
    if not pages_with_text:
        raise DocumentIngestionError(
            "No extractable text found; the PDF may be empty or scanned."
        )

    for page in pages_with_text:
        page.metadata["source"] = pdf_path.name
    return pages_with_text


def ingest_pdf(
    pdf_path: Path, settings: Settings | None = None
) -> int:
    """Index one PDF and return the number of chunks stored."""
    current_settings = settings or get_settings()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise DocumentIngestionError(f"Not a readable PDF file: {pdf_path}")

    chunks = split_documents(_load_pdf(pdf_path), current_settings)
    if not chunks:
        raise DocumentIngestionError("The PDF did not produce any text chunks.")

    store = get_vector_store(current_settings)
    store.add_documents(chunks)
    logger.info("Indexed %s as %d chunks", pdf_path.name, len(chunks))
    return len(chunks)


def ingest_directory(
    settings: Settings | None = None,
) -> IngestionReport:
    """Index unindexed PDFs in the configured document directory."""
    current_settings = settings or get_settings()
    current_settings.documents_dir.mkdir(parents=True, exist_ok=True)
    already_indexed = set(list_indexed_files(current_settings))
    report = IngestionReport()

    for pdf_path in sorted(current_settings.documents_dir.glob("*.pdf")):
        if pdf_path.name in already_indexed:
            report.skipped.append(pdf_path.name)
            continue
        try:
            ingest_pdf(pdf_path, current_settings)
            report.indexed.append(pdf_path.name)
        except Exception as exc:
            logger.exception("Could not index %s", pdf_path.name)
            report.errors[pdf_path.name] = str(exc)
    return report
