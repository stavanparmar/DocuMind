"""Streamlit user interface for the DocuMind RAG application."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from documind.chain import AnswerResult, _page_number, answer_question
from documind.config import get_settings
from documind.ingest import ingest_directory
from documind.memory import ChatTurn
from documind.retriever import delete_indexed_file, list_indexed_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
st.title("DocuMind")
st.caption("Ask questions about your PDFs and get answers grounded in their contents.")

settings = get_settings()
settings.documents_dir.mkdir(parents=True, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Your documents")
    try:
        indexed_files = list_indexed_files(settings)
    except Exception as exc:
        logger.exception("Could not load indexed document list")
        st.error(f"Could not load indexed documents: {exc}")
        indexed_files = []

    uploaded_files = st.file_uploader(
        "Add PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        indexed_names = set(indexed_files)
        for uploaded_file in uploaded_files:
            filename = Path(uploaded_file.name).name
            if filename != uploaded_file.name:
                st.error(f"Invalid PDF filename: {uploaded_file.name}")
                continue
            if filename in indexed_names:
                st.info(f"{filename} is already indexed; delete it before replacing it.")
                continue
            destination = settings.documents_dir / filename
            try:
                destination.write_bytes(uploaded_file.getvalue())
            except OSError as exc:
                logger.exception("Could not save uploaded PDF %s", filename)
                st.error(f"Could not save {filename}: {exc}")
        report = ingest_directory(settings)
        if report.indexed:
            st.success(f"Indexed {len(report.indexed)} file(s).")
        if report.skipped:
            st.info(f"Already indexed: {', '.join(report.skipped)}")
        for filename, error in report.errors.items():
            st.error(f"{filename}: {error}")
        try:
            indexed_files = list_indexed_files(settings)
        except Exception as exc:
            logger.exception("Could not refresh indexed document list")
            st.error(f"Could not refresh indexed documents: {exc}")

    if indexed_files:
        st.write("Indexed files:")
        for filename in indexed_files:
            file_col, delete_col = st.columns([3, 1])
            file_col.write(filename)
            if delete_col.button("Delete", key=f"delete-{filename}"):
                try:
                    delete_indexed_file(filename, settings)
                except Exception as exc:
                    logger.exception("Could not delete %s", filename)
                    st.error(f"Could not delete {filename}: {exc}")
                else:
                    st.rerun()
    else:
        st.caption("No indexed PDFs yet.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        chunks: list[Document] = message.get("documents", [])
        if chunks:
            with st.expander(f"Retrieved chunks ({len(chunks)})"):
                for index, document in enumerate(chunks, start=1):
                    source = str(document.metadata.get("source") or "Unknown file").strip() or "Unknown file"
                    page = _page_number(document)
                    st.markdown(f"**[{source}, p.{page}] — chunk {index}**")
                    st.write(document.page_content)

if prompt := st.chat_input("Ask a question about your documents"):
    history: list[ChatTurn] = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            result: AnswerResult = answer_question(prompt, history, settings)
        except Exception as exc:
            logger.exception("Question answering failed")
            st.error(f"Could not answer the question: {exc}")
        else:
            st.markdown(result.answer)
            if result.documents:
                with st.expander(f"Retrieved chunks ({len(result.documents)})"):
                    for index, document in enumerate(result.documents, start=1):
                        source = str(document.metadata.get("source") or "Unknown file").strip() or "Unknown file"
                        page = _page_number(document)
                        st.markdown(f"**[{source}, p.{page}] — chunk {index}**")
                        st.write(document.page_content)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "documents": result.documents,
                }
            )
