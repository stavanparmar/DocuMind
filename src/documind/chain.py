"""Grounded answer generation with citations and chat-history support."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from documind.config import Settings, get_settings
from documind.memory import ChatTurn, rewrite_follow_up_question
from documind.retriever import search_documents

logger = logging.getLogger(__name__)

NOT_IN_DOCUMENTS = "That information is not in the provided documents."


@dataclass
class AnswerResult:
    """Generated response together with its rewritten query and evidence."""

    answer: str
    standalone_question: str
    documents: list[Document]


def _create_llm(settings: Settings) -> ChatAnthropic:
    """Create a Claude chat model after validating required credentials."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the project .env file."
        )
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
    )


def _page_number(document: Document) -> int:
    """Safely coerce metadata page numbers to a one-based page index."""
    page = document.metadata.get("page", 0)
    try:
        return int(page) + 1
    except (TypeError, ValueError):
        if page is None:
            return 1
        text = str(page).strip()
        if not text:
            return 1
        try:
            return int(float(text)) + 1
        except ValueError:
            return 1


def _citation(document: Document) -> str:
    """Format a one-based PDF page citation for a retrieved chunk."""
    source = str(document.metadata.get("source") or "Unknown file").strip() or "Unknown file"
    page = _page_number(document)
    return f"[{source}, p.{page}]"


def _context(documents: list[Document]) -> str:
    """Label retrieved text so the model can cite its supporting page."""
    return "\n\n".join(
        f"Source: {_citation(document)}\n{document.page_content}"
        for document in documents
    )


def answer_question(
    question: str,
    history: list[ChatTurn] | None = None,
    settings: Settings | None = None,
) -> AnswerResult:
    """Rewrite a follow-up, retrieve evidence, and generate a cited answer."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    current_settings = settings or get_settings()
    llm = _create_llm(current_settings)
    standalone_question = rewrite_follow_up_question(question, history or [], llm)
    documents = search_documents(standalone_question, current_settings)

    if not documents:
        logger.info("No documents retrieved for question: %s", standalone_question)
        return AnswerResult(
            answer=NOT_IN_DOCUMENTS,
            standalone_question=standalone_question,
            documents=[],
        )

    prompt = [
        SystemMessage(
            content=(
                "Answer the user's question only from the supplied document context. "
                "If the context does not contain the answer, reply exactly: "
                f'"{NOT_IN_DOCUMENTS}" Do not use outside knowledge or guess. '
                "For every factual claim you make, cite its supporting source using "
                "the exact format [file, p.N]. Only cite sources present in the "
                "context. If multiple chunks support the answer, include each "
                "relevant citation."
            )
        ),
        HumanMessage(
            content=f"Document context:\n{_context(documents)}\n\n"
            f"Question: {standalone_question}"
        ),
    ]
    response = llm.invoke(prompt)
    answer = str(response.content).strip()
    if not answer:
        raise RuntimeError("Claude returned an empty answer.")
    logger.info("Generated an answer using %d retrieved chunks", len(documents))
    return AnswerResult(
        answer=answer,
        standalone_question=standalone_question,
        documents=documents,
    )
