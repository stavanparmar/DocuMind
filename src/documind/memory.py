"""Chat-history formatting and follow-up question rewriting."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable


ChatTurn = dict[str, str]


def format_chat_history(history: Sequence[ChatTurn]) -> str:
    """Format prior user and assistant messages as readable conversation text."""
    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in history
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    )


def rewrite_follow_up_question(
    question: str, history: Sequence[ChatTurn], llm: Runnable
) -> str:
    """Resolve references in a follow-up using chat history before retrieval."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    conversation = format_chat_history(history)
    if not conversation:
        return question.strip()

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Rewrite the latest user question as a standalone question using "
                    "the conversation only to resolve references. Do not answer it. "
                    "Keep the original meaning and return only the rewritten question."
                )
            ),
            HumanMessage(
                content=f"Conversation:\n{conversation}\n\n"
                f"Latest question:\n{question.strip()}"
            ),
        ]
    )
    rewritten = str(response.content).strip()
    if not rewritten:
        raise ValueError("The model returned an empty standalone question.")
    return rewritten
