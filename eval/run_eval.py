"""Run the configured question set against the local DocuMind index."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TypedDict

from documind.chain import answer_question
from documind.config import get_settings

logger = logging.getLogger(__name__)
QUESTIONS_PATH = Path(__file__).with_name("questions.json")


class EvaluationQuestion(TypedDict):
    question: str
    expected_answer: str


def _normalize(text: str) -> str:
    """Normalize text for a simple case-insensitive expected-phrase match."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvaluationQuestion]:
    """Read and validate question/expected-answer pairs from JSON."""
    with path.open(encoding="utf-8") as question_file:
        raw_questions = json.load(question_file)
    if not isinstance(raw_questions, list):
        raise ValueError("Evaluation questions must be a JSON list.")

    questions: list[EvaluationQuestion] = []
    for index, item in enumerate(raw_questions):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("question"), str)
            or not item["question"].strip()
            or not isinstance(item.get("expected_answer"), str)
            or not item["expected_answer"].strip()
        ):
            raise ValueError(f"Invalid question/expected_answer pair at index {index}.")
        questions.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
            }
        )
    return questions


def main() -> None:
    """Evaluate expected answer phrase hit rate for the configured question set."""
    logging.basicConfig(level=logging.INFO)
    questions = load_questions()
    if not questions:
        raise ValueError("No evaluation questions found in eval/questions.json.")

    hits = 0
    settings = get_settings()
    for item in questions:
        result = answer_question(item["question"], settings=settings)
        hit = _normalize(item["expected_answer"]) in _normalize(result.answer)
        hits += int(hit)
        print(
            f"{'HIT' if hit else 'MISS'}: {item['question']}\n"
            f"  expected: {item['expected_answer']}\n"
            f"  answer:   {result.answer}\n"
        )

    print(f"Hit rate: {hits}/{len(questions)} ({hits / len(questions):.1%})")


if __name__ == "__main__":
    main()
