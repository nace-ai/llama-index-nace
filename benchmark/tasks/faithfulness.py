"""Faithfulness on HaluEval QA (MIT): the right answer is faithful, the hallucinated one is not."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .common import fetch

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
SEED = 20261028


@dataclass(frozen=True)
class FaithfulnessExample:
    """One (question, answer, knowledge) triple and whether the answer is faithful."""

    id: str
    query: str
    response: str
    contexts: list[str]
    label: bool


def _length_matched(row: dict[str, str]) -> bool:
    right, wrong = len(row["right_answer"].split()), len(row["hallucinated_answer"].split())
    return right >= 2 and wrong >= 2 and abs(right - wrong) <= 1


def load(items: int = 250, *, matched: bool = False) -> list[FaithfulnessExample]:
    """A fixed-seed sample of HaluEval items, two examples each (shuffled).

    With `matched`, only items whose two answers differ by at most one word (both at least two
    words) are eligible, which removes the answer-length shortcut. The pool may be smaller
    than `items`; then all of it is used.
    """
    path = fetch("halueval_qa.json", URL)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pool = list(enumerate(rows))
    if matched:
        pool = [(i, r) for i, r in pool if _length_matched(r)]
    rng = random.Random(SEED)
    sample = rng.sample(pool, min(items, len(pool)))
    examples = []
    for i, row in sample:
        for kind, answer, label in (
            ("right", row["right_answer"], True),
            ("hallucinated", row["hallucinated_answer"], False),
        ):
            examples.append(
                FaithfulnessExample(
                    f"{i}-{kind}", row["question"], answer, [row["knowledge"]], label
                )
            )
    rng.shuffle(examples)
    return examples
