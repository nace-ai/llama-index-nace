"""Routing: public questions from four domains, one tool per domain.

Sources (all public): GSM8K test (MIT), MBPP (CC BY 4.0), PubMedQA labelled (MIT),
SQuAD v1.1 dev (CC BY-SA 4.0).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from llama_index.core.tools import ToolMetadata

from .common import fetch

SEED = 20261028
PER_DOMAIN = 80

TOOLS = [
    ToolMetadata(
        name="math_word_problems",
        description="Solves arithmetic and algebra word problems that need step-by-step "
        "calculation and a numeric answer.",
    ),
    ToolMetadata(
        name="python_programming",
        description="Writes and explains Python functions and code for a described "
        "programming task.",
    ),
    ToolMetadata(
        name="biomedical_literature",
        description="Answers biomedical and clinical research questions from PubMed "
        "abstracts: diseases, treatments, cells, genes and trials.",
    ),
    ToolMetadata(
        name="wikipedia_knowledge",
        description="Answers general-knowledge factual questions about history, geography, "
        "people, culture, sport and other non-medical encyclopedic topics.",
    ),
]


@dataclass(frozen=True)
class RoutingExample:
    """A query, the tools in this query's order, and the index of the correct one."""

    id: str
    query: str
    tools: list[ToolMetadata]
    label: int


def _gsm8k() -> list[str]:
    path = fetch(
        "gsm8k.jsonl",
        "https://raw.githubusercontent.com/openai/grade-school-math/master/"
        "grade_school_math/data/test.jsonl",
    )
    return [json.loads(line)["question"] for line in path.read_text().splitlines() if line]


def _mbpp() -> list[str]:
    path = fetch(
        "mbpp.jsonl",
        "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
    )
    return [json.loads(line)["text"] for line in path.read_text().splitlines() if line]


def _pubmedqa() -> list[str]:
    path = fetch(
        "pqal.json",
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    )
    return [row["QUESTION"] for row in json.loads(path.read_text()).values()]


def _squad() -> list[str]:
    path = fetch("squad.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
    data = json.loads(path.read_text())["data"]
    return [qa["question"] for article in data for p in article["paragraphs"] for qa in p["qas"]]


def load(per_domain: int = PER_DOMAIN) -> list[RoutingExample]:
    """`per_domain` fixed-seed queries per tool; tool order is shuffled per query."""
    rng = random.Random(SEED)
    sources = [_gsm8k(), _mbpp(), _pubmedqa(), _squad()]
    examples = []
    for tool_index, questions in enumerate(sources):
        for qi in rng.sample(range(len(questions)), per_domain):
            question = questions[qi]
            order = list(range(len(TOOLS)))
            rng.shuffle(order)
            examples.append(
                RoutingExample(
                    f"{TOOLS[tool_index].name}-{qi}",
                    question,
                    [TOOLS[i] for i in order],
                    order.index(tool_index),
                )
            )
    rng.shuffle(examples)
    return examples
