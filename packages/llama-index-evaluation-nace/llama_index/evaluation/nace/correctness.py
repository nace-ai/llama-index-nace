"""Correctness evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import nace_sdk

from ._drex.answers import DrexResponse, ScoreAnswer
from .base import NaceEvaluationResult, NaceEvaluator, _Request, answer_of

INSTRUCTIONS = (
    "How correct is the response as an answer to the query? "
    "Judge it against the reference answer when one is given."
)
LEVELS = [
    "Not relevant to the query",
    "Relevant but mostly incorrect",
    "Relevant with some mistakes",
    "Relevant and correct",
    "Relevant, correct and complete",
]


class NaceCorrectnessEvaluator(NaceEvaluator):
    """Correctness on LlamaIndex's 1-5 scale, from a 5-level Drex `score` question.

    `score` is 1 plus Drex's expected level index; `probability` is P(level 4 or 5).
    Pass `reference=` to `evaluate` / `aevaluate` to judge against a reference answer.

    Args:
        score_threshold: Minimum score to pass (default 4.0).
        **kwargs: `model`, `api_key`, `base_url`, `timeout`, `max_images`, `client`, ...
            A custom `base_url` needs the provider's `api_key`; OpenRouter uses
            `OPENROUTER_API_KEY`.

    """

    def __init__(self, *, score_threshold: float = 4.0, **kwargs: Any) -> None:
        """Create the evaluator."""
        super().__init__(instructions=INSTRUCTIONS, **kwargs)
        self._score_threshold = score_threshold

    def _question(self) -> nace_sdk.Score:
        return nace_sdk.Score(instructions=self._instructions.format(), criteria=list(LEVELS))

    def _extra_state(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return {"reference": kwargs.get("reference")}

    def _request(
        self,
        query: str | None,
        response: str | None,
        contexts: Sequence[str] | None,
        images: Sequence[Any] | None,
        image_paths: Sequence[Any] | None,
        image_urls: Sequence[str] | None,
        kwargs: Mapping[str, Any],
    ) -> _Request:
        if query is None or response is None:
            msg = "query and response must be provided"
            raise ValueError(msg)
        return super()._request(query, response, contexts, images, image_paths, image_urls, kwargs)

    def _result(
        self,
        response: DrexResponse,
        query: str | None,
        response_str: str | None,
        contexts: Sequence[str] | None,
    ) -> NaceEvaluationResult:
        answer = answer_of(response, ScoreAnswer)
        score = 1 + answer.score
        p_ge4 = sum(p for level, p in answer.probabilities.items() if level >= len(LEVELS) - 2)
        return NaceEvaluationResult(
            query=query,
            response=response_str,
            contexts=contexts,
            score=score,
            passing=score >= self._score_threshold,
            feedback=f"score={score:.2f}, P(≥4)={p_ge4:.2f}",
            probability=p_ge4,
            probabilities=dict(answer.probabilities),
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
        )
