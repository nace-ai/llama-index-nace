"""Shared logic of the yes/no evaluators (faithfulness, relevancy)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import nace_sdk

from ._drex.answers import DrexResponse, NoulAnswer
from .base import NaceEvaluationResult, NaceEvaluator, _Request, answer_of


class BinaryEvaluator(NaceEvaluator):
    """A `noul` question judged against a threshold."""

    _criteria: ClassVar[dict[str, str]]
    _label: str
    _needs_query: bool = False

    def __init__(
        self,
        *,
        instructions: str,
        threshold: float = 0.5,
        raise_error: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create the evaluator; `kwargs` go to `NaceEvaluator`.

        Raises:
            ValueError: `threshold` is not between 0 and 1.

        """
        if not 0.0 <= threshold <= 1.0:
            message = f"threshold must be between 0 and 1 (got {threshold})."
            raise ValueError(message)
        super().__init__(instructions=instructions, **kwargs)
        self._threshold = threshold
        self._raise_error = raise_error

    def _question(self) -> nace_sdk.Noul:
        return nace_sdk.Noul(
            instructions=self._instructions.format(),
            criteria=nace_sdk.NoulCriteria(**self._criteria),  # type: ignore[typeddict-item]
        )

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
        if self._needs_query and query is None:
            msg = "query, response and contexts (or images) must be provided"
            raise ValueError(msg)
        if response is None or not (contexts or images or image_paths or image_urls):
            msg = "response and contexts (or images) must be provided"
            raise ValueError(msg)
        return super()._request(query, response, contexts, images, image_paths, image_urls, kwargs)

    def _result(
        self,
        response: DrexResponse,
        query: str | None,
        response_str: str | None,
        contexts: Sequence[str] | None,
    ) -> NaceEvaluationResult:
        answer = answer_of(response, NoulAnswer)
        p = answer.noul
        passing = p >= self._threshold
        if self._raise_error and not passing:
            msg = "The response is invalid"
            raise ValueError(msg)
        return NaceEvaluationResult(
            query=query,
            response=response_str,
            contexts=contexts,
            score=p,
            passing=passing,
            feedback=f"P({self._label})={p:.2f}",
            probability=p,
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
        )
