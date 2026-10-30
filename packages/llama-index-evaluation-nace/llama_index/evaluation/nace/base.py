"""`NaceEvaluator`: the shared base for evaluators that judge answers with Drex."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import nace_sdk
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType

from ._drex.answers import ChoiceAnswer, DrexResponse, NoulAnswer, RawAnswer, ScoreAnswer, Usage
from ._drex.client import DrexClient
from ._drex.errors import NaceAPIResponseValidationError
from ._drex.images import to_drex_image
from ._images import split_sources
from ._version import __version__

CLIENT_NAME = f"llama-index-evaluation-nace/{__version__}"
QUESTION_ID = "evaluate"

_Answer = TypeVar("_Answer", NoulAnswer, ScoreAnswer)


def answer_of(response: DrexResponse, kind: type[_Answer]) -> _Answer:
    """The evaluation answer, checked to be of `kind`.

    Raises:
        NaceAPIResponseValidationError: Drex's response has no answer to the question, or
            one of another type.

    """
    answer: NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer | None = response.answers.get(
        QUESTION_ID
    )
    if answer is None:
        message = "Drex's response has no answer to the evaluation question."
        raise NaceAPIResponseValidationError(message)
    if not isinstance(answer, kind):
        message = "Drex returned an unexpected answer type."
        raise NaceAPIResponseValidationError(message)
    return answer


class NaceEvaluationResult(EvaluationResult):
    """An evaluation with Drex's probabilities and request metadata."""

    probability: float | None = None
    probabilities: dict[int, float] | None = None
    model: str | None = None
    request_id: str | None = None
    usage: Usage | None = None


class _Request:
    """Everything `ask` needs, plus what `_result` reports back."""

    def __init__(
        self,
        query: str | None,
        response: str | None,
        contexts: Sequence[str] | None,
        state: dict[str, Any],
        images: list[Any],
    ) -> None:
        self.query = query
        self.response = response
        self.contexts = contexts
        self.state = state
        self.images = images


class NaceEvaluator(BaseEvaluator):
    """Base for Drex evaluators; subclasses supply `_question` and `_result`.

    Args:
        instructions: The question text; editable through `update_prompts`.
        max_images: Keep at most this many image sources, the highest node scores first.
            Without it, more than 8 image sources (Drex's limit) raise
            `NaceImageLimitError`.
        client: A prepared `DrexClient`. Otherwise one is built from `client_kwargs`
            (`model`, `api_key`, `base_url`, `timeout`, ...). A custom `base_url` needs the
            provider's `api_key`; OpenRouter uses `OPENROUTER_API_KEY`.

    """

    def __init__(
        self,
        *,
        instructions: str,
        max_images: int | None = None,
        client: DrexClient | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create the evaluator.

        Raises:
            ValueError: `max_images` is under 1.

        """
        if max_images is not None and max_images < 1:
            message = f"max_images must be at least 1 (got {max_images})."
            raise ValueError(message)
        self._instructions = PromptTemplate(instructions)
        self._max_images = max_images
        self._client = client if client is not None else DrexClient(CLIENT_NAME, **client_kwargs)

    def _get_prompts(self) -> PromptDictType:
        return {"instructions": self._instructions}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        instructions = prompts_dict.get("instructions")
        if instructions is not None:
            self._instructions = instructions  # type: ignore[assignment]

    def _question(self) -> nace_sdk.Noul | nace_sdk.Score:
        raise NotImplementedError

    def _result(
        self,
        response: DrexResponse,
        query: str | None,
        response_str: str | None,
        contexts: Sequence[str] | None,
    ) -> NaceEvaluationResult:
        raise NotImplementedError

    def _extra_state(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Extra state keys taken from `evaluate` kwargs (for example `reference`)."""
        return {}

    def _build_state(
        self,
        query: str | None,
        response: str | None,
        contexts: Sequence[str] | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = {
            "query": query,
            "context": list(contexts) if contexts is not None else None,
            "response": response,
            **extra,
        }
        return {key: value for key, value in state.items() if value is not None}

    def _images(
        self,
        images: Sequence[Any] | None,
        image_paths: Sequence[Any] | None,
        image_urls: Sequence[str] | None,
    ) -> list[Any]:
        items = [*(images or ()), *(image_paths or ()), *(image_urls or ())]
        return [to_drex_image(item) for item in items]

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
        state = self._build_state(query, response, contexts, self._extra_state(kwargs))
        return _Request(
            query, response, contexts, state, self._images(images, image_paths, image_urls)
        )

    def _finish(self, request: _Request, drex_response: DrexResponse) -> NaceEvaluationResult:
        return self._result(drex_response, request.query, request.response, request.contexts)

    def evaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        *,
        images: Sequence[Any] | None = None,
        image_paths: Sequence[Any] | None = None,
        image_urls: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate synchronously; safe to call where an event loop is already running."""
        request = self._request(query, response, contexts, images, image_paths, image_urls, kwargs)
        drex_response = self._client.ask(
            request.state, {QUESTION_ID: self._question()}, images=request.images
        )
        return self._finish(request, drex_response)

    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        *,
        images: Sequence[Any] | None = None,
        image_paths: Sequence[Any] | None = None,
        image_urls: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate asynchronously."""
        request = self._request(query, response, contexts, images, image_paths, image_urls, kwargs)
        drex_response = await self._client.aask(
            request.state, {QUESTION_ID: self._question()}, images=request.images
        )
        return self._finish(request, drex_response)

    def _from_response(
        self, response: Response | None, kwargs: dict[str, Any]
    ) -> tuple[str | None, list[str] | None, dict[str, Any]]:
        if response is None:
            return None, None, kwargs
        contexts, images = split_sources(response.source_nodes, max_images=self._max_images)
        extra_images = kwargs.pop("images", None) or []
        return response.response, contexts, {**kwargs, "images": [*images, *extra_images]}

    def evaluate_response(
        self,
        query: str | None = None,
        response: Response | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a `Response`; image source nodes are sent as images."""
        response_str, contexts, options = self._from_response(response, kwargs)
        return self.evaluate(query=query, response=response_str, contexts=contexts, **options)

    async def aevaluate_response(
        self,
        query: str | None = None,
        response: Response | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Async `evaluate_response`; what `BatchEvalRunner` calls."""
        response_str, contexts, options = self._from_response(response, kwargs)
        return await self.aevaluate(
            query=query, response=response_str, contexts=contexts, **options
        )
