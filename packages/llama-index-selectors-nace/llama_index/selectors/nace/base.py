"""`NaceSingleSelector`: route a query to one tool with a Drex decision model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import nace_sdk
from llama_index.core.base.base_selector import BaseSelector, SelectorResult, SingleSelection
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import ToolMetadata

from ._drex.answers import ChoiceAnswer, DrexResponse, NoulAnswer, Usage
from ._drex.client import DrexClient
from ._drex.errors import (
    NaceAPIConnectionError,
    NaceAPIError,
    NaceAPIResponseValidationError,
    NaceError,
    NaceInternalServerError,
    NaceLowConfidenceError,
    NaceRateLimitError,
)
from ._version import __version__

CLIENT_NAME = f"llama-index-selectors-nace/{__version__}"
DEFAULT_SINGLE_INSTRUCTIONS = "Pick the option best suited to answer the user's query."
DEFAULT_MULTI_INSTRUCTIONS = (
    "Is the option below needed to answer the user's query? Option {label}: {description}"
)
QUESTION_ID = "route"

# Failures a fallback route can absorb: Drex is down, busy or unreachable. Authentication,
# credit and request errors are the caller's to fix, so they always propagate.
_TRANSIENT = (NaceInternalServerError, NaceRateLimitError, NaceAPIConnectionError)


class NaceSelectorResult(SelectorResult):
    """A selection with Drex's probabilities and request metadata."""

    probabilities: dict[str, float]
    labels: list[str]
    confidence: float | None = None
    model: str
    request_id: str | None = None
    usage: Usage
    fell_back: bool = False


def option_labels(choices: Sequence[ToolMetadata]) -> list[str]:
    """Tool names when every one is present and unique, else `option_1..n`."""
    names = [choice.name for choice in choices]
    if all(names) and len(set(names)) == len(names):
        return [str(name) for name in names]
    return [f"option_{i}" for i in range(1, len(choices) + 1)]


def _check_choices(choices: Sequence[ToolMetadata]) -> None:
    if not choices:
        message = "choices must not be empty; pass at least one tool to select from."
        raise ValueError(message)


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        message = f"{name} must be between 0 and 1 (got {value})."
        raise ValueError(message)


def _check_default_index(default_index: int | None) -> None:
    if default_index is None:
        return
    if isinstance(default_index, bool) or not isinstance(default_index, int) or default_index < 0:
        message = f"default_index must be a non-negative integer (got {default_index!r})."
        raise ValueError(message)


def _query_images(query: QueryBundle) -> list[str]:
    return [query.image_path] if query.image_path else []


def _p(value: float) -> str:
    return f"{value:.2f}"


class _NaceSelector(BaseSelector):
    """Shared plumbing: the Drex client and the editable `instructions` prompt."""

    def __init__(self, instructions: str, client: DrexClient | None, **client_kwargs: Any) -> None:
        super().__init__()
        self._instructions = PromptTemplate(instructions)
        self._client = client if client is not None else DrexClient(CLIENT_NAME, **client_kwargs)

    def _get_prompts(self) -> PromptDictType:
        return {"instructions": self._instructions}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        instructions = prompts_dict.get("instructions")
        if instructions is not None:
            self._instructions = instructions  # type: ignore[assignment]


class NaceSingleSelector(_NaceSelector):
    """Pick one tool for a query with Drex instead of an LLM.

    Args:
        instructions: What Drex is asked to pick.
        min_confidence: Lowest top probability to accept; below it `default_index` is used,
            or `NaceLowConfidenceError` is raised when there is none.
        default_index: The tool to use when Drex is unsure or unavailable (5xx, rate
            limit, connection errors).
        client: A prepared `DrexClient`. Otherwise one is built from `client_kwargs`
            (`api_key`, `base_url`, `model`, `timeout`, ...). A custom `base_url` needs the
            provider's `api_key`; OpenRouter uses `OPENROUTER_API_KEY`.

    """

    def __init__(
        self,
        *,
        instructions: str = DEFAULT_SINGLE_INSTRUCTIONS,
        min_confidence: float | None = None,
        default_index: int | None = None,
        client: DrexClient | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create the selector.

        Raises:
            ValueError: `min_confidence` is outside 0..1 or `default_index` is not a
                non-negative integer.

        """
        if min_confidence is not None:
            _check_probability("min_confidence", min_confidence)
        _check_default_index(default_index)
        super().__init__(instructions, client, **client_kwargs)
        self._min_confidence = min_confidence
        self._default_index = default_index

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> NaceSingleSelector:
        """Same as the constructor, for symmetry with LlamaIndex's selectors."""
        return cls(**kwargs)

    def _request(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> tuple[dict[str, nace_sdk.Choice], list[str], list[str]]:
        _check_choices(choices)
        if self._default_index is not None and self._default_index >= len(choices):
            message = (
                f"default_index={self._default_index} is out of range for {len(choices)} choices."
            )
            raise ValueError(message)
        labels = option_labels(choices)
        question = nace_sdk.Choice(
            instructions=self._instructions.format(),
            # A missing or empty description is sent as null, which Drex reads as "no text".
            criteria={
                label: choice.description or None
                for label, choice in zip(labels, choices, strict=True)
            },
        )
        images = _query_images(query)
        return {QUESTION_ID: question}, labels, images

    def _select(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        questions, labels, images = self._request(choices, query)
        try:
            response = self._client.ask(query.query_str, questions, images=images)
        except _TRANSIENT as error:
            return self._fallback(error, labels, bool(images))
        return self._result(response, labels)

    async def _aselect(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        questions, labels, images = self._request(choices, query)
        try:
            response = await self._client.aask(query.query_str, questions, images=images)
        except _TRANSIENT as error:
            return self._fallback(error, labels, bool(images))
        return self._result(response, labels)

    def _fallback(self, error: NaceError, labels: list[str], has_images: bool) -> SelectorResult:
        if self._default_index is None:
            raise error
        status = error.status if isinstance(error, NaceAPIError) else None
        cause = f"Drex returned {status}" if status else type(error).__name__
        return NaceSelectorResult(
            selections=[
                SingleSelection(
                    index=self._default_index,
                    reason=f"{cause}; used default_index={self._default_index}",
                )
            ],
            probabilities={},
            labels=labels,
            model=self._client.resolve_model(has_images=has_images),
            request_id=error.request_id if isinstance(error, NaceAPIError) else None,
            usage=Usage(),
            fell_back=True,
        )

    def _result(self, response: DrexResponse, labels: list[str]) -> SelectorResult:
        answer = response.answers.get(QUESTION_ID)
        if not isinstance(answer, ChoiceAnswer) or answer.choice not in labels:
            message = "Drex's answer doesn't match any of the options."
            raise NaceAPIResponseValidationError(message)
        index = labels.index(answer.choice)
        top = answer.probabilities.get(answer.choice, answer.confidence)
        fell_back = False
        reason = f"p={_p(top)}" if top is not None else "selected by Drex"
        if self._min_confidence is not None and top is not None and top < self._min_confidence:
            if self._default_index is None:
                message = (
                    f"Drex's best option scored {_p(top)}, below min_confidence="
                    f"{self._min_confidence}."
                )
                raise NaceLowConfidenceError(message, answer.probabilities)
            index, fell_back = self._default_index, True
            reason = (
                f"p={_p(top)} below min_confidence={self._min_confidence}; "
                f"used default_index={self._default_index}"
            )
        return NaceSelectorResult(
            selections=[SingleSelection(index=index, reason=reason)],
            probabilities=answer.probabilities,
            labels=labels,
            confidence=answer.confidence,
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
            fell_back=fell_back,
        )


class NaceMultiSelector(_NaceSelector):
    """Select every tool Drex judges relevant, one yes/no question per tool.

    All questions go in a single request. Tools are returned most likely first. If none
    reaches `threshold`, the single most likely tool is returned with `fell_back=True`.

    The default question asks whether each tool is *needed*, which is conservative on purely
    definitional questions ("what does 429 mean?"): tools can score below 0.5 and the result
    falls back to the top one. Lower `threshold`, or pass your own `instructions`.

    Args:
        instructions: Template for each question; `{label}` and `{description}` are filled.
        threshold: Lowest probability to select a tool.
        max_outputs: Keep at most this many tools.
        client: A prepared `DrexClient`. Otherwise one is built from `client_kwargs`
            (`api_key`, `base_url`, `model`, `timeout`, ...). A custom `base_url` needs the
            provider's `api_key`; OpenRouter uses `OPENROUTER_API_KEY`.

    """

    def __init__(
        self,
        *,
        instructions: str = DEFAULT_MULTI_INSTRUCTIONS,
        threshold: float = 0.5,
        max_outputs: int | None = None,
        client: DrexClient | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Create the selector.

        Raises:
            ValueError: `threshold` is outside 0..1 or `max_outputs` is under 1.

        """
        _check_probability("threshold", threshold)
        if max_outputs is not None and max_outputs < 1:
            message = f"max_outputs must be at least 1 (got {max_outputs})."
            raise ValueError(message)
        super().__init__(instructions, client, **client_kwargs)
        self._threshold = threshold
        self._max_outputs = max_outputs

    @classmethod
    def from_defaults(cls, **kwargs: Any) -> NaceMultiSelector:
        """Same as the constructor, for symmetry with LlamaIndex's selectors."""
        return cls(**kwargs)

    def _request(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> tuple[dict[str, nace_sdk.Noul], list[str], list[str]]:
        _check_choices(choices)
        labels = option_labels(choices)
        questions = {
            label: nace_sdk.Noul(
                instructions=self._instructions.format(
                    label=label, description=choice.description or ""
                )
            )
            for label, choice in zip(labels, choices, strict=True)
        }
        return questions, labels, _query_images(query)

    def _select(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        questions, labels, images = self._request(choices, query)
        return self._result(self._client.ask(query.query_str, questions, images=images), labels)

    async def _aselect(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        questions, labels, images = self._request(choices, query)
        response = await self._client.aask(query.query_str, questions, images=images)
        return self._result(response, labels)

    def _result(self, response: DrexResponse, labels: list[str]) -> SelectorResult:
        probabilities: dict[str, float] = {}
        for label in labels:
            answer = response.answers.get(label)
            if not isinstance(answer, NoulAnswer):
                message = "Drex's answers don't match the options."
                raise NaceAPIResponseValidationError(message)
            probabilities[label] = answer.noul
        ranked = sorted(range(len(labels)), key=lambda i: -probabilities[labels[i]])
        chosen = [i for i in ranked if probabilities[labels[i]] >= self._threshold]
        fell_back = not chosen
        if fell_back:
            top = ranked[0]
            reasons = {
                top: f"p={_p(probabilities[labels[top]])}, below threshold "
                f"{self._threshold}; top option"
            }
            chosen = [top]
        else:
            reasons = {i: f"p={_p(probabilities[labels[i]])}" for i in chosen}
        if self._max_outputs is not None:
            chosen = chosen[: self._max_outputs]
        return NaceSelectorResult(
            selections=[SingleSelection(index=i, reason=reasons[i]) for i in chosen],
            probabilities=probabilities,
            labels=labels,
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
            fell_back=fell_back,
        )
