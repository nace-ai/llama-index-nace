"""Relevancy evaluator."""

from __future__ import annotations

from typing import Any, ClassVar

from ._binary import BinaryEvaluator

INSTRUCTIONS = "Does the response answer the query, in line with the context?"


class NaceRelevancyEvaluator(BinaryEvaluator):
    """Relevancy, mirroring LlamaIndex's `RelevancyEvaluator`; `score` is Drex's P(yes).

    Args:
        threshold: Minimum probability to pass (default 0.5).
        raise_error: Raise `ValueError` when the result fails, as LlamaIndex does.
        **kwargs: `model`, `api_key`, `base_url`, `timeout`, `max_images`, `client`, ...
            A custom `base_url` needs the provider's `api_key`; OpenRouter uses
            `OPENROUTER_API_KEY`.

    """

    _label = "relevant"
    _needs_query = True
    _criteria: ClassVar[dict[str, str]] = {
        "true": "The response answers the query and is consistent with the context.",
        "false": "The response does not answer the query, or is not in line with the context.",
    }

    def __init__(self, *, threshold: float = 0.5, raise_error: bool = False, **kwargs: Any) -> None:
        """Create the evaluator."""
        super().__init__(
            instructions=INSTRUCTIONS,
            threshold=threshold,
            raise_error=raise_error,
            **kwargs,
        )
