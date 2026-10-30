"""Strict faithfulness evaluator."""

from __future__ import annotations

from typing import Any, ClassVar

from ._binary import BinaryEvaluator

INSTRUCTIONS = "Is every claim in the response supported by the context?"


class NaceFaithfulnessEvaluator(BinaryEvaluator):
    """Strict faithfulness: `score` is Drex's P(yes), `passing` is `P >= threshold`.

    Args:
        threshold: Minimum probability to pass (default 0.5).
        raise_error: Raise `ValueError` when the result fails, as LlamaIndex does.
        **kwargs: `model`, `api_key`, `base_url`, `timeout`, `max_images`, `client`, ...
            A custom `base_url` needs the provider's `api_key`; OpenRouter uses
            `OPENROUTER_API_KEY`.

    """

    _label = "faithful"
    _criteria: ClassVar[dict[str, str]] = {
        "true": "Every factual claim in the response is stated in, or directly follows from, "
        "the context.",
        "false": "At least one claim in the response is missing from or contradicts the context.",
    }

    def __init__(self, *, threshold: float = 0.5, raise_error: bool = False, **kwargs: Any) -> None:
        """Create the evaluator."""
        super().__init__(
            instructions=INSTRUCTIONS,
            threshold=threshold,
            raise_error=raise_error,
            **kwargs,
        )
