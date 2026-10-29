"""LlamaIndex evaluators backed by Drex decision models from Nace."""

from ._drex.errors import (
    NaceAPIError,
    NaceError,
    NaceImageError,
    NaceImageLimitError,
    NaceImageModelError,
)
from ._version import __version__
from .base import NaceEvaluationResult
from .correctness import NaceCorrectnessEvaluator
from .faithfulness import NaceFaithfulnessEvaluator
from .relevancy import NaceRelevancyEvaluator

__all__ = [
    "NaceAPIError",
    "NaceCorrectnessEvaluator",
    "NaceError",
    "NaceEvaluationResult",
    "NaceFaithfulnessEvaluator",
    "NaceImageError",
    "NaceImageLimitError",
    "NaceImageModelError",
    "NaceRelevancyEvaluator",
    "__version__",
]
