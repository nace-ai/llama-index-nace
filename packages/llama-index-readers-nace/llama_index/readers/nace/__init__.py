"""LlamaIndex reader that parses documents with NDI (Nace Document Intelligence)."""

from ._drex.errors import (
    NaceAPIConnectionError,
    NaceAPIError,
    NaceAPIResponseValidationError,
    NaceAPITimeoutError,
    NaceAuthenticationError,
    NaceBadRequestError,
    NaceError,
    NaceImageError,
    NaceImageLimitError,
    NaceImageModelError,
    NaceInsufficientCreditError,
    NaceInternalServerError,
    NaceLowConfidenceError,
    NaceNotFoundError,
    NaceOverloadedError,
    NacePermissionDeniedError,
    NaceRateLimitError,
    NaceRedirectError,
    NaceUnprocessableEntityError,
)
from ._version import __version__
from .base import NaceReader

__all__ = [
    "NaceAPIConnectionError",
    "NaceAPIError",
    "NaceAPIResponseValidationError",
    "NaceAPITimeoutError",
    "NaceAuthenticationError",
    "NaceBadRequestError",
    "NaceError",
    "NaceImageError",
    "NaceImageLimitError",
    "NaceImageModelError",
    "NaceInsufficientCreditError",
    "NaceInternalServerError",
    "NaceLowConfidenceError",
    "NaceNotFoundError",
    "NaceOverloadedError",
    "NacePermissionDeniedError",
    "NaceRateLimitError",
    "NaceReader",
    "NaceRedirectError",
    "NaceUnprocessableEntityError",
    "__version__",
]
