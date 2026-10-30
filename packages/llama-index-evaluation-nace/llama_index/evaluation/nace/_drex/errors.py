"""Errors raised by the Nace LlamaIndex integrations.

Messages never include the request or response body, so state cannot leak into logs.
"""

from __future__ import annotations

from typing import Any

import httpx
import nace_sdk

BILLING_URL = "https://console.nace.ai/dashboard/billing"
# The text of the SDK error raised when an API redirect changes origin (nace-sdk 0.6.2 raises it
# itself; `origins.guard_request` raises it on older SDKs).
REDIRECT_MESSAGE = "Refusing an API redirect to a different origin"
_REDIRECT_DETAIL = (
    f"{REDIRECT_MESSAGE} (scheme, host or port changed). Credentials and the request stay with "
    "the configured base_url; set base_url to the final API URL."
)


class NaceError(Exception):
    """Base class for every error this package raises."""


class NaceImageError(NaceError, ValueError):
    """An image can't be sent as given (a remote URL, a missing file, ...)."""


class NaceImageModelError(NaceImageError):
    """Images were sent to a model that reads text only."""


class NaceImageLimitError(NaceImageError):
    """A request holds more images than the model accepts."""


class NaceLowConfidenceError(NaceError):
    """Drex's best answer fell below the requested confidence."""

    def __init__(self, message: str, probabilities: dict[str, float] | None = None) -> None:
        """Keep the per-option probabilities next to the message."""
        super().__init__(message)
        self.probabilities: dict[str, float] = dict(probabilities or {})


class NaceAPIError(NaceError):
    """Drex answered with an unsuccessful HTTP status.

    `body` and `headers` hold the raw response and must not be logged; `str` and `repr` never
    include them.
    """

    def __init__(self, error: nace_sdk.NaceAPIError, message: str | None = None) -> None:
        """Wrap an SDK error, keeping its status, request id and issues."""
        super().__init__(message or _message(error))
        self.status: int = error.status
        self.request_id: str | None = error.request_id
        self.type: str | None = error.type
        self.issues: list[dict[str, str]] = error.issues
        self.headers: httpx.Headers = error.headers
        self.body: Any = error.body

    @property
    def status_code(self) -> int:
        """Alias for `status`."""
        return self.status

    def __repr__(self) -> str:
        """Describe the error without the response body."""
        return f"{type(self).__name__}({str(self)!r})"


class NaceBadRequestError(NaceAPIError):
    """HTTP 400."""


class NaceAuthenticationError(NaceAPIError):
    """HTTP 401: the API key is missing, wrong or revoked."""


class NaceInsufficientCreditError(NaceAPIError):
    """HTTP 402: the account has no available funds."""


class NacePermissionDeniedError(NaceAPIError):
    """HTTP 403."""


class NaceNotFoundError(NaceAPIError):
    """HTTP 404."""


class NaceUnprocessableEntityError(NaceAPIError):
    """HTTP 422: Drex rejected the request; `issues` says where."""


class NaceRateLimitError(NaceAPIError):
    """HTTP 429, after the retries ran out."""


class NaceInternalServerError(NaceAPIError):
    """HTTP 5xx, after the retries ran out."""


class NaceOverloadedError(NaceInternalServerError):
    """HTTP 529: Drex is overloaded; retried with `retry-after-ms` before raising."""


class NaceAPIConnectionError(NaceError):
    """Drex could not be reached."""


class NaceAPITimeoutError(NaceAPIConnectionError):
    """The request timed out."""


class NaceAPIResponseValidationError(NaceError):
    """Drex answered 200 with a body this package can't read."""


class NaceRedirectError(NaceError):
    """A redirect pointed to a different origin; it was refused before sending.

    Credentials and the request stayed with the configured base_url.
    """


_BY_STATUS: dict[int, type[NaceAPIError]] = {
    400: NaceBadRequestError,
    401: NaceAuthenticationError,
    402: NaceInsufficientCreditError,
    403: NacePermissionDeniedError,
    404: NaceNotFoundError,
    422: NaceUnprocessableEntityError,
    429: NaceRateLimitError,
    529: NaceOverloadedError,
}

_MESSAGE_KEYS = ("error", "message", "detail")


def _safe_detail(error: nace_sdk.NaceAPIError) -> str | None:
    """Return the server's message, but never a raw dump of a response body.

    Plain-text bodies (HTML error pages, proxy text) are never shown.

    The SDK falls back to the serialized body when it finds no message. That fallback can hold
    request state echoed back by the server, so it is dropped here.
    """
    body = error.body
    if not isinstance(body, dict):
        return None
    for key in _MESSAGE_KEYS:
        value = body.get(key)
        if isinstance(value, dict):
            value = value.get("message")
        if isinstance(value, str):
            return error.message or None
    return None


MAX_MESSAGE_CHARS = 500


def _clip(text: str, budget: int) -> str:
    """Cut `text` to at most `budget` characters, ending in an ellipsis when it was cut."""
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 1)] + "…"


def _message(error: nace_sdk.NaceAPIError) -> str:
    detail = _safe_detail(error)
    text = f"Drex returned {error.status}" + (f": {detail}" if detail else "")
    lines = [
        f"{i.get('path') or '(request)'}: {i['message']}" for i in error.issues if i.get("message")
    ]
    # Drex's top-level message often repeats a single issue; don't print it twice.
    if lines and lines != [detail] and [i.get("message") for i in error.issues] != [detail]:
        text += "\n" + "\n".join(f"  {line}" for line in lines)
    # Recovery guidance is kept whole when it fits; only the descriptive text is cut.
    guidance = ""
    if error.status == 402:
        guidance += f"\nAdd funds at {BILLING_URL}"
    if error.request_id:
        guidance += f"\n(request_id={error.request_id})"
    guidance = _clip(guidance, 300)
    return _clip(text, MAX_MESSAGE_CHARS - len(guidance)) + guidance


def convert_error(error: Exception) -> Exception:
    """Map an `nace_sdk` error to this package's error classes."""
    if isinstance(error, nace_sdk.NaceAPITimeoutError):
        return NaceAPITimeoutError(str(error))
    if isinstance(error, nace_sdk.NaceAPIConnectionError):
        return NaceAPIConnectionError(str(error))
    if isinstance(error, nace_sdk.NaceAPIResponseValidationError):
        return NaceAPIResponseValidationError(str(error))
    if isinstance(error, nace_sdk.NaceAPIError):
        cls = _BY_STATUS.get(error.status)
        if cls is None:
            cls = NaceInternalServerError if error.status >= 500 else NaceAPIError
        return cls(error, _message(error))
    if isinstance(error, nace_sdk.NaceError) and str(error).startswith(REDIRECT_MESSAGE):
        return NaceRedirectError(_REDIRECT_DETAIL)
    if isinstance(error, nace_sdk.NaceError):
        return NaceError(str(error))
    return error


__all__ = [
    "BILLING_URL",
    "REDIRECT_MESSAGE",
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
    "NaceRedirectError",
    "NaceUnprocessableEntityError",
    "convert_error",
]
