"""`DrexClient`: one place that talks to Drex, shared by selectors and evaluators."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import nace_sdk

from .answers import DrexResponse
from .errors import (
    NaceAPIResponseValidationError,
    NaceImageError,
    NaceImageLimitError,
    NaceImageModelError,
    convert_error,
)
from .images import as_data_url, to_drex_image
from .origins import is_openrouter_host, resolve_api_key, resolve_destination

# Updated by hand when Drex ships a new default model; see "Releasing" in the repository README.
DEFAULT_MODEL = "drex-max-v1.1"
IMAGE_MODEL = "drex-max-v1.1"
# OpenRouter serves Drex Max as `nace-ai/drex-max`, so that is the default there.
OPENROUTER_DEFAULT_MODEL = "drex-max"
MIN_TIMEOUT = 60.0
OPENROUTER_PREFIX = "nace-ai/"

# Facts about released, pinned versions. Every other model id (new releases, aliases such as
# drex-latest) is sent as given and Drex decides, including the per-request question limit.
TEXT_ONLY_MODELS = frozenset({"drex-v1.5", "drex-v2.0"})
MAX_IMAGES = {"drex-max": 8, "drex-max-v1.1": 8}

Question = nace_sdk.Noul | nace_sdk.Choice | nace_sdk.Score


def _normalize(model: str) -> str:
    """The bare model id: no vendor prefix, no leading `~`."""
    return model.rsplit("/", 1)[-1].lstrip("~")


def _set_user_agent(request: httpx.Request) -> None:
    """Send the client name as `User-Agent` too; the SDK sets it as `X-Nace-Client`.

    Read from the request, not from a `DrexClient`, so one httpx client shared by several
    `DrexClient`s sends each one's own name and the hook is installed once.
    """
    name = request.headers.get("X-Nace-Client")
    if name:
        request.headers["User-Agent"] = name


async def _aset_user_agent(request: httpx.Request) -> None:
    _set_user_agent(request)


class DrexClient:
    """Sync and async access to `POST /v1/systemone` with the checks Drex documents."""

    def __init__(
        self,
        client_name: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = MIN_TIMEOUT,
        max_retries: int = 2,
        openrouter: bool | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Build the sync and async SDK clients.

        `http_client` and `http_async_client` stay the caller's: `close()` and `aclose()`
        close only the httpx clients this object created.

        Raises:
            ValueError: `timeout` is under 60 s, `model` is blank, `openrouter=True` without
                a `base_url`, the destination is not a valid http(s) URL, no key may be used
                for it (a custom `base_url=` needs `api_key=`; OpenRouter needs its own key),
                or the SDK refuses the configuration.

        """
        if timeout < MIN_TIMEOUT:
            message = f"timeout must be at least {MIN_TIMEOUT:g} seconds; Drex may take 55 s."
            raise ValueError(message)
        if model is not None and not model.strip():
            message = "model must not be empty."
            raise ValueError(message)
        self.client_name = client_name
        self.model = model.strip() if model else None
        self.timeout = timeout
        self.model_kwargs: dict[str, Any] = dict(model_kwargs or {})
        destination = resolve_destination(base_url)
        self.base_url: str = destination.url
        self.openrouter = (
            openrouter if openrouter is not None else is_openrouter_host(self.base_url)
        )
        if self.openrouter and destination.source == "default":
            message = "openrouter=True needs a base_url, such as https://openrouter.ai/api/v1."
            raise ValueError(message)
        resolved_key = resolve_api_key(api_key, destination)

        headers = dict(default_headers or {})
        if self.openrouter:
            headers.setdefault("HTTP-Referer", "https://nace.ai")
            headers.setdefault("X-OpenRouter-Title", "Drex for LlamaIndex")
        self._owns_http = http_client is None
        self._owns_async_http = http_async_client is None
        sync_http = http_client or httpx.Client(timeout=timeout)
        if _set_user_agent not in sync_http.event_hooks["request"]:
            sync_http.event_hooks["request"].append(_set_user_agent)
        async_http = http_async_client or httpx.AsyncClient(timeout=timeout)
        if _aset_user_agent not in async_http.event_hooks["request"]:
            async_http.event_hooks["request"].append(_aset_user_agent)
        options: dict[str, Any] = {
            "api_key": resolved_key,
            "base_url": self._sdk_base_url(),
            # timeout=None: the default 30 s total budget would stop retries after a slow
            # attempt, which is the normal case for Drex (60 s floor); max_retries bounds it.
            "retry": nace_sdk.RetryPolicy(max_retries=max_retries, timeout=None),
            "timeout": timeout,
            "headers": headers,
            "client_info": client_name,
        }
        try:
            self._client = nace_sdk.NaceClient(**options, http_client=sync_http)
            self._aclient = nace_sdk.AsyncNaceClient(**options, http_client=async_http)
        except nace_sdk.NaceError as error:
            raise ValueError(str(error)) from error

    def close(self) -> None:
        """Close the sync httpx client if this object created it; a supplied one stays open."""
        if self._owns_http:
            self._client.close()

    async def aclose(self) -> None:
        """Close both httpx clients this object created; supplied ones stay open."""
        self.close()
        if self._owns_async_http:
            await self._aclient.aclose()

    def _sdk_base_url(self) -> str:
        if self.openrouter:
            # OpenRouter documents `.../api/v1`; the SDK appends `/v1/systemone` itself.
            return self.base_url.rstrip("/").removesuffix("/v1")
        return self.base_url

    def resolve_model(self, *, has_images: bool) -> str:
        """The model id a request is sent with.

        An explicit `model` is sent exactly as given. Otherwise the default is
        `NACE_DEFAULT_DECISION_MODEL` when set, else `DEFAULT_MODEL` (`drex-max-v1.1`), or
        `OPENROUTER_DEFAULT_MODEL` on OpenRouter. When the request has images and that default
        is a known text-only model, the backend's image model is used instead. On OpenRouter a
        default without a vendor gets the `nace-ai/` prefix.
        """
        if self.model:
            return self.model
        default = os.environ.get("NACE_DEFAULT_DECISION_MODEL", "").strip()
        if not default:
            default = OPENROUTER_DEFAULT_MODEL if self.openrouter else DEFAULT_MODEL
        if has_images and _normalize(default) in TEXT_ONLY_MODELS:
            default = self._image_model()
        # A default that already names a vendor (`nace-ai/drex-v2.0`) is sent as is.
        if self.openrouter and "/" not in default:
            return OPENROUTER_PREFIX + default
        return default

    def _image_model(self) -> str:
        return OPENROUTER_DEFAULT_MODEL if self.openrouter else IMAGE_MODEL

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        images: Sequence[Any] = (),
    ) -> DrexResponse:
        """Ask every question about `state` (and `images`) in one request.

        Raises:
            NaceImageModelError: Images were sent to a text-only model.
            NaceImageLimitError: Too many images for the model.
            NaceAPIError: Drex refused the request (subclasses per status).
            NaceAPIConnectionError: Drex could not be reached.
            NaceAPIResponseValidationError: Drex answered with a body that can't be read.

        """
        call = self._prepare(state, questions, images)
        try:
            result = self._client.system_one(**call)
        except nace_sdk.NaceError as error:
            raise convert_error(error) from _cause(error)
        return _response(result)

    async def aask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        images: Sequence[Any] = (),
    ) -> DrexResponse:
        """Async `ask`."""
        call = self._prepare(state, questions, images)
        try:
            result = await self._aclient.system_one(**call)
        except nace_sdk.NaceError as error:
            raise convert_error(error) from _cause(error)
        return _response(result)

    def _prepare(
        self, state: Any, questions: Mapping[str, Question], images: Sequence[Any]
    ) -> dict[str, Any]:
        if not questions:
            message = "At least one question is required."
            raise ValueError(message)
        model = self.resolve_model(has_images=bool(images))
        bare = _normalize(model)
        # Hints keep any vendor prefix: an explicit model is sent as given (OpenRouter ids).
        prefix = model.removesuffix(bare)
        if images and bare in TEXT_ONLY_MODELS:
            message = (
                f"{model} reads text only, so it can't take images. "
                f"Use an image model such as model='{prefix}{self._image_model()}'."
            )
            raise NaceImageModelError(message)
        if bare in MAX_IMAGES and len(images) > MAX_IMAGES[bare]:
            message = (
                f"{model} accepts at most {MAX_IMAGES[bare]} images per request "
                f"(got {len(images)})."
            )
            raise NaceImageLimitError(message)
        converted = [to_drex_image(image) for image in images]
        if converted and self.openrouter:
            state, converted = _openrouter_state(state, converted), []
        return {
            "state": state,
            "questions": {name: _question(q) for name, q in questions.items()},
            "model": model,
            "images": converted or None,
            "extra_body": self.model_kwargs or None,
        }


def _cause(error: nace_sdk.NaceError) -> nace_sdk.NaceError | None:
    """The SDK error to chain, or None when it could show response data in a traceback.

    An SDK `NaceAPIError`'s `str` falls back to the raw response body (HTML pages verbatim),
    and the SDK's validation error wraps pydantic's, which quotes response values. Our
    converted error already keeps status, request id, type, issues, headers and body.
    Connection and timeout errors carry no response, so they stay chained.
    """
    if isinstance(error, nace_sdk.NaceAPIError | nace_sdk.NaceAPIResponseValidationError):
        return None
    return error


def _question(question: Question) -> Question:
    """Drex rejects `instructions: null` and OpenRouter requires the key, so send `""`."""
    if question.instructions is None:
        return question.model_copy(update={"instructions": ""})
    return question


def _response(result: nace_sdk.SystemOneResponse) -> DrexResponse:
    raw = result.raw_http_response
    try:
        body = raw.json()
        if isinstance(body, dict):
            return DrexResponse.from_body(body, raw.headers.get("x-request-id"))
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    # Raised outside the `except` so no exception context is attached: pydantic's
    # ValidationError quotes the offending response values, which must never surface.
    message = "Drex returned a response this package can't read."
    raise NaceAPIResponseValidationError(message)


def _openrouter_state(state: Any, images: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenRouter takes images as `image_url` parts of a `state` content array."""
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": state if isinstance(state, str) else json.dumps(state)}
    ]
    for image in images:
        if not isinstance(image, str):
            message = "OpenRouter accepts inline images only, not uploaded workspace files."
            raise NaceImageError(message)
        parts.append({"type": "image_url", "image_url": {"url": as_data_url(image)}})
    return parts


__all__ = [
    "DEFAULT_MODEL",
    "IMAGE_MODEL",
    "MAX_IMAGES",
    "OPENROUTER_DEFAULT_MODEL",
    "TEXT_ONLY_MODELS",
    "DrexClient",
]
