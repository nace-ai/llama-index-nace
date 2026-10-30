"""Where a request may go and which key it may carry.

The integration, not the SDK, decides the destination and the API key, so an ambient Nace key
can only reach Nace production or an on-prem server the operator configured through the
environment.
"""

from __future__ import annotations

import os
from typing import Literal, NamedTuple
from urllib.parse import urlsplit, urlunsplit

import httpx

DEFAULT_BASE_URL = "https://console.nace.ai"
NACE_PRODUCTION_HOSTS = frozenset({"console.nace.ai", "drex.nace.ai"})
OPENROUTER_HOST = "openrouter.ai"

_REFUSED_CUSTOM = (
    "base_url points to a provider other than Nace, so pass that provider's key as api_key=. "
    "NACE_API_KEY is only sent to Nace, or to NACE_BASE_URL when you set both."
)
_OPENROUTER_KEY = (
    "OpenRouter needs api_key= or OPENROUTER_API_KEY; NACE_API_KEY is never sent to OpenRouter."
)
_NO_KEY = "No API key was provided. Pass api_key or set the NACE_API_KEY environment variable."
_BLANK_KEY = "api_key must not be empty."
_INVALID_URL = "base_url must be an absolute http(s) URL without credentials, query or fragment."
_INSECURE_NACE = "Nace's API is served over HTTPS on port 443; use https://console.nace.ai."


class Destination(NamedTuple):
    """The API origin a client talks to, and where the choice came from."""

    url: str
    source: Literal["argument", "environment", "default"]


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _host(hostname: str | None) -> str:
    return (hostname or "").lower().rstrip(".")


def _split(url: str) -> tuple[str, str, int | None] | None:
    """Scheme, normalized host and explicit port, or None when `url` can't be parsed."""
    try:
        parts = urlsplit(url)
        return parts.scheme.lower(), _host(parts.hostname), parts.port
    except ValueError:
        return None


def is_nace_production(url: str | httpx.URL) -> bool:
    """Whether `url` is https on port 443 of a Nace production host."""
    parsed = _split(str(url))
    if parsed is None:
        return False
    scheme, host, port = parsed
    return scheme == "https" and host in NACE_PRODUCTION_HOSTS and port in {None, 443}


def is_openrouter_host(url: str | None) -> bool:
    """Whether the host of `url` is exactly `openrouter.ai` (subdomains do not count)."""
    if not url:
        return False
    parsed = _split(str(url))
    return parsed is not None and parsed[1] == OPENROUTER_HOST


def _normalize(url: str) -> str:
    """Validate `url` and return it with a lower-case host and no trailing slash.

    Messages never echo `url`, which may carry a password or token.
    """
    try:
        parts = urlsplit(url)
        port = parts.port  # raises for a malformed or out-of-range port
    except ValueError:
        raise ValueError(_INVALID_URL) from None
    scheme = parts.scheme.lower()
    host = _host(parts.hostname)
    if (
        scheme not in {"http", "https"}
        or not host
        or "@" in parts.netloc
        or parts.query
        or parts.fragment
        or "?" in url
        or "#" in url
    ):
        raise ValueError(_INVALID_URL)
    if host in NACE_PRODUCTION_HOSTS and (scheme != "https" or port not in {None, 443}):
        raise ValueError(_INSECURE_NACE)
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path.rstrip("/"), "", ""))


def resolve_destination(base_url: str | None) -> Destination:
    """The API origin: `base_url`, else `NACE_BASE_URL`, else legacy `DREX_BASE_URL`, else Nace.

    Blank values count as not set.

    Raises:
        ValueError: The chosen URL is not an absolute http(s) URL without credentials, query or
            fragment, or names a Nace production host over plain http or another port.

    """
    if base_url is not None and base_url.strip():
        return Destination(_normalize(base_url.strip()), "argument")
    for name in ("NACE_BASE_URL", "DREX_BASE_URL"):
        if value := _env(name):
            return Destination(_normalize(value), "environment")
    return Destination(_normalize(DEFAULT_BASE_URL), "default")


def resolve_api_key(api_key: str | None, destination: Destination) -> str:
    """The key for `destination`: an explicit one, else the only ambient key allowed there.

    - An explicit `api_key` always wins; a blank one is refused, never replaced from the
      environment.
    - OpenRouter only ever gets `OPENROUTER_API_KEY`.
    - Nace production gets `NACE_API_KEY`, else legacy `DREX_API_KEY`.
    - Another origin gets that same pair only when the operator chose it through the
      environment (`NACE_BASE_URL` / `DREX_BASE_URL`); a `base_url=` argument needs `api_key=`.

    Raises:
        ValueError: The key is blank or missing, or the ambient key may not go to the origin.

    """
    if api_key is not None:
        if not api_key.strip():
            raise ValueError(_BLANK_KEY)
        return api_key.strip()
    if is_openrouter_host(destination.url):
        if key := _env("OPENROUTER_API_KEY"):
            return key
        raise ValueError(_OPENROUTER_KEY)
    if destination.source == "argument" and not is_nace_production(destination.url):
        raise ValueError(_REFUSED_CUSTOM)
    if key := _env("NACE_API_KEY") or _env("DREX_API_KEY"):
        return key
    raise ValueError(_NO_KEY)
