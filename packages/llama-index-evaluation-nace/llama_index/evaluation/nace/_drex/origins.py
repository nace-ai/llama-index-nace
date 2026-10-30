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
import nace_sdk

from .errors import REDIRECT_MESSAGE

DEFAULT_BASE_URL = "https://console.nace.ai"
NACE_PRODUCTION_HOSTS = frozenset({"console.nace.ai", "drex.nace.ai"})
OPENROUTER_HOST = "openrouter.ai"
ORIGIN_EXTENSION = "llama_index_nace.origin"
BYPASS_HEADERS = ("x-vercel-protection-bypass", "x-vercel-set-bypass-cookie")
# `X-Nace-Client` values of requests this integration owns; the SDK-wide `X-Nace-SDK` header is
# on every SDK request, including other users' clients that share an httpx client.
OWNED_CLIENT_PREFIXES = ("llama-index-selectors-nace/", "llama-index-evaluation-nace/")
BYPASS_COOKIES = frozenset({"_vercel_jwt"})
_DEFAULT_PORTS = {"http": 80, "https": 443}

_REFUSED_CUSTOM = (
    "base_url points to a provider other than Nace, so pass that provider's key as api_key=. "
    "NACE_API_KEY is only sent to Nace, or to NACE_BASE_URL when you set both."
)
_OPENROUTER_KEY = (
    "OpenRouter mode needs api_key=, or OPENROUTER_API_KEY for https://openrouter.ai; "
    "NACE_API_KEY is never sent in OpenRouter mode."
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
    """Lower-case a DNS name or IPv6 address and drop a trailing dot, keeping an IPv6 zone id.

    The interface name after `%` is case-sensitive.
    """
    address, percent, zone = (hostname or "").partition("%")
    return address.lower().rstrip(".") + percent + zone


def _netloc_host(netloc: str) -> str:
    """The host part of `netloc`, case untouched (`urlsplit().hostname` lower-cases it all)."""
    host = netloc.rpartition("@")[2]
    if host.startswith("["):
        return host[1:].partition("]")[0]
    return host.partition(":")[0]


def _split(url: str) -> tuple[str, str, int | None] | None:
    """Scheme, normalized host and explicit port, or None when `url` can't be parsed."""
    try:
        parts = urlsplit(url)
        return parts.scheme.lower(), _host(_netloc_host(parts.netloc)), parts.port
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


def is_openrouter_production(url: str | None) -> bool:
    """Whether `url` is https on port 443 of `openrouter.ai`, where its ambient key may go."""
    parsed = _split(str(url)) if url else None
    if parsed is None:
        return False
    scheme, host, port = parsed
    return scheme == "https" and host == OPENROUTER_HOST and port in {None, 443}


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
    host = _host(_netloc_host(parts.netloc))
    if (
        port == 0
        or scheme not in {"http", "https"}
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


def resolve_api_key(
    api_key: str | None, destination: Destination, *, openrouter: bool = False
) -> str:
    """The key for `destination`: an explicit one, else the only ambient key allowed there.

    - An explicit `api_key` always wins; a blank one is refused, never replaced from the
      environment.
    - In OpenRouter mode (`openrouter` true, or the host is `openrouter.ai`) the ambient key is
      never `NACE_API_KEY` / `DREX_API_KEY`: it is `OPENROUTER_API_KEY`, and only for
      `https://openrouter.ai` on port 443. Any other OpenRouter-mode destination needs an
      explicit `api_key`.
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
    if openrouter or is_openrouter_host(destination.url):
        if is_openrouter_production(destination.url) and (key := _env("OPENROUTER_API_KEY")):
            return key
        raise ValueError(_OPENROUTER_KEY)
    if destination.source == "argument" and not is_nace_production(destination.url):
        raise ValueError(_REFUSED_CUSTOM)
    if key := _env("NACE_API_KEY") or _env("DREX_API_KEY"):
        return key
    raise ValueError(_NO_KEY)


def _strip_bypass_cookies(request: httpx.Request) -> None:
    """Drop the bypass cookies from this request's `Cookie` fields, nothing else.

    Each `Cookie` field is filtered on its own (`headers.get` would join repeated fields with a
    comma), other cookies keep their field, and a field left empty is removed. Only the outgoing
    header changes; the caller's cookie jar is never touched.
    """
    fields = request.headers.get_list("cookie")
    if not fields:
        return
    rebuilt: list[str] = []
    for field in fields:
        pairs = [pair.strip() for pair in field.split(";") if pair.strip()]
        kept = [pair for pair in pairs if pair.partition("=")[0].strip() not in BYPASS_COOKIES]
        if kept:
            rebuilt.append("; ".join(kept))
    if rebuilt == fields:
        return
    others = [(name, value) for name, value in request.headers.multi_items() if name != "cookie"]
    request.headers = httpx.Headers([*others, *(("cookie", field) for field in rebuilt)])


def guard_request(request: httpx.Request) -> None:
    """Request hook for every httpx client a `DrexClient` uses; sync and async share it.

    Acts only on requests this integration sends (`X-Nace-Client` starts with one of
    `OWNED_CLIENT_PREFIXES`), so other traffic on a shared client, including a standalone
    `nace_sdk` client, is left alone, and is idempotent, because several `DrexClient`s (and
    both packages' copies of this hook) may sit on one client.

    - Binds the request to the origin of its first hop: a later hop (a redirect) to another
      scheme, host or port, or to a URL with userinfo, is refused before it is sent, with the
      error the SDK itself raises, so credentials and the request body never leave.
    - Removes the internal deployment headers, and the Vercel bypass cookie from the `Cookie`
      header (other cookies stay), from every request that isn't going to Nace production,
      whatever put them there.
    - Sends the client name as `User-Agent` too; the SDK sets it as `X-Nace-Client`.

    Raises:
        nace_sdk.NaceError: The request left its first origin.

    """
    name = request.headers.get("x-nace-client", "")
    if not name.startswith(OWNED_CLIENT_PREFIXES):
        return
    url = request.url
    scheme = url.scheme.lower()
    # Only a missing port takes the scheme's default; port 0 is a different origin.
    port = url.port if url.port is not None else _DEFAULT_PORTS.get(scheme)
    origin = (scheme, _host(url.host), port)
    first = request.extensions.setdefault(ORIGIN_EXTENSION, origin)
    if origin != first or url.userinfo:
        raise nace_sdk.NaceError(REDIRECT_MESSAGE)
    if not is_nace_production(url):
        for header in BYPASS_HEADERS:
            request.headers.pop(header, None)
        _strip_bypass_cookies(request)
    request.headers["User-Agent"] = name


async def aguard_request(request: httpx.Request) -> None:
    """Async client twin of `guard_request`."""
    guard_request(request)
