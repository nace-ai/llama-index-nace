"""Which key goes to which host: ambient credentials never reach a custom or foreign origin."""

from __future__ import annotations

from typing import Any

import httpx
import nace_sdk
import pytest
from fakes import FakeDrex
from llama_index.selectors.nace._drex.client import DrexClient
from llama_index.selectors.nace._drex.errors import NaceRedirectError
from llama_index.selectors.nace._version import __version__

CLIENT_NAME = f"llama-index-selectors-nace/{__version__}"
CALL = ("state", {"q": nace_sdk.Noul(instructions="ok?")})
AMBIENT = "nace_sk_test_ambient"
ONPREM = "nace_sk_test_onprem"
PROVIDER = "sk-provider-test"
OPENROUTER_KEY = "sk-or-test-env"


def build(drex: FakeDrex, **kwargs: Any) -> DrexClient:
    options: dict[str, Any] = {
        "http_client": httpx.Client(transport=drex.transport()),
        "http_async_client": httpx.AsyncClient(transport=drex.async_transport()),
        "max_retries": 0,
    }
    options.update(kwargs)
    return DrexClient(CLIENT_NAME, **options)


async def send(client: DrexClient, *, asynchronous: bool) -> None:
    if asynchronous:
        await client.aask(*CALL)
    else:
        client.ask(*CALL)


def bearer(drex: FakeDrex) -> str:
    assert len(drex.requests) == 1
    return drex.requests[0].headers["authorization"]


@pytest.fixture
def ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_API_KEY", AMBIENT)
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "test-bypass")


def test_ticket_reproduction_is_refused_before_any_request(ambient: None, drex: FakeDrex) -> None:
    with pytest.raises(ValueError, match="api_key"):
        build(drex, base_url="https://provider.example")
    assert drex.requests == []


@pytest.mark.parametrize("env_name", ["NACE_API_KEY", "DREX_API_KEY"])
@pytest.mark.parametrize(
    "base_url",
    [
        "https://provider.example",
        "http://127.0.0.1:8000",
        "https://console.nace.ai.evil.test",
        "https://evil.nace.ai",
        "https://eu.openrouter.ai/api/v1",
        "https://evilopenrouter.ai/api",
    ],
)
def test_explicit_custom_base_url_refuses_ambient_key(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, env_name: str, base_url: str
) -> None:
    monkeypatch.setenv(env_name, AMBIENT)
    with pytest.raises(ValueError, match="api_key"):
        build(drex, base_url=base_url)
    assert drex.requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_custom_base_url_with_explicit_key(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    client = build(drex, base_url="https://provider.example", api_key=PROVIDER)
    await send(client, asynchronous=asynchronous)
    assert bearer(drex) == f"Bearer {PROVIDER}"
    assert str(drex.requests[0].url) == "https://provider.example/v1/systemone"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("prefix", "url"),
    [
        ("NACE", "https://onprem.example:8443/service"),
        ("NACE", "http://127.0.0.1:8000/service"),
        ("NACE", "http://[::1]:8000/service"),
        ("DREX", "https://onprem.example/service"),
    ],
)
async def test_environment_pair_reaches_custom_server(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, prefix: str, url: str, asynchronous: bool
) -> None:
    monkeypatch.setenv(f"{prefix}_BASE_URL", url)
    monkeypatch.setenv(f"{prefix}_API_KEY", ONPREM)
    if prefix == "DREX":
        monkeypatch.setenv("NACE_API_KEY", " ")
        monkeypatch.setenv("NACE_BASE_URL", " ")
    client = build(drex)
    await send(client, asynchronous=asynchronous)
    assert str(drex.requests[0].url) == f"{url}/v1/systemone"
    assert bearer(drex) == f"Bearer {ONPREM}"


def test_legacy_base_url_is_resolved_by_the_integration(
    ambient: None, monkeypatch: pytest.MonkeyPatch, drex: FakeDrex
) -> None:
    monkeypatch.setenv("DREX_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build(drex)
    assert drex.requests == []
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    client = build(drex)
    client.ask(*CALL)
    assert drex.requests[0].url.path == "/api/v1/systemone"
    assert bearer(drex) == f"Bearer {OPENROUTER_KEY}"
    assert client.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("openrouter", [None, False, True])
@pytest.mark.parametrize("via", ["argument", "environment"])
@pytest.mark.parametrize("url", ["https://openrouter.ai/api/v1", "https://OPENROUTER.AI./api/v1"])
def test_openrouter_never_inherits_nace_key(
    ambient: None,
    monkeypatch: pytest.MonkeyPatch,
    drex: FakeDrex,
    url: str,
    via: str,
    openrouter: bool | None,
) -> None:
    kwargs: dict[str, Any] = {"openrouter": openrouter}
    if via == "argument":
        kwargs["base_url"] = url
    else:
        monkeypatch.setenv("NACE_BASE_URL", url)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build(drex, **kwargs)
    assert drex.requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_openrouter_uses_its_key(
    ambient: None, monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, asynchronous: bool
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    await send(build(drex, base_url="https://openrouter.ai/api/v1"), asynchronous=asynchronous)
    assert bearer(drex) == f"Bearer {OPENROUTER_KEY}"
    drex.requests.clear()
    client = build(drex, base_url="https://openrouter.ai/api/v1", api_key=PROVIDER)
    await send(client, asynchronous=asynchronous)
    assert bearer(drex) == f"Bearer {PROVIDER}"


@pytest.mark.parametrize(
    "base_url",
    [
        None,
        "https://console.nace.ai",
        "https://CONSOLE.NACE.AI.",
        "https://drex.nace.ai:443",
        "https://console.nace.ai/",
    ],
)
def test_nace_production_variants_use_ambient_key(
    ambient: None, drex: FakeDrex, base_url: str | None
) -> None:
    build(drex, base_url=base_url).ask(*CALL)
    assert bearer(drex) == f"Bearer {AMBIENT}"
    host = drex.requests[0].url.host
    assert host == host.lower()
    assert not host.endswith(".")


@pytest.mark.parametrize("via", ["argument", "environment"])
@pytest.mark.parametrize("url", ["http://console.nace.ai", "https://console.nace.ai:8443"])
def test_insecure_nace_production_url_refused(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, url: str, via: str
) -> None:
    kwargs: dict[str, Any] = {"api_key": PROVIDER}
    if via == "argument":
        kwargs["base_url"] = url
    else:
        monkeypatch.setenv("NACE_BASE_URL", url)
    with pytest.raises(ValueError, match="HTTPS"):
        build(drex, **kwargs)
    assert drex.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@console.nace.ai",
        "https://:secret@provider.example",
        "ftp://provider.example",
        "https://provider.example/?token=secret",
        "https://provider.example/#secret",
        "console.nace.ai",
        "https://provider.example:99999",
    ],
)
def test_invalid_base_url_refused_without_echo(drex: FakeDrex, url: str) -> None:
    with pytest.raises(ValueError, match="base_url") as info:
        build(drex, base_url=url, api_key=PROVIDER)
    assert "secret" not in str(info.value)
    assert url not in str(info.value)


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_explicit_key_refused(ambient: None, drex: FakeDrex, blank: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        build(drex, api_key=blank)
    assert drex.requests == []


def test_blank_base_url_means_not_set(
    ambient: None, monkeypatch: pytest.MonkeyPatch, drex: FakeDrex
) -> None:
    monkeypatch.setenv("NACE_BASE_URL", "https://onprem.example")
    build(drex, base_url="  ").ask(*CALL)
    assert drex.requests[0].url.host == "onprem.example"
    drex.requests.clear()
    monkeypatch.delenv("NACE_BASE_URL")
    build(drex, base_url="  ").ask(*CALL)
    assert drex.requests[0].url.host == "console.nace.ai"


def test_openrouter_flag_without_destination_still_rejected(ambient: None, drex: FakeDrex) -> None:
    with pytest.raises(ValueError, match="base_url"):
        build(drex, openrouter=True)


def test_missing_key_on_nace_production(drex: FakeDrex) -> None:
    with pytest.raises(ValueError, match="NACE_API_KEY"):
        build(drex)


def test_legacy_key_is_used_when_nace_key_is_blank(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex
) -> None:
    monkeypatch.setenv("NACE_API_KEY", " ")
    monkeypatch.setenv("DREX_API_KEY", "nace_sk_test_legacy")
    build(drex).ask(*CALL)
    assert bearer(drex) == "Bearer nace_sk_test_legacy"


# --- bypass headers and origin binding ---------------------------------------------------------

BYPASS = "x-vercel-protection-bypass"
BYPASS_COOKIE = "x-vercel-set-bypass-cookie"


def redirecting(drex: FakeDrex, target: str, *, times: int = 1) -> None:
    for _ in range(times):
        drex.replies.append((307, {"location": target}, {}))


def following_clients(
    drex: FakeDrex, **client_kwargs: Any
) -> tuple[httpx.Client, httpx.AsyncClient]:
    return (
        httpx.Client(transport=drex.transport(), follow_redirects=True, **client_kwargs),
        httpx.AsyncClient(transport=drex.async_transport(), follow_redirects=True, **client_kwargs),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "base_url",
    [
        "https://provider.example",
        "https://openrouter.ai/api/v1",
        "https://evil.nace.ai",
        "https://console.nace.ai.evil.test",
    ],
)
async def test_bypass_only_reaches_nace_production(
    ambient: None, drex: FakeDrex, base_url: str, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex, headers={BYPASS: "client-default"})
    client = build(
        drex,
        base_url=base_url,
        api_key=PROVIDER,
        default_headers={"X-Vercel-Set-Bypass-Cookie": "true", BYPASS: "explicit"},
        http_client=sync_http,
        http_async_client=async_http,
    )
    await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 1
    assert BYPASS not in drex.requests[0].headers
    assert BYPASS_COOKIE not in drex.requests[0].headers


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_bypass_kept_on_nace_production(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    await send(build(drex), asynchronous=asynchronous)
    assert drex.requests[0].headers[BYPASS] == "test-bypass"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        "https://attacker.example/v1/systemone",
        "https://drex.nace.ai/v1/systemone",
        "http://console.nace.ai/v1/systemone",
        "https://console.nace.ai:8443/v1/systemone",
        "https://user:pw@console.nace.ai/v1/systemone",
    ],
)
async def test_cross_origin_redirect_refused_before_send(
    ambient: None, drex: FakeDrex, target: str, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = build(drex, http_client=sync_http, http_async_client=async_http, max_retries=2)
    redirecting(drex, target)
    with pytest.raises(NaceRedirectError, match="different origin"):
        await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 1
    assert drex.requests[0].url.host == "console.nace.ai"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_same_origin_redirect_keeps_credentials(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    seen: list[str] = []

    def record(request: httpx.Request) -> None:
        seen.append(str(request.url))

    async def arecord(request: httpx.Request) -> None:
        record(request)

    sync_http, async_http = following_clients(drex)
    sync_http.event_hooks["request"].append(record)
    async_http.event_hooks["request"].append(arecord)
    client = build(drex, http_client=sync_http, http_async_client=async_http)
    redirecting(drex, "/moved/v1/systemone")
    await send(client, asynchronous=asynchronous)
    assert [r.url.path for r in drex.requests] == ["/v1/systemone", "/moved/v1/systemone"]
    assert len(seen) == 2
    for request in drex.requests:
        assert request.headers["authorization"] == f"Bearer {AMBIENT}"
        assert request.headers[BYPASS] == "test-bypass"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_on_prem_scheme_upgrade_refused(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, asynchronous: bool
) -> None:
    monkeypatch.setenv("NACE_BASE_URL", "http://onprem.example")
    monkeypatch.setenv("NACE_API_KEY", ONPREM)
    sync_http, async_http = following_clients(drex)
    client = build(drex, http_client=sync_http, http_async_client=async_http)
    redirecting(drex, "https://onprem.example/v1/systemone")
    with pytest.raises(NaceRedirectError):
        await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 1


def unrelated_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    """Records the caller's own traffic: `/x` redirects to another host, `/y` answers."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/x":
            return httpx.Response(307, headers={"location": "https://other.example/y"})
        return httpx.Response(200)

    return httpx.MockTransport(handler)


def test_unrelated_requests_on_a_shared_client_are_untouched(ambient: None, drex: FakeDrex) -> None:
    seen: list[httpx.Request] = []
    http = httpx.Client(transport=unrelated_transport(seen), follow_redirects=True)
    build(drex, http_client=http)
    response = http.get("https://unrelated.example/x", headers={BYPASS: "own"})
    assert response.status_code == 200
    assert [r.url.host for r in seen] == ["unrelated.example", "other.example"]
    assert seen[0].headers[BYPASS] == "own"


async def test_unrelated_async_requests_on_a_shared_client_are_untouched(
    ambient: None, drex: FakeDrex
) -> None:
    seen: list[httpx.Request] = []
    http = httpx.AsyncClient(transport=unrelated_transport(seen), follow_redirects=True)
    build(drex, http_async_client=http)
    response = await http.get("https://unrelated.example/x", headers={BYPASS: "own"})
    assert response.status_code == 200
    assert [r.url.host for r in seen] == ["unrelated.example", "other.example"]
    assert seen[0].headers[BYPASS] == "own"


@pytest.mark.parametrize("via", ["argument", "environment"])
@pytest.mark.parametrize(
    "url",
    [
        "http://openrouter.ai/api/v1",
        "https://openrouter.ai:8443/api/v1",
        "http://openrouter.ai:443/api/v1",
    ],
)
def test_openrouter_key_only_for_https_443(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, url: str, via: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    kwargs: dict[str, Any] = {}
    if via == "argument":
        kwargs["base_url"] = url
    else:
        monkeypatch.setenv("NACE_BASE_URL", url)
    with pytest.raises(ValueError, match="api_key"):
        build(drex, **kwargs)
    assert drex.requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_openrouter_key_for_explicit_https_443(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, asynchronous: bool
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    client = build(drex, base_url="https://OpenRouter.ai.:443/api/v1")
    await send(client, asynchronous=asynchronous)
    assert bearer(drex) == f"Bearer {OPENROUTER_KEY}"


@pytest.mark.parametrize("env_name", ["NACE_API_KEY", "DREX_API_KEY"])
@pytest.mark.parametrize("with_openrouter_key", [False, True])
@pytest.mark.parametrize("base_url_name", ["NACE_BASE_URL", "DREX_BASE_URL"])
def test_openrouter_mode_on_env_host_never_sends_nace_key(
    monkeypatch: pytest.MonkeyPatch,
    drex: FakeDrex,
    env_name: str,
    with_openrouter_key: bool,
    base_url_name: str,
) -> None:
    monkeypatch.setenv(base_url_name, "https://gateway.example/api/v1")
    monkeypatch.setenv(env_name, AMBIENT)
    if with_openrouter_key:
        monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY") as error:
        build(drex, openrouter=True)
    assert "api_key=" in str(error.value)
    assert drex.requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_openrouter_mode_on_env_host_with_explicit_key(
    ambient: None, monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, asynchronous: bool
) -> None:
    monkeypatch.setenv("NACE_BASE_URL", "https://gateway.example/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    client = build(drex, openrouter=True, api_key=PROVIDER)
    await send(client, asynchronous=asynchronous)
    assert bearer(drex) == f"Bearer {PROVIDER}"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_env_on_prem_without_openrouter_flag_keeps_nace_key(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, asynchronous: bool
) -> None:
    monkeypatch.setenv("NACE_BASE_URL", "https://gateway.example/api/v1")
    monkeypatch.setenv("NACE_API_KEY", ONPREM)
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)
    await send(build(drex), asynchronous=asynchronous)
    assert bearer(drex) == f"Bearer {ONPREM}"
