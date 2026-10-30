"""Which key goes to which host: ambient credentials never reach a custom or foreign origin."""

from __future__ import annotations

from typing import Any

import httpx
import nace_sdk
import pytest
from fakes import FakeDrex
from llama_index.selectors.nace._drex.client import DrexClient
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
