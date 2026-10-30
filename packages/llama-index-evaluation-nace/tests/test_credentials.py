"""Which key goes to which host: ambient credentials never reach a custom or foreign origin."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

import httpx
import nace_sdk
import pytest
from fakes import FakeDrex
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._drex.errors import NaceRedirectError
from llama_index.evaluation.nace._drex.origins import aguard_request, guard_request
from llama_index.evaluation.nace._version import __version__

CLIENT_NAME = f"llama-index-evaluation-nace/{__version__}"
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


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_two_packages_share_one_client(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    from llama_index.selectors.nace._drex import client as selectors_client
    from llama_index.selectors.nace._drex import errors as selectors_errors

    sync_http, async_http = following_clients(drex)
    first = selectors_client.DrexClient(
        "llama-index-selectors-nace/test",
        http_client=sync_http,
        http_async_client=async_http,
        max_retries=0,
    )
    second = build(drex, http_client=sync_http, http_async_client=async_http)
    await send(first, asynchronous=asynchronous)  # type: ignore[arg-type]
    await send(second, asynchronous=asynchronous)
    assert len(drex.requests) == 2
    drex.requests.clear()

    redirecting(drex, "https://attacker.example/v1/systemone")
    with pytest.raises(NaceRedirectError) as ours:
        await send(second, asynchronous=asynchronous)
    assert type(ours.value) is NaceRedirectError
    redirecting(drex, "https://attacker.example/v1/systemone")
    with pytest.raises(selectors_errors.NaceRedirectError) as theirs:
        await send(first, asynchronous=asynchronous)  # type: ignore[arg-type]
    assert type(theirs.value) is selectors_errors.NaceRedirectError
    assert len(sync_http.event_hooks["request"]) == 2
    assert len(async_http.event_hooks["request"]) == 2


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


@pytest.mark.skipif(
    tuple(int(part) for part in version("nace-sdk").split(".")[:3]) >= (0, 6, 2),
    reason="nace-sdk 0.6.2+ refuses cross-origin redirects itself, for every client",
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("client_info", [None, "other-app/1.0", "llama-index-selectors-nacex/1"])
async def test_standalone_sdk_client_on_a_shared_client_is_untouched(
    ambient: None, drex: FakeDrex, asynchronous: bool, client_info: str | None
) -> None:
    sync_http, async_http = following_clients(drex)
    ours = build(drex, http_client=sync_http, http_async_client=async_http)
    options: dict[str, Any] = {
        "api_key": PROVIDER,
        "base_url": "https://console.nace.ai",
        "client_info": client_info,
    }
    redirecting(drex, "https://storage.example/v1/systemone")
    if asynchronous:
        other = nace_sdk.AsyncNaceClient(**options, http_client=async_http)
        await other.system_one(*CALL)
    else:
        standalone = nace_sdk.NaceClient(**options, http_client=sync_http)
        standalone.system_one(*CALL)
    assert [r.url.host for r in drex.requests] == ["console.nace.ai", "storage.example"]
    # The integration's own requests on the same client are still bound to their origin.
    redirecting(drex, "https://storage.example/v1/systemone")
    with pytest.raises(NaceRedirectError):
        await send(ours, asynchronous=asynchronous)


def _jar_client(drex: FakeDrex, *, asynchronous: bool, **client_kwargs: Any) -> Any:
    """A caller-supplied client holding a domain-less bypass cookie and an ordinary one."""
    cls = httpx.AsyncClient if asynchronous else httpx.Client
    transport = drex.async_transport() if asynchronous else drex.transport()
    http = cls(transport=transport, **client_kwargs)
    http.cookies.set("_vercel_jwt", "jwt-test")
    http.cookies.set("session", "keep-me")
    return http


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("via", ["jar", "client_header", "default_headers"])
@pytest.mark.parametrize(
    ("base_url", "kept"),
    [("https://provider.example", False), ("https://console.nace.ai", True)],
)
async def test_bypass_cookie_only_reaches_nace_production(
    ambient: None, drex: FakeDrex, asynchronous: bool, via: str, base_url: str, kept: bool
) -> None:
    cookie = "session=keep-me; _vercel_jwt=jwt-test; theme=dark"
    options: dict[str, Any] = {"base_url": base_url, "api_key": PROVIDER}
    client_kwargs: dict[str, Any] = {}
    if via == "client_header":
        client_kwargs["headers"] = {"Cookie": cookie}
    if via == "default_headers":
        options["default_headers"] = {"Cookie": cookie}
    http = _jar_client(drex, asynchronous=asynchronous, **client_kwargs)
    options["http_async_client" if asynchronous else "http_client"] = http
    await send(build(drex, **options), asynchronous=asynchronous)
    sent = drex.requests[0].headers.get("cookie", "")
    assert ("_vercel_jwt=jwt-test" in sent) is kept
    assert "session=keep-me" in sent
    if via != "jar":
        assert "theme=dark" in sent
    # The caller's jar is not modified.
    assert http.cookies.get("_vercel_jwt") == "jwt-test"
    assert http.cookies.get("session") == "keep-me"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_cookie_header_dropped_when_only_the_bypass_cookie_was_in_it(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    http = _jar_client(drex, asynchronous=asynchronous, headers={"Cookie": "_vercel_jwt=jwt-test"})
    options: dict[str, Any] = {"base_url": "https://provider.example", "api_key": PROVIDER}
    options["http_async_client" if asynchronous else "http_client"] = http
    await send(build(drex, **options), asynchronous=asynchronous)
    assert "cookie" not in drex.requests[0].headers


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("base_url", "target"),
    [
        ("https://provider.example", "https://provider.example:0/moved"),
        ("http://onprem.example", "http://onprem.example:0/moved"),
    ],
)
async def test_redirect_to_port_zero_is_a_different_origin(
    ambient: None, drex: FakeDrex, base_url: str, target: str, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = build(
        drex,
        base_url=base_url,
        api_key=PROVIDER,
        http_client=sync_http,
        http_async_client=async_http,
    )
    redirecting(drex, target)
    with pytest.raises(NaceRedirectError, match="different origin"):
        await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 1


@pytest.mark.parametrize("via", ["argument", "environment"])
@pytest.mark.parametrize("url", ["https://provider.example:0", "http://127.0.0.1:0/service"])
def test_port_zero_base_url_refused(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, url: str, via: str
) -> None:
    kwargs: dict[str, Any] = {"api_key": PROVIDER}
    if via == "argument":
        kwargs["base_url"] = url
    else:
        monkeypatch.setenv("NACE_BASE_URL", url)
    with pytest.raises(ValueError, match="absolute http"):
        build(drex, **kwargs)
    assert drex.requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("zone", ["%25MyVPN", "%MyVPN"])
async def test_ipv6_zone_id_case_is_preserved(
    drex: FakeDrex, zone: str, asynchronous: bool
) -> None:
    base = f"http://[FE80::1{zone}]:8000/service"
    client = build(drex, base_url=base, api_key=PROVIDER)
    assert client.base_url == f"http://[fe80::1{zone}]:8000/service"
    await send(client, asynchronous=asynchronous)
    assert "MyVPN" in str(drex.requests[0].url)
    assert "myvpn" not in str(drex.requests[0].url)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_ipv6_zone_ids_that_differ_in_case_are_different_origins(
    drex: FakeDrex, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = build(
        drex,
        base_url="http://[fe80::1%25MyVPN]:8000",
        api_key=PROVIDER,
        http_client=sync_http,
        http_async_client=async_http,
    )
    redirecting(drex, "http://[fe80::1%25myvpn]:8000/v1/systemone")
    with pytest.raises(NaceRedirectError, match="different origin"):
        await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (["session=ok", "_vercel_jwt=jwt-test"], ["session=ok"]),
        (["_vercel_jwt=jwt-test", "session=ok"], ["session=ok"]),
        (["a=1; _vercel_jwt=jwt-test", "b=2; c=3"], ["a=1", "b=2; c=3"]),
        (["_vercel_jwt=jwt-test", "_vercel_jwt=jwt-two", "session=ok"], ["session=ok"]),
        (["a=1; b=2", "_vercel_jwt=jwt-test"], ["a=1; b=2"]),
    ],
)
async def test_repeated_cookie_fields_are_filtered_one_by_one(
    ambient: None, drex: FakeDrex, asynchronous: bool, fields: list[str], expected: list[str]
) -> None:
    headers = httpx.Headers([("Cookie", field) for field in fields])
    http = _jar_client(drex, asynchronous=asynchronous, headers=headers)
    options: dict[str, Any] = {"base_url": "https://provider.example", "api_key": PROVIDER}
    options["http_async_client" if asynchronous else "http_client"] = http
    await send(build(drex, **options), asynchronous=asynchronous)
    sent = drex.requests[0].headers.get_list("cookie")
    assert "_vercel_jwt" not in " ".join(sent)
    assert sent == expected


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_repeated_cookie_fields_kept_for_nace_production(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    headers = httpx.Headers([("Cookie", "_vercel_jwt=jwt-test"), ("Cookie", "session=ok")])
    http = _jar_client(drex, asynchronous=asynchronous, headers=headers)
    options: dict[str, Any] = {"base_url": "https://console.nace.ai", "api_key": PROVIDER}
    options["http_async_client" if asynchronous else "http_client"] = http
    await send(build(drex, **options), asynchronous=asynchronous)
    assert drex.requests[0].headers.get_list("cookie") == [
        "_vercel_jwt=jwt-test",
        "session=ok",
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("name", ["my-app/1.0", "llama-index-selectors-nacex/1", ""])
async def test_custom_client_name_is_still_guarded(
    ambient: None, drex: FakeDrex, asynchronous: bool, name: str
) -> None:
    sync_http, async_http = following_clients(drex, headers={BYPASS: "client-default"})
    client = DrexClient(
        name,
        base_url="https://provider.example",
        api_key=PROVIDER,
        default_headers={BYPASS_COOKIE: "true", "Cookie": "_vercel_jwt=jwt-test; a=1"},
        http_client=sync_http,
        http_async_client=async_http,
        max_retries=0,
    )
    redirecting(drex, "https://attacker.example/v1/systemone")
    with pytest.raises(NaceRedirectError, match="different origin"):
        await send(client, asynchronous=asynchronous)
    assert [r.url.host for r in drex.requests] == ["provider.example"]
    sent = drex.requests[0].headers
    assert BYPASS not in sent
    assert BYPASS_COOKIE not in sent
    assert sent["cookie"] == "a=1"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_integration_marker_cannot_be_overridden_by_default_headers(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = DrexClient(
        "my-app/1.0",
        base_url="https://provider.example",
        api_key=PROVIDER,
        default_headers={"x-nace-integration": "other"},
        http_client=sync_http,
        http_async_client=async_http,
        max_retries=0,
    )
    redirecting(drex, "https://attacker.example/v1/systemone")
    with pytest.raises(NaceRedirectError, match="different origin"):
        await send(client, asynchronous=asynchronous)
    assert drex.requests[0].headers.get_list("x-nace-integration") == ["llama-index-nace"]


@pytest.mark.skipif(
    tuple(int(part) for part in version("nace-sdk").split(".")[:3]) >= (0, 6, 2),
    reason="nace-sdk 0.6.2+ compares origins itself and does not equate other spellings",
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("base_url", "target"),
    [
        ("http://[::1]:8000", "http://[0:0:0:0:0:0:0:1]:8000/moved/v1/systemone"),
        ("http://[0:0:0:0:0:0:0:1]:8000", "http://[::1]:8000/moved/v1/systemone"),
        ("http://[2001:DB8::A]:8000", "http://[2001:0db8:0:0:0:0:0:000a]:8000/moved"),
        ("http://[::ffff:127.0.0.1]:8000", "http://[0:0:0:0:0:ffff:7f00:1]:8000/moved"),
    ],
)
async def test_same_origin_redirect_between_ip_literal_spellings_is_allowed(
    drex: FakeDrex, base_url: str, target: str, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = build(
        drex,
        base_url=base_url,
        api_key=PROVIDER,
        http_client=sync_http,
        http_async_client=async_http,
    )
    redirecting(drex, target)
    await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 2
    assert drex.requests[1].url.path.startswith("/moved")


@pytest.mark.skipif(
    tuple(int(part) for part in version("nace-sdk").split(".")[:3]) >= (0, 6, 2),
    reason="nace-sdk 0.6.2+ compares origins itself and does not equate zone-id spellings",
)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_same_origin_redirect_keeps_zone_id_with_expanded_address(
    drex: FakeDrex, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = build(
        drex,
        base_url="http://[fe80::1%25MyVPN]:8000",
        api_key=PROVIDER,
        http_client=sync_http,
        http_async_client=async_http,
    )
    redirecting(drex, "http://[FE80:0:0:0:0:0:0:1%25MyVPN]:8000/moved")
    await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("base_url", "target"),
    [
        ("http://[::1]:8000", "http://[0:0:0:0:0:0:0:2]:8000/moved"),
        ("http://[::1]:8000", "http://[0:0:0:0:0:0:0:1]:8001/moved"),
        ("http://[::ffff:127.0.0.1]:8000", "http://[0:0:0:0:0:ffff:7f00:2]:8000/moved"),
    ],
)
async def test_redirect_to_another_ip_address_is_still_refused(
    drex: FakeDrex, base_url: str, target: str, asynchronous: bool
) -> None:
    sync_http, async_http = following_clients(drex)
    client = build(
        drex,
        base_url=base_url,
        api_key=PROVIDER,
        http_client=sync_http,
        http_async_client=async_http,
    )
    redirecting(drex, target)
    with pytest.raises(NaceRedirectError, match="different origin"):
        await send(client, asynchronous=asynchronous)
    assert len(drex.requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_ipv6_literal_base_url_is_built_in_compressed_form(
    drex: FakeDrex, asynchronous: bool
) -> None:
    client = build(
        drex, base_url="http://[2001:0DB8:0:0:0:0:0:000A%25MyVPN]:8000/s", api_key=PROVIDER
    )
    assert client.base_url == "http://[2001:db8::a%25MyVPN]:8000/s"
    await send(client, asynchronous=asynchronous)
    assert "MyVPN" in str(drex.requests[0].url)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_non_ascii_header_bytes_survive_cookie_filtering(
    ambient: None, drex: FakeDrex, asynchronous: bool
) -> None:
    headers = [
        (b"X-Label", b"caf\xe9"),
        (b"Cookie", b"theme=dark; _vercel_jwt=jwt-test"),
        (b"x-LAST", b"1"),
    ]
    http = _jar_client(drex, asynchronous=asynchronous, headers=headers)
    options: dict[str, Any] = {"base_url": "https://provider.example", "api_key": PROVIDER}
    options["http_async_client" if asynchronous else "http_client"] = http
    await send(build(drex, **options), asynchronous=asynchronous)
    raw = drex.requests[0].headers.raw
    assert (b"X-Label", b"caf\xe9") in raw
    assert (b"x-LAST", b"1") in raw
    assert all(b"_vercel_jwt" not in value for _, value in raw)
    assert [v for n, v in raw if n.lower() == b"cookie"] == [b"theme=dark"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_cookie_filtering_keeps_other_headers_in_order_and_case(asynchronous: bool) -> None:
    marker = (b"X-Nace-Integration", b"llama-index-nace")
    request = httpx.Request(
        "POST",
        "https://provider.example/v1/systemone",
        headers=[
            marker,
            (b"X-First", b"caf\xe9"),
            (b"cookie", b"_vercel_jwt=jwt-test"),
            (b"X-MiXeD", b"2"),
            (b"Cookie", b"a=1; _vercel_jwt=jwt-two; b=2"),
            (b"X-Last", b"3"),
        ],
    )
    if asynchronous:
        await aguard_request(request)
    else:
        guard_request(request)
    assert request.headers.raw == [
        (b"Host", b"provider.example"),
        (b"Content-Length", b"0"),
        marker,
        (b"X-First", b"caf\xe9"),
        (b"X-MiXeD", b"2"),
        (b"Cookie", b"a=1; b=2"),
        (b"X-Last", b"3"),
    ]
