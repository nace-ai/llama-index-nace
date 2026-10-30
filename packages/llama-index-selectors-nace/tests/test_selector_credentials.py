"""The ticket's selector matrix: a custom provider never receives the ambient Nace key."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fakes import FakeDrex
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import ToolMetadata
from llama_index.selectors.nace import NaceMultiSelector, NaceSingleSelector
from llama_index.selectors.nace._drex.errors import NaceRedirectError

TOOLS = [ToolMetadata(name="a", description="A"), ToolMetadata(name="b", description="B")]
BUNDLE = QueryBundle("q")
AMBIENT = "nace_sk_test_ambient"
PROVIDER = "sk-provider-test"
BYPASS = "x-vercel-protection-bypass"

SELECTORS = pytest.mark.parametrize("selector_class", [NaceSingleSelector, NaceMultiSelector])
MODES = pytest.mark.parametrize("asynchronous", [False, True])


@pytest.fixture(autouse=True)
def _ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_API_KEY", AMBIENT)
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "test-bypass")


def clients(drex: FakeDrex, *, follow: bool = False) -> dict[str, Any]:
    return {
        "http_client": httpx.Client(transport=drex.transport(), follow_redirects=follow),
        "http_async_client": httpx.AsyncClient(
            transport=drex.async_transport(), follow_redirects=follow
        ),
        "max_retries": 0,
    }


async def select(selector: Any, *, asynchronous: bool) -> Any:
    if asynchronous:
        return await selector.aselect(TOOLS, BUNDLE)
    return selector.select(TOOLS, BUNDLE)


@SELECTORS
def test_custom_provider_without_key_is_refused_at_construction(
    selector_class: Any, drex: FakeDrex
) -> None:
    with pytest.raises(ValueError, match="api_key"):
        selector_class(base_url="https://provider.example", **clients(drex))
    assert drex.requests == []


@SELECTORS
@MODES
async def test_custom_provider_with_key_sends_only_that_key(
    selector_class: Any, drex: FakeDrex, asynchronous: bool
) -> None:
    selector = selector_class(
        base_url="https://provider.example", api_key=PROVIDER, **clients(drex)
    )
    await select(selector, asynchronous=asynchronous)
    request = drex.requests[0]
    assert request.headers["authorization"] == f"Bearer {PROVIDER}"
    assert BYPASS not in request.headers
    assert request.url.host == "provider.example"


@SELECTORS
@MODES
async def test_nace_production_unchanged(
    selector_class: Any, drex: FakeDrex, asynchronous: bool
) -> None:
    result = await select(selector_class(**clients(drex)), asynchronous=asynchronous)
    assert result.selections
    request = drex.requests[0]
    assert request.headers["authorization"] == f"Bearer {AMBIENT}"
    assert request.headers[BYPASS] == "test-bypass"
    assert request.url.host == "console.nace.ai"


@MODES
async def test_redirect_refusal_is_not_a_fallback(drex: FakeDrex, asynchronous: bool) -> None:
    drex.replies.append((307, {"location": "https://attacker.example/v1/systemone"}, {}))
    selector = NaceSingleSelector(default_index=0, **clients(drex, follow=True))
    with pytest.raises(NaceRedirectError):
        await select(selector, asynchronous=asynchronous)
    assert len(drex.requests) == 1
