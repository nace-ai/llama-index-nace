"""The reader's credential matrix: no key or internal header ever reaches a foreign host."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fakes import FakeNdi
from llama_index.readers.nace import NaceReader
from llama_index.readers.nace._drex.errors import NaceRedirectError

URL = "https://example.test/files/report.pdf"
AMBIENT = "nace_sk_test_ambient"
PROVIDER = "sk-provider-test"


@pytest.fixture(autouse=True)
def _ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_API_KEY", AMBIENT)


def test_custom_base_url_without_key_is_refused_at_construction(ndi: FakeNdi) -> None:
    with pytest.raises(ValueError, match="api_key"):
        NaceReader(
            base_url="https://example.test", http_client=httpx.Client(transport=ndi.transport())
        )
    assert ndi.requests == []


def test_openrouter_is_refused_at_construction(ndi: FakeNdi) -> None:
    with pytest.raises(ValueError, match="OpenRouter"):
        NaceReader(
            base_url="https://openrouter.ai/api/v1",
            api_key=PROVIDER,
            http_client=httpx.Client(transport=ndi.transport()),
        )
    assert ndi.requests == []


def test_redirect_of_the_parse_request_is_refused(
    make_reader: Callable[..., NaceReader], ndi: FakeNdi
) -> None:
    ndi.replies["parse"] = [(307, {"location": "https://elsewhere.test/v1/documents/parse"}, {})]
    with pytest.raises(NaceRedirectError):
        make_reader(
            http_client=httpx.Client(transport=ndi.transport(), follow_redirects=True)
        ).load_data(URL)
    assert {request.url.host for request in ndi.requests} == {"ndi.test"}


def test_bypass_header_not_sent_to_custom_base_url(
    make_reader: Callable[..., NaceReader], ndi: FakeNdi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "made-up-bypass")
    documents = make_reader(base_url="https://example.test", api_key=PROVIDER).load_data(URL)
    assert documents
    assert all("x-vercel-protection-bypass" not in request.headers for request in ndi.requests)
