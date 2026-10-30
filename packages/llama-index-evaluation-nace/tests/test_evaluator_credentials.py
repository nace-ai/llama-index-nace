"""The ticket's evaluator matrix: a custom provider never receives the ambient Nace key."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fakes import FakeDrex
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.evaluation.nace import (
    NaceCorrectnessEvaluator,
    NaceFaithfulnessEvaluator,
    NaceRelevancyEvaluator,
)
from llama_index.evaluation.nace._drex.errors import NaceRedirectError

AMBIENT = "nace_sk_test_ambient"
PROVIDER = "sk-provider-test"
BYPASS = "x-vercel-protection-bypass"

ARGUMENTS: dict[type, dict[str, Any]] = {
    NaceFaithfulnessEvaluator: {"response": "r", "contexts": ["c"]},
    NaceRelevancyEvaluator: {"query": "q", "response": "r", "contexts": ["c"]},
    NaceCorrectnessEvaluator: {"query": "q", "response": "r"},
}

EVALUATORS = pytest.mark.parametrize("evaluator_class", list(ARGUMENTS))
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


async def evaluate(evaluator: Any, *, asynchronous: bool) -> Any:
    arguments = ARGUMENTS[type(evaluator)]
    if asynchronous:
        return await evaluator.aevaluate(**arguments)
    return evaluator.evaluate(**arguments)


@EVALUATORS
def test_custom_provider_without_key_is_refused_at_construction(
    evaluator_class: Any, drex: FakeDrex
) -> None:
    with pytest.raises(ValueError, match="api_key"):
        evaluator_class(base_url="https://provider.example", **clients(drex))
    assert drex.requests == []


@EVALUATORS
@MODES
async def test_custom_provider_with_key_sends_only_that_key(
    evaluator_class: Any, drex: FakeDrex, asynchronous: bool
) -> None:
    evaluator = evaluator_class(
        base_url="https://provider.example", api_key=PROVIDER, **clients(drex)
    )
    await evaluate(evaluator, asynchronous=asynchronous)
    request = drex.requests[0]
    assert request.headers["authorization"] == f"Bearer {PROVIDER}"
    assert BYPASS not in request.headers
    assert request.url.host == "provider.example"


@EVALUATORS
@MODES
async def test_nace_production_unchanged(
    evaluator_class: Any, drex: FakeDrex, asynchronous: bool
) -> None:
    result = await evaluate(evaluator_class(**clients(drex)), asynchronous=asynchronous)
    assert result.score is not None
    request = drex.requests[0]
    assert request.headers["authorization"] == f"Bearer {AMBIENT}"
    assert request.headers[BYPASS] == "test-bypass"
    assert request.url.host == "console.nace.ai"


@EVALUATORS
@MODES
async def test_redirect_refusal_raises(
    evaluator_class: Any, drex: FakeDrex, asynchronous: bool
) -> None:
    drex.replies.append((307, {"location": "https://attacker.example/v1/systemone"}, {}))
    evaluator = evaluator_class(**clients(drex, follow=True))
    with pytest.raises(NaceRedirectError):
        await evaluate(evaluator, asynchronous=asynchronous)
    assert len(drex.requests) == 1


async def test_aevaluate_response_with_custom_provider_sends_only_its_key(
    drex: FakeDrex,
) -> None:
    evaluator = NaceFaithfulnessEvaluator(
        base_url="https://provider.example", api_key=PROVIDER, **clients(drex)
    )
    response = Response(response="r", source_nodes=[NodeWithScore(node=TextNode(text="c"))])
    await evaluator.aevaluate_response(query="q", response=response)
    request = drex.requests[0]
    assert request.headers["authorization"] == f"Bearer {PROVIDER}"
    assert BYPASS not in request.headers
