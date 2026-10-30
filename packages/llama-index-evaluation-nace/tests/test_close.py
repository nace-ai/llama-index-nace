"""Evaluators release the httpx clients they own and leave supplied ones alone."""

from __future__ import annotations

import pytest
from llama_index.evaluation.nace import (
    NaceCorrectnessEvaluator,
    NaceFaithfulnessEvaluator,
    NaceRelevancyEvaluator,
)
from llama_index.evaluation.nace._drex.client import DrexClient

KEY = "nace_sk_test_ambient"
EVALUATORS = [NaceCorrectnessEvaluator, NaceFaithfulnessEvaluator, NaceRelevancyEvaluator]


def _sync_closed(client: DrexClient) -> bool:
    return client._client._http_client.is_closed  # type: ignore[attr-defined,no-any-return]


def _async_closed(client: DrexClient) -> bool:
    return client._aclient._http_client.is_closed  # type: ignore[attr-defined,no-any-return]


@pytest.mark.parametrize("evaluator", EVALUATORS)
async def test_close_releases_owned_client(evaluator: type) -> None:
    sync_evaluator = evaluator(api_key=KEY)
    sync_evaluator.close()
    assert _sync_closed(sync_evaluator._client)

    async_evaluator = evaluator(api_key=KEY)
    await async_evaluator.aclose()
    assert _sync_closed(async_evaluator._client)
    assert _async_closed(async_evaluator._client)


@pytest.mark.parametrize("evaluator", EVALUATORS)
async def test_close_leaves_supplied_client_open(evaluator: type) -> None:
    supplied = DrexClient("x/1", api_key=KEY)
    chosen = evaluator(client=supplied)
    chosen.close()
    await chosen.aclose()
    assert not _sync_closed(supplied)
    assert not _async_closed(supplied)
    supplied.close()
    await supplied.aclose()
