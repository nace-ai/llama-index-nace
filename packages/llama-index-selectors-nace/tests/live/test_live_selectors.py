"""Live routing decisions through the selectors. Invented tools only."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import live_options
from llama_index.core.tools import ToolMetadata
from llama_index.selectors.nace import NaceMultiSelector, NaceSelectorResult, NaceSingleSelector

pytestmark = pytest.mark.live

TOOLS = [
    ToolMetadata(
        name="refund_policy",
        description="Answers questions about refunds, returns and billing disputes.",
    ),
    ToolMetadata(
        name="api_docs",
        description="Answers developer questions about REST endpoints, SDKs and rate limits.",
    ),
]


def test_single_selector_routes_to_right_tool(record: Any) -> None:
    selector = NaceSingleSelector(**live_options())
    billing = selector.select(TOOLS, "I was charged twice, how do I get my money back?")
    assert isinstance(billing, NaceSelectorResult)
    assert billing.ind == 0
    assert billing.probabilities["refund_policy"] > 0.5
    record(billing.model, billing.request_id)
    dev = selector.select(TOOLS, "Which header do I send to authenticate against the REST API?")
    assert isinstance(dev, NaceSelectorResult)
    assert dev.ind == 1
    record(dev.model, dev.request_id)


async def test_single_selector_async(record: Any) -> None:
    selector = NaceSingleSelector(**live_options())
    result = await selector.aselect(TOOLS, "What is the request limit per minute for the SDK?")
    assert isinstance(result, NaceSelectorResult)
    assert result.ind == 1
    record(result.model, result.request_id)


def test_multi_selector_picks_only_relevant(record: Any) -> None:
    selector = NaceMultiSelector(**live_options())
    refund = selector.select(TOOLS, "I was charged twice, how do I get my money back?")
    assert isinstance(refund, NaceSelectorResult)
    assert [s.index for s in refund.selections] == [0]
    assert not refund.fell_back
    record(refund.model, refund.request_id)
    docs = selector.select(
        TOOLS, "Which REST endpoint lists my invoices, and what is its rate limit?"
    )
    assert isinstance(docs, NaceSelectorResult)
    assert [s.index for s in docs.selections] == [1]
    assert not docs.fell_back
    record(docs.model, docs.request_id)


def test_multi_selector_picks_both_when_both_needed(record: Any) -> None:
    selector = NaceMultiSelector(**live_options())
    query = (
        "I was charged twice, how do I get my money back? "
        "Also, which REST endpoint lists my invoices?"
    )
    result = selector.select(TOOLS, query)
    assert isinstance(result, NaceSelectorResult)
    assert {s.index for s in result.selections} == {0, 1}
    record(result.model, result.request_id)
