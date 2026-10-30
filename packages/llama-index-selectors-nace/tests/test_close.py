"""Selectors release the httpx clients they own and leave supplied ones alone."""

from __future__ import annotations

import pytest
from llama_index.selectors.nace import NaceMultiSelector, NaceSingleSelector
from llama_index.selectors.nace._drex.client import DrexClient

KEY = "nace_sk_test_ambient"


def _sync_closed(client: DrexClient) -> bool:
    return client._client._http_client.is_closed  # type: ignore[attr-defined,no-any-return]


def _async_closed(client: DrexClient) -> bool:
    return client._aclient._http_client.is_closed  # type: ignore[attr-defined,no-any-return]


@pytest.mark.parametrize("selector", [NaceSingleSelector, NaceMultiSelector])
async def test_close_releases_owned_client(selector: type) -> None:
    sync_selector = selector(api_key=KEY)
    sync_selector.close()
    assert _sync_closed(sync_selector._client)

    async_selector = selector(api_key=KEY)
    await async_selector.aclose()
    assert _sync_closed(async_selector._client)
    assert _async_closed(async_selector._client)


@pytest.mark.parametrize("selector", [NaceSingleSelector, NaceMultiSelector])
async def test_close_leaves_supplied_client_open(selector: type) -> None:
    supplied = DrexClient("x/1", api_key=KEY)
    chosen = selector(client=supplied)
    chosen.close()
    await chosen.aclose()
    assert not _sync_closed(supplied)
    assert not _async_closed(supplied)
    supplied.close()
    await supplied.aclose()
