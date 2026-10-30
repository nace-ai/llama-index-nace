"""A 402 points to the credits page of the service that answered."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import nace_sdk
import pytest
from fakes import FakeDrex, error
from llama_index.selectors.nace._drex.client import DrexClient
from llama_index.selectors.nace._drex.errors import (
    BILLING_URL,
    OPENROUTER_CREDITS_URL,
    NaceInsufficientCreditError,
    convert_error,
)

MakeClient = Callable[..., DrexClient]
QUESTIONS = {"q": nace_sdk.Noul(instructions="ok?")}


def test_openrouter_402_links_openrouter_credits(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(402, "no credits")]
    client = make_client(base_url="https://openrouter.ai/api/v1", api_key="sk-provider-test")
    with pytest.raises(NaceInsufficientCreditError) as info:
        client.ask("hi", QUESTIONS)
    text = str(info.value)
    assert OPENROUTER_CREDITS_URL == "https://openrouter.ai/settings/credits"
    assert OPENROUTER_CREDITS_URL in text
    assert BILLING_URL not in text


async def test_openrouter_402_async(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(402, "no credits")]
    client = make_client(base_url="https://openrouter.ai/api/v1", api_key="sk-provider-test")
    with pytest.raises(NaceInsufficientCreditError) as info:
        await client.aask("hi", QUESTIONS)
    assert OPENROUTER_CREDITS_URL in str(info.value)
    assert BILLING_URL not in str(info.value)


def test_nace_402_links_nace_billing(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(402, "no credits")]
    with pytest.raises(NaceInsufficientCreditError) as info:
        make_client().ask("hi", QUESTIONS)
    assert BILLING_URL in str(info.value)
    assert OPENROUTER_CREDITS_URL not in str(info.value)


def test_convert_error_defaults_to_nace_billing() -> None:
    sdk = nace_sdk.NaceInsufficientCreditError(402, {"error": {"message": "x"}}, httpx.Headers({}))
    assert BILLING_URL in str(convert_error(sdk))
    assert OPENROUTER_CREDITS_URL in str(convert_error(sdk, openrouter=True))
