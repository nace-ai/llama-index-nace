from __future__ import annotations

import asyncio
import os
import time
import types
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fakes import FakeDrex
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._version import __version__
from nace_sdk._core import retry

CLIENT_NAME = f"llama-index-evaluation-nace/{__version__}"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip `live` tests unless NACE_API_KEY_LIVE is set, so the default run stays offline."""
    if os.environ.get("NACE_API_KEY_LIVE"):
        return
    skip = pytest.mark.skip(reason="set NACE_API_KEY_LIVE to run live tests")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return  # live tests read their own NACE_*_LIVE variables
    for name in (
        "NACE_API_KEY",
        "NACE_BASE_URL",
        "NACE_DEFAULT_DECISION_MODEL",
        "DREX_API_KEY",
        "DREX_BASE_URL",
        "DREX_DEFAULT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry waits instead of sleeping."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    # Replace the module references inside the SDK's retry code only, not the
    # global `time.sleep` / `asyncio.sleep`.
    fake_time = types.SimpleNamespace(
        **{k: getattr(time, k) for k in dir(time) if not k.startswith("_")}
    )
    fake_time.sleep = recorded.append
    fake_asyncio = types.SimpleNamespace(
        **{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("_")}
    )
    fake_asyncio.sleep = fake_sleep
    monkeypatch.setattr(retry, "time", fake_time)
    monkeypatch.setattr(retry, "asyncio", fake_asyncio)
    return recorded


@pytest.fixture
def drex() -> FakeDrex:
    return FakeDrex()


@pytest.fixture
def make_client(drex: FakeDrex, sleeps: list[float]) -> Callable[..., DrexClient]:
    def make(**kwargs: Any) -> DrexClient:
        options: dict[str, Any] = {
            "api_key": "nace_sk_test",
            "base_url": "https://drex.test",
            "http_client": httpx.Client(transport=drex.transport()),
            "http_async_client": httpx.AsyncClient(transport=drex.async_transport()),
        }
        options.update(kwargs)
        return DrexClient(CLIENT_NAME, **options)

    return make
