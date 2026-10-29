"""Live fixtures: real Drex calls, prod unless NACE_BASE_URL_LIVE is set."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

KEY = os.environ.get("NACE_API_KEY_LIVE")
BASE_URL = os.environ.get("NACE_BASE_URL_LIVE") or None  # unset or empty means prod
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY_LIVE")

_REQUEST_IDS: list[tuple[str, str, str | None]] = []


def live_options() -> dict[str, str | None]:
    """Client options; a function, not a fixture, so a failure traceback never shows the key."""
    return {"api_key": KEY, "base_url": BASE_URL}


@pytest.fixture
def record(request: pytest.FixtureRequest):
    """Remember `(test, model, request_id)`; printed once at the end of the run."""

    def add(model: str, request_id: str | None) -> None:
        assert request_id, "Drex returned no request id"
        _REQUEST_IDS.append((request.node.name, model, request_id))

    return add


@pytest.fixture(scope="session", autouse=True)
def _print_request_ids() -> Iterator[None]:
    yield
    if _REQUEST_IDS:
        print("\nLive request ids:")
        for test, model, request_id in _REQUEST_IDS:
            print(f"  {test:<58} {model:<10} {request_id}")
