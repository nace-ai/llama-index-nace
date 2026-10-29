"""Shared download cache for the benchmark datasets."""

from __future__ import annotations

from pathlib import Path

import httpx

CACHE = Path(__file__).resolve().parent.parent / ".cache"


def fetch(name: str, url: str) -> Path:
    """Download `url` to the git-ignored cache once and return the local path."""
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path
