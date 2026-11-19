"""Live: `NaceReader` parses a small made-up file through the real NDI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from llama_index.readers.nace import NaceReader

pytestmark = pytest.mark.live

KEY = os.environ.get("NACE_API_KEY_LIVE")
BASE_URL = os.environ.get("NACE_BASE_URL_LIVE") or None  # unset or empty means prod
PHRASE = "the blue whale is the largest known animal"


def test_live_reader_parses_a_small_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text(f"# Live probe\n\nMade-up content: {PHRASE}.\n", encoding="utf-8")
    reader = NaceReader(api_key=KEY, base_url=BASE_URL)
    try:
        documents = reader.load_data(path)
    finally:
        reader.close()
    assert documents
    assert any(PHRASE in document.text for document in documents)
    assert documents[0].metadata["file_name"] == "notes.md"
