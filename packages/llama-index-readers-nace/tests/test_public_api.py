"""The public namespace exports every error users catch and keeps supplied clients open."""

from __future__ import annotations

import inspect

import httpx
import llama_index.readers.nace as package
import pytest
from fakes import FakeNdi
from llama_index.readers.nace import NaceReader
from llama_index.readers.nace._drex import errors


def _error_names() -> list[str]:
    return [
        name
        for name in errors.__all__
        if inspect.isclass(getattr(errors, name)) and issubclass(getattr(errors, name), Exception)
    ]


@pytest.mark.parametrize("name", _error_names())
def test_public_errors_exported(name: str) -> None:
    assert name in package.__all__
    assert getattr(package, name) is getattr(errors, name)


def test_reader_is_exported() -> None:
    assert package.NaceReader is NaceReader
    assert "NaceReader" in package.__all__
    assert "__version__" in package.__all__
    assert package.__version__


def test_close_leaves_supplied_http_client_open(ndi: FakeNdi) -> None:
    supplied = httpx.Client(transport=ndi.transport())
    reader = NaceReader(api_key="nace_sk_test", base_url="https://ndi.test", http_client=supplied)
    reader.close()
    assert not supplied.is_closed
    supplied.close()
