"""The public namespace exports every error users catch and refuses foreign clients."""

from __future__ import annotations

import inspect

import llama_index.evaluation.nace._drex.client as other_client
import llama_index.selectors.nace as package
import pytest
from llama_index.selectors.nace import NaceMultiSelector, NaceSingleSelector
from llama_index.selectors.nace._drex import errors


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


@pytest.mark.parametrize("selector", [NaceSingleSelector, NaceMultiSelector])
def test_other_package_client_refused(selector: type) -> None:
    foreign = other_client.DrexClient("other", api_key="nace_sk_test_ambient")
    with pytest.raises(TypeError, match="DrexClient"):
        selector(client=foreign)
