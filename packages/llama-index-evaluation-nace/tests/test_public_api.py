"""The public namespace exports every error users catch and refuses foreign clients."""

from __future__ import annotations

import inspect

import llama_index.evaluation.nace as package
import llama_index.selectors.nace._drex.client as other_client
import pytest
from llama_index.evaluation.nace import NaceCorrectnessEvaluator
from llama_index.evaluation.nace._drex import errors


def _error_names() -> list[str]:
    return [
        name
        for name in errors.__all__
        if inspect.isclass(getattr(errors, name))
        and issubclass(getattr(errors, name), Exception)
        and name != "NaceLowConfidenceError"
    ]


@pytest.mark.parametrize("name", _error_names())
def test_public_errors_exported(name: str) -> None:
    assert name in package.__all__
    assert getattr(package, name) is getattr(errors, name)


def test_other_package_client_refused() -> None:
    foreign = other_client.DrexClient("other", api_key="nace_sk_test_ambient")
    with pytest.raises(TypeError, match="DrexClient"):
        NaceCorrectnessEvaluator(client=foreign)  # type: ignore[arg-type]
