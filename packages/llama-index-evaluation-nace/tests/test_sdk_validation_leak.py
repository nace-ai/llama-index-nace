"""The SDK's own response-validation error never reaches users with response-controlled text."""

from __future__ import annotations

import traceback
from collections.abc import Callable

import nace_sdk
import pytest
from fakes import FakeDrex
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._drex.errors import NaceAPIResponseValidationError

MakeClient = Callable[..., DrexClient]
SECRET = "PATIENT-SSN-123"
QUESTIONS = {"q": nace_sdk.Noul(instructions="ok?")}
BODY = (
    b'{"model":"m","usage":{"input_tokens":1,"output_tokens":0},'
    b'"answers":{"PATIENT-SSN-123":{"type":"noul","noul":true}}}'
)


def _assert_clean(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    for text in (str(error), repr(error), rendered):
        assert SECRET not in text
        assert "noul" not in text.replace("Noul", "")
    assert error.__cause__ is None


def test_sdk_validation_error_text_is_replaced(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [(200, {}, BODY)]
    with pytest.raises(NaceAPIResponseValidationError) as info:
        make_client().ask("hi", QUESTIONS)
    _assert_clean(info.value)


async def test_sdk_validation_error_text_is_replaced_async(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    drex.replies = [(200, {}, BODY)]
    with pytest.raises(NaceAPIResponseValidationError) as info:
        await make_client().aask("hi", QUESTIONS)
    _assert_clean(info.value)
