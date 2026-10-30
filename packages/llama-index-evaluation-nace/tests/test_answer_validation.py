"""Out-of-range and non-finite values in Drex answers are rejected without echoing them."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import nace_sdk
import pytest
from fakes import FakeDrex
from llama_index.evaluation.nace._drex.answers import DrexResponse, RawAnswer
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._drex.errors import NaceAPIResponseValidationError

MakeClient = Callable[..., DrexClient]
QUESTIONS = {"q": nace_sdk.Noul(instructions="ok?")}


def body(answers: str) -> bytes:
    return b'{"model":"m","usage":{"input_tokens":1},"answers":' + answers.encode() + b"}"


def reply(answers: str) -> tuple[int, dict[str, str], bytes]:
    return 200, {}, body(answers)


def single(payload: str) -> str:
    return '{"q":' + payload + "}"


async def aask(client: DrexClient) -> DrexResponse:
    return await client.aask("hi", QUESTIONS)


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity", "1.7", "-0.3", "true", '"0.9"'])
async def test_invalid_noul_is_a_validation_error(
    raw: str, drex: FakeDrex, make_client: MakeClient
) -> None:
    payload = single('{"type":"noul","noul":' + raw + "}")
    drex.replies = [reply(payload), reply(payload)]
    client = make_client()
    with pytest.raises(NaceAPIResponseValidationError, match=r"answers\.q") as sync_info:
        client.ask("hi", QUESTIONS)
    with pytest.raises(NaceAPIResponseValidationError, match=r"answers\.q") as async_info:
        await aask(client)
    for info in (sync_info, async_info):
        text = str(info.value)
        for banned in ("1.7", "NaN", "Infinity", "0.9"):
            assert banned not in text
        assert info.value.__cause__ is None


@pytest.mark.parametrize(("raw", "expected"), [("0", 0.0), ("1", 1.0)])
def test_integer_bounds_are_valid_probabilities(
    raw: str, expected: float, drex: FakeDrex, make_client: MakeClient
) -> None:
    drex.replies = [reply(single('{"type":"noul","noul":' + raw + "}"))]
    answer = make_client().ask("hi", QUESTIONS).answers["q"]
    assert answer.noul == expected  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("confidence", "probabilities"),
    [("0.9", '{"a":1.5}'), ("0.9", '{"a":NaN}'), ("3", '{"a":0.9}'), ("-1", '{"a":0.9}')],
)
def test_invalid_choice_fields(
    confidence: str, probabilities: str, drex: FakeDrex, make_client: MakeClient
) -> None:
    payload = (
        '{"type":"choice","choice":"a","confidence":'
        + confidence
        + ',"probabilities":'
        + probabilities
        + "}"
    )
    drex.replies = [reply(single(payload))]
    with pytest.raises(NaceAPIResponseValidationError, match=r"answers\.q"):
        make_client().ask("hi", QUESTIONS)


@pytest.mark.parametrize(
    ("score", "probabilities"),
    [
        ("NaN", '{"0":1}'),
        ("-1", '{"0":1}'),
        ("Infinity", '{"0":1}'),
        ("1", '{"0":-0.5}'),
        ("1", '{"0":2}'),
    ],
)
def test_invalid_score_fields(
    score: str, probabilities: str, drex: FakeDrex, make_client: MakeClient
) -> None:
    payload = (
        '{"type":"score","score":'
        + score
        + ',"confidence":0.9,"legend":["a","b","c"],"probabilities":'
        + probabilities
        + "}"
    )
    drex.replies = [reply(single(payload))]
    with pytest.raises(NaceAPIResponseValidationError, match=r"answers\.q"):
        make_client().ask("hi", QUESTIONS)


def test_score_level_range_is_left_to_the_evaluator(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    payload = (
        '{"type":"score","score":1,"confidence":0.9,"legend":["a","b","c"],'
        '"probabilities":{"9":1.0}}'
    )
    drex.replies = [reply(single(payload))]
    answer = make_client().ask("hi", QUESTIONS).answers["q"]
    assert answer.probabilities == {9: 1.0}  # type: ignore[union-attr]


def test_unknown_answer_types_still_kept(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [reply(single('{"type":"future","x":1}'))]
    answer = make_client().ask("hi", QUESTIONS).answers["q"]
    assert isinstance(answer, RawAnswer)
    data: dict[str, Any] = answer.data
    assert data == {"x": 1}
