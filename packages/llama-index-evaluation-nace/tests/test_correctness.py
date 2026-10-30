from __future__ import annotations

from collections.abc import Callable

import pytest
from fakes import FakeDrex, ok_body
from llama_index.core.base.response.schema import Response
from llama_index.evaluation.nace import NaceCorrectnessEvaluator, NaceEvaluationResult
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._drex.errors import NaceAPIResponseValidationError

MakeClient = Callable[..., DrexClient]
PROBS = {"0": 0.0, "1": 0.05, "2": 0.13, "3": 0.6, "4": 0.22}


def score(expected: float, probs: dict[str, float] | None = None) -> dict[str, dict[str, object]]:
    return {
        "evaluate": {
            "type": "score",
            "score": expected,
            "legend": [f"level {i}" for i in range(5)],
            "probabilities": probs or PROBS,
            "confidence": 0.9,
        }
    }


def test_score_maps_to_1_to_5(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = score(3.31)
    result = NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert isinstance(result, NaceEvaluationResult)
    assert result.score == pytest.approx(4.31)
    assert (result.query, result.response) == ("q", "r")
    question = drex.bodies[0]["questions"]["evaluate"]
    assert question["type"] == "score"
    assert question["criteria"][0] == "Not relevant to the query"
    assert question["criteria"][4] == "Relevant, correct and complete"
    assert len(question["criteria"]) == 5


def test_passing_uses_score_threshold(drex: FakeDrex, make_client: MakeClient) -> None:
    evaluator = NaceCorrectnessEvaluator(client=make_client())
    drex.overrides = score(3.31)
    assert evaluator.evaluate(query="q", response="r").passing is True
    drex.overrides = score(2.9)
    assert evaluator.evaluate(query="q", response="r").passing is False
    lenient = NaceCorrectnessEvaluator(client=make_client(), score_threshold=3.0)
    assert lenient.evaluate(query="q", response="r").passing is True


def test_feedback(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = score(3.31)
    result = NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert result.feedback == "score=4.31, P(≥4)=0.82"
    assert result.probability == pytest.approx(0.82)


def test_probabilities_exposed(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = score(3.31)
    result = NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert isinstance(result, NaceEvaluationResult)
    assert result.probabilities == {0: 0.0, 1: 0.05, 2: 0.13, 3: 0.6, 4: 0.22}


def test_reference_in_state_when_given(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceCorrectnessEvaluator(client=make_client()).evaluate(
        query="q", response="r", reference="ref"
    )
    assert drex.bodies[0]["state"] == {"query": "q", "response": "r", "reference": "ref"}


def test_reference_absent_when_none(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert "reference" not in drex.bodies[0]["state"]


def test_requires_query_and_response(drex: FakeDrex, make_client: MakeClient) -> None:
    evaluator = NaceCorrectnessEvaluator(client=make_client())
    with pytest.raises(ValueError, match="query"):
        evaluator.evaluate(response="r")
    with pytest.raises(ValueError, match="response"):
        evaluator.evaluate(query="q")
    assert drex.bodies == []


async def test_async_and_evaluate_response(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = score(3.31)
    evaluator = NaceCorrectnessEvaluator(client=make_client())
    result = await evaluator.aevaluate(query="q", response="r", reference="ref")
    assert result.score == pytest.approx(4.31)
    via_response = await evaluator.aevaluate_response(
        query="q", response=Response(response="r"), reference="ref"
    )
    assert via_response.score == pytest.approx(4.31)
    assert drex.bodies[1]["state"]["reference"] == "ref"


def test_instructions_overridable(drex: FakeDrex, make_client: MakeClient) -> None:
    from llama_index.core.prompts import PromptTemplate

    evaluator = NaceCorrectnessEvaluator(client=make_client())
    evaluator.update_prompts({"instructions": PromptTemplate("Custom?")})
    evaluator.evaluate(query="q", response="r")
    assert drex.bodies[0]["questions"]["evaluate"]["instructions"] == "Custom?"


def test_unexpected_answer_type_is_a_validation_error(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    from llama_index.evaluation.nace import NaceError

    drex.overrides = {"evaluate": {"type": "noul", "noul": 0.5}}
    with pytest.raises(NaceAPIResponseValidationError, match="unexpected answer type") as info:
        NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert isinstance(info.value, NaceError)
    assert not isinstance(info.value, ValueError)


def test_missing_answer_is_a_validation_error(drex: FakeDrex, make_client: MakeClient) -> None:
    from llama_index.evaluation.nace import NaceError

    drex.replies.append(lambda body: (200, {}, {**ok_body(body), "answers": {}}))
    with pytest.raises(NaceAPIResponseValidationError, match="no answer") as info:
        NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert isinstance(info.value, NaceError)
    assert not isinstance(info.value, KeyError | ValueError)


@pytest.mark.parametrize("threshold", [0.5, 5.5, float("nan")])
def test_score_threshold_must_be_on_the_scale(threshold: float) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        NaceCorrectnessEvaluator(score_threshold=threshold, api_key="nace_sk_test")


@pytest.mark.parametrize("threshold", [1, 5])
def test_score_threshold_bounds_are_accepted(threshold: float) -> None:
    NaceCorrectnessEvaluator(score_threshold=threshold, api_key="nace_sk_test")


@pytest.mark.parametrize(
    ("expected", "probs", "field"),
    [(4.5, None, "score"), (1.0, {"0": 0.5, "5": 0.5}, "probabilities")],
)
def test_correctness_score_outside_levels(
    drex: FakeDrex,
    make_client: MakeClient,
    expected: float,
    probs: dict[str, float] | None,
    field: str,
) -> None:
    drex.overrides = score(expected, probs)
    with pytest.raises(NaceAPIResponseValidationError, match=rf"answers\.evaluate\.{field}"):
        NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")


def test_correctness_score_upper_levels_accepted(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = score(4.0, {str(i): 0.2 for i in range(5)})
    result = NaceCorrectnessEvaluator(client=make_client()).evaluate(query="q", response="r")
    assert result.score == pytest.approx(5.0)
