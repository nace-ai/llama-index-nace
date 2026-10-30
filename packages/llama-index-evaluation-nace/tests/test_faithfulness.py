from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from pathlib import Path

import pytest
from fakes import FakeDrex, ok_body
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.schema import ImageNode, NodeWithScore
from llama_index.evaluation.nace import (
    NaceEvaluationResult,
    NaceFaithfulnessEvaluator,
    NaceRelevancyEvaluator,
)
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._drex.errors import NaceAPIResponseValidationError

MakeClient = Callable[..., DrexClient]
RED = (Path(__file__).parent / "fixtures" / "red.png").read_bytes()
RED_B64 = base64.b64encode(RED).decode()


def noul(p: float) -> dict[str, dict[str, float | str]]:
    return {"evaluate": {"type": "noul", "noul": p}}


def test_faithfulness_result_fields(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(0.87)
    result = NaceFaithfulnessEvaluator(client=make_client()).evaluate(
        query="q", response="r", contexts=["c"]
    )
    assert isinstance(result, NaceEvaluationResult)
    assert result.score == 0.87
    assert result.passing is True
    assert result.feedback == "P(faithful)=0.87"
    assert result.probability == 0.87
    assert (result.query, result.response, result.contexts) == ("q", "r", ["c"])
    assert result.request_id == "req_test"


def test_threshold(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(0.4)
    evaluator = NaceFaithfulnessEvaluator(client=make_client())
    assert evaluator.evaluate(response="r", contexts=["c"]).passing is False
    lenient = NaceFaithfulnessEvaluator(client=make_client(), threshold=0.3)
    assert lenient.evaluate(response="r", contexts=["c"]).passing is True


def test_state_shape(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceFaithfulnessEvaluator(client=make_client()).evaluate(
        query="q", response="r", contexts=["c1", "c2"]
    )
    body = drex.bodies[0]
    assert body["state"] == {"query": "q", "context": ["c1", "c2"], "response": "r"}
    question = body["questions"]["evaluate"]
    assert question["type"] == "noul"
    assert "supported by the context" in question["instructions"]
    assert set(question["criteria"]) == {"true", "false"}


def test_raise_error(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(0.2)
    evaluator = NaceFaithfulnessEvaluator(client=make_client(), raise_error=True)
    with pytest.raises(ValueError, match="invalid"):
        evaluator.evaluate(response="r", contexts=["c"])
    assert len(drex.bodies) == 1


def test_missing_response_raises_value_error(drex: FakeDrex, make_client: MakeClient) -> None:
    with pytest.raises(ValueError, match="response"):
        NaceFaithfulnessEvaluator(client=make_client()).evaluate(contexts=["c"])
    with pytest.raises(ValueError, match="contexts"):
        NaceFaithfulnessEvaluator(client=make_client()).evaluate(response="r")
    assert drex.bodies == []


def test_faithfulness_images_only(drex: FakeDrex, make_client: MakeClient) -> None:
    nodes = [NodeWithScore(node=ImageNode(image=RED_B64), score=0.5)]
    result = NaceFaithfulnessEvaluator(client=make_client()).evaluate_response(
        query="q", response=Response(response="a red square", source_nodes=nodes)
    )
    body = drex.bodies[0]
    assert body["model"] == "drex-max-v1.1"
    assert len(body["images"]) == 1
    assert body["state"].get("context", []) in ([], ["[Image 1] "])
    assert result.passing is True


def test_images_kwarg_satisfies_contexts(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceFaithfulnessEvaluator(client=make_client()).evaluate(response="r", images=[RED])
    assert len(drex.bodies[0]["images"]) == 1


async def test_batch_eval_runner(drex: FakeDrex, make_client: MakeClient) -> None:
    runner = BatchEvalRunner(
        {
            "faithfulness": NaceFaithfulnessEvaluator(client=make_client()),
            "relevancy": NaceRelevancyEvaluator(client=make_client()),
        },
        workers=2,
    )
    results = await runner.aevaluate_response_strs(
        queries=["q1", "q2"],
        response_strs=["r1", "r2"],
        contexts_list=[["c1"], ["c2"]],
    )
    assert set(results) == {"faithfulness", "relevancy"}
    for items in results.values():
        assert len(items) == 2
        assert all(isinstance(item, NaceEvaluationResult) for item in items)
    assert results["relevancy"][0].feedback == "P(relevant)=0.90"


async def test_evaluate_inside_running_loop(drex: FakeDrex, make_client: MakeClient) -> None:
    assert asyncio.get_running_loop()
    result = NaceFaithfulnessEvaluator(client=make_client()).evaluate(response="r", contexts=["c"])
    assert result.passing is True


def test_relevancy_requires_query(drex: FakeDrex, make_client: MakeClient) -> None:
    with pytest.raises(ValueError, match="query"):
        NaceRelevancyEvaluator(client=make_client()).evaluate(response="r", contexts=["c"])
    assert drex.bodies == []


def test_relevancy_fields(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(0.3)
    evaluator = NaceRelevancyEvaluator(client=make_client(), raise_error=True)
    with pytest.raises(ValueError, match="invalid"):
        evaluator.evaluate(query="q", response="r", contexts=["c"])
    question = drex.bodies[0]["questions"]["evaluate"]
    assert "answer the query" in question["instructions"]


def test_prompts_roundtrip(drex: FakeDrex, make_client: MakeClient) -> None:
    evaluator = NaceFaithfulnessEvaluator(client=make_client())
    assert "instructions" in evaluator.get_prompts()
    from llama_index.core.prompts import PromptTemplate

    evaluator.update_prompts({"instructions": PromptTemplate("Custom?")})
    evaluator.evaluate(response="r", contexts=["c"])
    assert drex.bodies[0]["questions"]["evaluate"]["instructions"] == "Custom?"


@pytest.mark.parametrize("evaluator_cls", [NaceFaithfulnessEvaluator, NaceRelevancyEvaluator])
def test_missing_or_wrong_answer_is_a_validation_error(
    drex: FakeDrex, make_client: MakeClient, evaluator_cls: type
) -> None:
    from llama_index.evaluation.nace import NaceError

    evaluator = evaluator_cls(client=make_client())
    drex.replies.append(lambda body: (200, {}, {**ok_body(body), "answers": {}}))
    with pytest.raises(NaceAPIResponseValidationError, match="no answer") as info:
        evaluator.evaluate(query="q", response="r", contexts=["c"])
    assert not isinstance(info.value, KeyError | ValueError)
    drex.overrides = {
        "evaluate": {
            "type": "choice",
            "choice": "a",
            "confidence": 1.0,
            "probabilities": {"a": 1.0},
        }
    }
    with pytest.raises(NaceAPIResponseValidationError, match="unexpected answer type") as info:
        evaluator.evaluate(query="q", response="r", contexts=["c"])
    assert isinstance(info.value, NaceError)
    assert not isinstance(info.value, ValueError)


@pytest.mark.parametrize("evaluator_cls", [NaceFaithfulnessEvaluator, NaceRelevancyEvaluator])
@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
def test_threshold_must_be_a_probability(evaluator_cls: type, threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluator_cls(threshold=threshold, api_key="nace_sk_test")
