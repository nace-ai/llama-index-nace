from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import nace_sdk
import pytest
from fakes import FakeDrex
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode
from llama_index.evaluation.nace import NaceEvaluationResult
from llama_index.evaluation.nace._drex.answers import DrexResponse, NoulAnswer
from llama_index.evaluation.nace._drex.client import DrexClient
from llama_index.evaluation.nace._drex.errors import NaceImageError
from llama_index.evaluation.nace._version import __version__
from llama_index.evaluation.nace.base import CLIENT_NAME, NaceEvaluator

FIXTURES = Path(__file__).parent / "fixtures"
RED = (FIXTURES / "red.png").read_bytes()
RED_URL = "data:image/png;base64," + base64.b64encode(RED).decode()
MakeClient = Callable[..., DrexClient]


class Probe(NaceEvaluator):
    """The smallest concrete evaluator: one noul question."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(instructions="Is it good?", **kwargs)

    def _question(self) -> nace_sdk.Noul:
        return nace_sdk.Noul(instructions=self._instructions.format())

    def _extra_state(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return {"reference": kwargs.get("reference")}

    def _result(
        self,
        response: DrexResponse,
        query: str | None,
        response_str: str | None,
        contexts: Sequence[str] | None,
    ) -> NaceEvaluationResult:
        answer = response.answers["evaluate"]
        assert isinstance(answer, NoulAnswer)
        return NaceEvaluationResult(
            query=query,
            response=response_str,
            contexts=contexts,
            score=answer.noul,
            passing=answer.noul >= 0.5,
            probability=answer.noul,
            model=response.model,
            request_id=response.request_id,
            usage=response.usage,
        )


def test_client_name() -> None:
    assert f"llama-index-evaluation-nace/{__version__}" == CLIENT_NAME


def test_client_built_from_kwargs() -> None:
    probe = Probe(api_key="k", model="drex-v2.0")
    assert probe._client.client_name == CLIENT_NAME
    assert probe._client.model == "drex-v2.0"


def test_state_drops_none(make_client: MakeClient) -> None:
    state = Probe(client=make_client())._build_state(
        "q", None, ["c"], {"reference": None, "x": "y"}
    )
    assert state == {"query": "q", "context": ["c"], "x": "y"}


def test_evaluate_sends_state_and_builds_result(drex: FakeDrex, make_client: MakeClient) -> None:
    result = Probe(client=make_client()).evaluate(
        query="q", response="r", contexts=["c1"], reference="gold"
    )
    assert isinstance(result, NaceEvaluationResult)
    assert result.score == 0.9
    assert result.contexts == ["c1"]
    assert result.model == "drex-max-v1.1"
    assert result.request_id == "req_test"
    body = drex.bodies[0]
    assert body["state"] == {"query": "q", "context": ["c1"], "response": "r", "reference": "gold"}
    assert body["questions"]["evaluate"] == {"type": "noul", "instructions": "Is it good?"}


async def test_aevaluate_matches_sync(drex: FakeDrex, make_client: MakeClient) -> None:
    probe = Probe(client=make_client())
    await probe.aevaluate(query="q", response="r", contexts=["c"])
    probe.evaluate(query="q", response="r", contexts=["c"])
    assert drex.bodies[0] == drex.bodies[1]


async def test_evaluate_inside_running_loop(drex: FakeDrex, make_client: MakeClient) -> None:
    assert asyncio.get_running_loop()
    assert Probe(client=make_client()).evaluate(query="q", response="r").passing is True


def test_image_kwargs_converted_and_model_defaults(drex: FakeDrex, make_client: MakeClient) -> None:
    probe = Probe(client=make_client())
    probe.evaluate(query="q", response="r", images=[RED], image_paths=[FIXTURES / "red.png"])
    assert drex.bodies[0]["images"] == [RED_URL, RED_URL]
    assert drex.bodies[0]["model"] == "drex-max-v1.1"


def test_image_urls_kwarg_data_url_ok_http_refused(drex: FakeDrex, make_client: MakeClient) -> None:
    probe = Probe(client=make_client())
    probe.evaluate(query="q", response="r", image_urls=[RED_URL])
    assert drex.bodies[0]["images"] == [RED_URL]
    with pytest.raises(NaceImageError, match="doesn't fetch"):
        probe.evaluate(query="q", response="r", image_urls=["https://x.test/a.png"])
    assert len(drex.bodies) == 1


def test_image_urls_refuse_file_paths(drex: FakeDrex, make_client: MakeClient) -> None:
    probe = Probe(client=make_client())
    with pytest.raises(NaceImageError, match="image_paths"):
        probe.evaluate(query="q", response="r", image_urls=[str(FIXTURES / "red.png")])
    assert drex.bodies == []


def test_missing_image_path_raises_before_request(drex: FakeDrex, make_client: MakeClient) -> None:
    with pytest.raises(NaceImageError):
        Probe(client=make_client()).evaluate(query="q", image_paths=["/no/such.png"])
    assert drex.bodies == []


def make_response() -> Response:
    nodes = [
        NodeWithScore(node=TextNode(text="plain"), score=0.5),
        NodeWithScore(node=ImageNode(image=RED_URL.split(",")[1], text="chart"), score=0.4),
    ]
    return Response(response="the answer", source_nodes=nodes)


def test_evaluate_response_splits_sources(drex: FakeDrex, make_client: MakeClient) -> None:
    result = Probe(client=make_client()).evaluate_response(query="q", response=make_response())
    body = drex.bodies[0]
    assert body["state"]["context"] == ["plain", "[Image 1] chart"]
    assert body["state"]["response"] == "the answer"
    assert body["images"] == [RED_URL]
    assert result.contexts == ["plain", "[Image 1] chart"]


async def test_aevaluate_response_splits_sources(drex: FakeDrex, make_client: MakeClient) -> None:
    await Probe(client=make_client()).aevaluate_response(query="q", response=make_response())
    assert drex.bodies[0]["images"] == [RED_URL]


def test_evaluate_response_none_and_extra_images(drex: FakeDrex, make_client: MakeClient) -> None:
    Probe(client=make_client()).evaluate_response(query="q", response=None, images=[RED])
    assert drex.bodies[0]["state"] == {"query": "q"}
    assert drex.bodies[0]["images"] == [RED_URL]


def test_max_images_applies_to_sources(drex: FakeDrex, make_client: MakeClient) -> None:
    nodes = [
        NodeWithScore(node=ImageNode(image=RED_URL.split(",")[1], text=str(i)), score=i)
        for i in range(10)
    ]
    probe = Probe(client=make_client(), max_images=2)
    probe.evaluate_response(query="q", response=Response(response="a", source_nodes=nodes))
    assert len(drex.bodies[0]["images"]) == 2


def test_max_images_under_eight_applies_to_sources(drex: FakeDrex, make_client: MakeClient) -> None:
    nodes = [
        NodeWithScore(node=ImageNode(image=RED_URL.split(",")[1], text=str(i)), score=i)
        for i in range(3)
    ]
    probe = Probe(client=make_client(), max_images=1)
    probe.evaluate_response(query="q", response=Response(response="a", source_nodes=nodes))
    assert len(drex.bodies[0]["images"]) == 1


@pytest.mark.parametrize("max_images", [0, -2])
def test_max_images_below_one_rejected(make_client: MakeClient, max_images: int) -> None:
    with pytest.raises(ValueError, match="max_images"):
        Probe(client=make_client(), max_images=max_images)


def test_prompts_roundtrip() -> None:
    from llama_index.core.prompts import PromptTemplate

    probe = Probe(api_key="k")
    assert set(probe.get_prompts()) == {"instructions"}
    probe.update_prompts({"instructions": PromptTemplate("New?")})
    assert probe._question().instructions == "New?"
    probe.update_prompts({})
    assert probe._question().instructions == "New?"


def test_abstract_base_cannot_be_used() -> None:
    evaluator = NaceEvaluator(instructions="x", api_key="k")  # type: ignore[abstract]
    with pytest.raises(NotImplementedError):
        evaluator._question()
    with pytest.raises(NotImplementedError):
        evaluator._result(None, None, None, None)  # type: ignore[arg-type]
