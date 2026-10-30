from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from fakes import FakeDrex, ok_body
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import QueryEngineTool, RetrieverTool, ToolMetadata
from llama_index.selectors.nace import NaceMultiSelector, NaceSelectorResult
from llama_index.selectors.nace._drex.client import DrexClient
from llama_index.selectors.nace._drex.errors import NaceAPIResponseValidationError
from llama_index.selectors.nace.base import DEFAULT_MULTI_INSTRUCTIONS

MakeClient = Callable[..., DrexClient]

TOOLS = [
    ToolMetadata(name="a", description="Alpha docs"),
    ToolMetadata(name="b", description="Beta docs"),
    ToolMetadata(name="c", description="Gamma docs"),
]


def noul(**ps: float) -> dict[str, dict[str, Any]]:
    return {k: {"type": "noul", "noul": v} for k, v in ps.items()}


def test_one_noul_per_option_single_request(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceMultiSelector(client=make_client()).select(TOOLS, "the query")
    assert len(drex.bodies) == 1
    body = drex.bodies[0]
    assert body["state"] == "the query"
    assert list(body["questions"]) == ["a", "b", "c"]
    assert all(q["type"] == "noul" for q in body["questions"].values())
    assert body["questions"]["b"]["instructions"] == (
        DEFAULT_MULTI_INSTRUCTIONS.format(label="b", description="Beta docs")
    )


def test_threshold_and_order(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.2, b=0.9, c=0.6)
    result = NaceMultiSelector(client=make_client()).select(TOOLS, "q")
    assert isinstance(result, NaceSelectorResult)
    assert result.inds == [1, 2]
    assert [s.reason for s in result.selections] == ["p=0.90", "p=0.60"]
    assert result.probabilities == {"a": 0.2, "b": 0.9, "c": 0.6}
    assert result.labels == ["a", "b", "c"]
    assert result.confidence is None
    assert result.fell_back is False
    assert result.request_id == "req_test"


def test_custom_threshold(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.2, b=0.9, c=0.6)
    result = NaceMultiSelector(threshold=0.1, client=make_client()).select(TOOLS, "q")
    assert result.inds == [1, 2, 0]


def test_max_outputs(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.2, b=0.9, c=0.6)
    result = NaceMultiSelector(max_outputs=1, client=make_client()).select(TOOLS, "q")
    assert result.inds == [1]


def test_never_empty(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.2, b=0.31, c=0.1)
    result = NaceMultiSelector(client=make_client()).select(TOOLS, "q")
    assert result.inds == [1]
    assert result.fell_back is True
    assert result.reason == "p=0.31, below threshold 0.5; top option"


def test_more_than_16_options_sent_in_one_request(drex: FakeDrex, make_client: MakeClient) -> None:
    # Drex enforces the question cap itself, so the selector doesn't refuse 17 options.
    tools = [ToolMetadata(name=f"t{i}", description="d") for i in range(17)]
    NaceMultiSelector(client=make_client()).select(tools, "q")
    assert len(drex.bodies) == 1
    assert len(drex.bodies[-1]["questions"]) == 17
    assert drex.bodies[-1]["model"] == "drex-max-v1.1"


def test_labels_fall_back_and_missing_answer(drex: FakeDrex, make_client: MakeClient) -> None:
    dup = [ToolMetadata(name="x", description="1"), ToolMetadata(name="x", description="2")]
    drex.overrides = noul(option_1=0.1, option_2=0.8)
    result = NaceMultiSelector(client=make_client()).select(dup, "q")
    assert result.inds == [1]
    assert result.labels == ["option_1", "option_2"]


def test_wrong_answer_type_raises(drex: FakeDrex, make_client: MakeClient) -> None:
    from llama_index.selectors.nace import NaceError

    drex.overrides = {
        "a": {"type": "choice", "choice": "a", "confidence": 1.0, "probabilities": {"a": 1.0}}
    }
    with pytest.raises(NaceError, match="match"):
        NaceMultiSelector(client=make_client()).select(TOOLS, "q")


def test_aselect_and_running_loop(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.7, b=0.1, c=0.1)
    selector = NaceMultiSelector.from_defaults(client=make_client())
    assert asyncio.run(selector.aselect(TOOLS, "q")).inds == [0]

    async def run() -> list[int]:
        return selector.select(TOOLS, "q").inds

    assert asyncio.run(run()) == [0]


def test_query_image_sent(drex: FakeDrex, make_client: MakeClient) -> None:
    from pathlib import Path

    image = str(Path(__file__).parent / "fixtures" / "red.png")
    bundle = QueryBundle(query_str="q", image_path=image)
    NaceMultiSelector(client=make_client()).select(TOOLS, bundle)
    assert drex.bodies[0]["model"] == "drex-max-v1.1"


def test_instructions_prompt_mixin(make_client: MakeClient) -> None:
    selector = NaceMultiSelector(instructions="Use {label}? {description}", client=make_client())
    assert "instructions" in selector.get_prompts()


class FixedRetriever(BaseRetriever):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=self._text), score=1.0)]


def test_router_retriever_multi(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.9, b=0.8, c=0.1)
    tools = [RetrieverTool(retriever=FixedRetriever(m.name or ""), metadata=m) for m in TOOLS]
    retriever = RouterRetriever(
        selector=NaceMultiSelector(client=make_client()), retriever_tools=tools, llm=MockLLM()
    )
    nodes = retriever.retrieve("q")
    assert sorted(n.node.get_content() for n in nodes) == ["a", "b"]


class Marker(CustomQueryEngine):
    name: str

    def custom_query(self, query_str: str) -> str:
        return self.name


def test_router_query_engine_multi(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = noul(a=0.9, b=0.8, c=0.1)
    tools = [QueryEngineTool(query_engine=Marker(name=m.name or ""), metadata=m) for m in TOOLS]
    router = RouterQueryEngine(
        selector=NaceMultiSelector(client=make_client()),
        query_engine_tools=tools,
        llm=MockLLM(),
    )
    response = router.query("q")
    assert str(response)
    assert len(router.query("q").metadata["selector_result"].selections) == 2


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"max_outputs": 0}, "max_outputs"),
        ({"max_outputs": -1}, "max_outputs"),
        ({"threshold": -0.01}, "threshold"),
        ({"threshold": 1.01}, "threshold"),
    ],
)
def test_invalid_settings_rejected(
    make_client: MakeClient, kwargs: dict[str, Any], name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        NaceMultiSelector(client=make_client(), **kwargs)


def test_boundary_settings_accepted(make_client: MakeClient) -> None:
    NaceMultiSelector(client=make_client(), threshold=0.0, max_outputs=1)
    NaceMultiSelector(client=make_client(), threshold=1.0)


def test_empty_choices_value_error(drex: FakeDrex, make_client: MakeClient) -> None:
    selector = NaceMultiSelector(client=make_client())
    with pytest.raises(ValueError, match="choices"):
        selector.select([], "q")
    with pytest.raises(ValueError, match="choices"):
        asyncio.run(selector.aselect([], "q"))
    assert drex.bodies == []


def test_missing_description_not_rendered_as_none(drex: FakeDrex, make_client: MakeClient) -> None:
    tools = [
        ToolMetadata(name="a", description=None),  # type: ignore[arg-type]
        ToolMetadata(name="b", description="Beta docs"),
    ]
    NaceMultiSelector(client=make_client()).select(tools, "q")
    instructions = drex.bodies[0]["questions"]["a"]["instructions"]
    assert "None" not in instructions
    assert instructions == DEFAULT_MULTI_INSTRUCTIONS.format(label="a", description="")


def test_missing_answer_is_a_validation_error(drex: FakeDrex, make_client: MakeClient) -> None:
    from llama_index.selectors.nace import NaceError

    drex.replies.append(lambda body: (200, {}, {**ok_body(body), "answers": {}}))
    with pytest.raises(NaceAPIResponseValidationError) as info:
        NaceMultiSelector(client=make_client()).select(TOOLS, "q")
    assert isinstance(info.value, NaceError)


def test_wrong_answer_type_is_a_validation_error(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = {
        "a": {"type": "choice", "choice": "a", "confidence": 1.0, "probabilities": {"a": 1.0}}
    }
    with pytest.raises(NaceAPIResponseValidationError):
        NaceMultiSelector(client=make_client()).select(TOOLS, "q")
