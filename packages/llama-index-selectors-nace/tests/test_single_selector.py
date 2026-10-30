from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeDrex, error, ok_body
from llama_index.core.base.base_selector import SingleSelection
from llama_index.core.llms import MockLLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.selectors.nace import (
    NaceError,
    NaceImageError,
    NaceLowConfidenceError,
    NaceSelectorResult,
    NaceSingleSelector,
)
from llama_index.selectors.nace._drex.client import DrexClient
from llama_index.selectors.nace._drex.errors import (
    NaceAPIResponseValidationError,
    NaceInternalServerError,
)
from llama_index.selectors.nace.base import DEFAULT_SINGLE_INSTRUCTIONS, option_labels

FIXTURES = Path(__file__).parent / "fixtures"
MakeClient = Callable[..., DrexClient]

CHOICES = [
    ToolMetadata(name="docs", description="Product documentation"),
    ToolMetadata(name="sql", description="Revenue tables"),
]


def reply(choice: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "route": {
            "type": "choice",
            "choice": choice,
            "confidence": probabilities[choice],
            "probabilities": probabilities,
        }
    }


def test_select_returns_top_choice_with_reason(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = reply("sql", {"docs": 0.07, "sql": 0.93})
    result = NaceSingleSelector(client=make_client()).select(CHOICES, "query text")
    assert isinstance(result, NaceSelectorResult)
    assert result.ind == 1
    assert result.reason == "p=0.93"
    assert result.probabilities == {"docs": 0.07, "sql": 0.93}
    assert result.labels == ["docs", "sql"]
    assert result.model == "drex-max-v1.1"
    assert result.request_id == "req_test"
    assert result.usage.input_tokens == 120
    assert result.fell_back is False


def test_choice_criteria_from_descriptions(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceSingleSelector(client=make_client()).select(CHOICES, "query text")
    body = drex.bodies[0]
    assert body["state"] == "query text"
    question = body["questions"]["route"]
    assert question["criteria"] == {"docs": "Product documentation", "sql": "Revenue tables"}
    assert question["instructions"] == DEFAULT_SINGLE_INSTRUCTIONS


def test_labels_fall_back_to_option_n(drex: FakeDrex, make_client: MakeClient) -> None:
    dup = [ToolMetadata(name="a", description="x"), ToolMetadata(name="a", description="y")]
    assert option_labels(dup) == ["option_1", "option_2"]
    assert option_labels([ToolMetadata(name="a", description="x")]) == ["a"]
    choices: list[Any] = [
        ToolMetadata(name="a", description="x"),
        ToolMetadata(name="b", description=None),  # type: ignore[arg-type]
        "plain string",
    ]
    result = NaceSingleSelector(client=make_client()).select(choices, "q")
    assert result.labels == ["option_1", "option_2", "option_3"]
    criteria = drex.bodies[0]["questions"]["route"]["criteria"]
    assert criteria == {"option_1": "x", "option_2": None, "option_3": "plain string"}
    assert result.ind == 0


def test_min_confidence_falls_back_to_default_index(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    drex.overrides = reply("sql", {"docs": 0.2, "sql": 0.41})
    selector = NaceSingleSelector(client=make_client(), min_confidence=0.6, default_index=0)
    result = selector.select(CHOICES, "q")
    assert result.ind == 0
    assert result.fell_back
    assert "0.41" in result.reason
    assert result.probabilities == {"docs": 0.2, "sql": 0.41}


def test_min_confidence_without_default_raises(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = reply("sql", {"docs": 0.2, "sql": 0.41})
    selector = NaceSingleSelector(client=make_client(), min_confidence=0.6)
    with pytest.raises(NaceLowConfidenceError) as info:
        selector.select(CHOICES, "q")
    assert info.value.probabilities == {"docs": 0.2, "sql": 0.41}


def test_confidence_at_threshold_is_accepted(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = reply("sql", {"docs": 0.4, "sql": 0.6})
    result = NaceSingleSelector(client=make_client(), min_confidence=0.6).select(CHOICES, "q")
    assert result.ind == 1
    assert not result.fell_back


def test_api_error_uses_default_index(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(500, "boom")] * 3
    selector = NaceSingleSelector(client=make_client(max_retries=2), default_index=1)
    result = selector.select(CHOICES, "q")
    assert result.ind == 1
    assert result.fell_back
    assert result.probabilities == {}
    assert result.labels == ["docs", "sql"]
    assert "500" in result.reason


def test_api_error_without_default_propagates(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(500, "boom")] * 3
    with pytest.raises(NaceInternalServerError):
        NaceSingleSelector(client=make_client(max_retries=2)).select(CHOICES, "q")


def test_auth_error_is_not_swallowed(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(401, "bad key")]
    with pytest.raises(NaceError):
        NaceSingleSelector(client=make_client(), default_index=0).select(CHOICES, "q")


def test_out_of_set_choice_raises(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = reply("other", {"other": 1.0})
    with pytest.raises(NaceError):
        NaceSingleSelector(client=make_client()).select(CHOICES, "q")


def test_missing_choice_answer_raises(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = {"route": {"type": "noul", "noul": 0.5}}
    with pytest.raises(NaceError):
        NaceSingleSelector(client=make_client()).select(CHOICES, "q")


def test_choice_missing_from_probabilities_uses_confidence(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    drex.overrides = {
        "route": {"type": "choice", "choice": "docs", "confidence": 0.7, "probabilities": {}}
    }
    result = NaceSingleSelector(client=make_client()).select(CHOICES, "q")
    assert result.ind == 0
    assert result.confidence == 0.7
    assert result.reason == "p=0.70"


def test_query_image_path_sends_image_on_max(drex: FakeDrex, make_client: MakeClient) -> None:
    query = QueryBundle(query_str="what is this", image_path=str(FIXTURES / "red.png"))
    NaceSingleSelector(client=make_client()).select(CHOICES, query)
    body = drex.bodies[0]
    assert body["model"] == "drex-max-v1.1"
    assert len(body["images"]) == 1


def test_query_image_path_missing(drex: FakeDrex, make_client: MakeClient) -> None:
    query = QueryBundle(query_str="q", image_path=str(FIXTURES / "nope.png"))
    with pytest.raises(NaceImageError):
        NaceSingleSelector(client=make_client()).select(CHOICES, query)
    assert drex.bodies == []


def test_prompts_roundtrip(drex: FakeDrex, make_client: MakeClient) -> None:
    selector = NaceSingleSelector(client=make_client())
    prompts = selector.get_prompts()
    assert set(prompts) == {"instructions"}
    selector.update_prompts({"instructions": PromptTemplate("X")})
    selector.select(CHOICES, "q")
    assert drex.bodies[0]["questions"]["route"]["instructions"] == "X"


def test_custom_instructions(drex: FakeDrex, make_client: MakeClient) -> None:
    NaceSingleSelector(client=make_client(), instructions="Route it.").select(CHOICES, "q")
    assert drex.bodies[0]["questions"]["route"]["instructions"] == "Route it."


class Marker(CustomQueryEngine):
    name: str

    def custom_query(self, query_str: str) -> str:
        return self.name


def test_router_query_engine_single(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = reply("sql", {"docs": 0.1, "sql": 0.9})
    tools = [QueryEngineTool(query_engine=Marker(name=c.name or ""), metadata=c) for c in CHOICES]
    router = RouterQueryEngine(
        selector=NaceSingleSelector(client=make_client()),
        query_engine_tools=tools,
        llm=MockLLM(),
        summarizer=None,
    )
    assert str(router.query("how much revenue?")) == "sql"


def test_select_inside_running_loop(drex: FakeDrex, make_client: MakeClient) -> None:
    selector = NaceSingleSelector(client=make_client())

    async def run() -> SingleSelection:
        return selector.select(CHOICES, "q").selections[0]

    assert asyncio.run(run()).index == 0


def test_aselect_matches_select(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.overrides = reply("sql", {"docs": 0.07, "sql": 0.93})
    selector = NaceSingleSelector(client=make_client())
    sync = selector.select(CHOICES, "q")
    result = asyncio.run(selector.aselect(CHOICES, "q"))
    assert result.model_dump(exclude={"usage"}) == sync.model_dump(exclude={"usage"})


def test_aselect_fallback(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies = [error(500, "boom")] * 3
    selector = NaceSingleSelector(client=make_client(max_retries=2), default_index=0)
    assert asyncio.run(selector.aselect(CHOICES, "q")).fell_back


def test_from_defaults_and_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_x")
    selector = NaceSingleSelector.from_defaults(model="drex-max", default_index=0)
    assert isinstance(selector._client, DrexClient)
    assert selector._client.model == "drex-max"


@pytest.mark.parametrize("min_confidence", [-0.1, 1.5])
def test_min_confidence_out_of_range_rejected(
    make_client: MakeClient, min_confidence: float
) -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        NaceSingleSelector(client=make_client(), min_confidence=min_confidence)


@pytest.mark.parametrize("min_confidence", [0.0, 1.0, None])
def test_min_confidence_bounds_accepted(
    make_client: MakeClient, min_confidence: float | None
) -> None:
    NaceSingleSelector(client=make_client(), min_confidence=min_confidence)


def test_empty_choices_value_error(drex: FakeDrex, make_client: MakeClient) -> None:
    selector = NaceSingleSelector(client=make_client())
    with pytest.raises(ValueError, match="choices"):
        selector.select([], "q")
    with pytest.raises(ValueError, match="choices"):
        asyncio.run(selector.aselect([], "q"))
    assert drex.bodies == []


def test_empty_description_sent_as_null(drex: FakeDrex, make_client: MakeClient) -> None:
    choices = [ToolMetadata(name="a", description=""), ToolMetadata(name="b", description="B")]
    NaceSingleSelector(client=make_client()).select(choices, "q")
    assert drex.bodies[0]["questions"]["route"]["criteria"] == {"a": None, "b": "B"}


@pytest.mark.parametrize("value", [-1, True, "0", 1.0])
def test_default_index_must_be_a_non_negative_int(value: Any) -> None:
    with pytest.raises(ValueError, match="default_index"):
        NaceSingleSelector(default_index=value, api_key="nace_sk_test")


def test_default_index_out_of_range_refused_before_request(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    selector = NaceSingleSelector(client=make_client(), default_index=2)
    with pytest.raises(ValueError, match="default_index"):
        selector.select(CHOICES, "q")
    with pytest.raises(ValueError, match="default_index"):
        asyncio.run(selector.aselect(CHOICES, "q"))
    assert drex.requests == []


def test_router_query_engine_never_gets_an_out_of_range_index(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    drex.replies = [error(503, "down")] * 3
    tools = [QueryEngineTool(query_engine=Marker(name=c.name or ""), metadata=c) for c in CHOICES]
    router = RouterQueryEngine(
        selector=NaceSingleSelector(client=make_client(), default_index=7),
        query_engine_tools=tools,
        llm=MockLLM(),
        summarizer=None,
    )
    with pytest.raises(ValueError, match="default_index"):
        router.query("q")


def test_missing_answer_is_a_validation_error(drex: FakeDrex, make_client: MakeClient) -> None:
    drex.replies.append(lambda body: (200, {}, {**ok_body(body), "answers": {}}))
    with pytest.raises(NaceAPIResponseValidationError) as info:
        NaceSingleSelector(client=make_client()).select(CHOICES, "q")
    assert isinstance(info.value, NaceError)


def test_choice_outside_options_is_a_validation_error(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    drex.overrides = reply("zzz", {"zzz": 1.0})
    with pytest.raises(NaceAPIResponseValidationError):
        NaceSingleSelector(client=make_client()).select(CHOICES, "q")


def test_nan_probability_cannot_bypass_min_confidence(
    drex: FakeDrex, make_client: MakeClient
) -> None:
    raw = (
        b'{"model":"m","usage":{},"answers":{"route":{"type":"choice","choice":"docs",'
        b'"confidence":0.9,"probabilities":{"docs":NaN,"sql":0.1}}}}'
    )
    drex.replies = [(200, {}, raw)]
    selector = NaceSingleSelector(client=make_client(), min_confidence=0.9)
    with pytest.raises(NaceAPIResponseValidationError):
        selector.select(CHOICES, "q")
