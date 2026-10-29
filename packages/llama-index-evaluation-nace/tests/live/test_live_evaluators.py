"""Live evaluator checks. Invented text and a generated chart only."""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from conftest import OPENROUTER_KEY, live_options
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import ImageNode, NodeWithScore, TextNode
from llama_index.evaluation.nace import (
    NaceCorrectnessEvaluator,
    NaceEvaluationResult,
    NaceFaithfulnessEvaluator,
    NaceRelevancyEvaluator,
)

pytestmark = pytest.mark.live

CONTEXT = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was completed in 1889 and is about 330 metres tall."
]
FAITHFUL = "The Eiffel Tower is in Paris and was completed in 1889."
UNFAITHFUL = "The Eiffel Tower is in Rome and was built by the Romans in 1889."
QUERY = "Where is the Eiffel Tower and when was it finished?"


def seen(result: Any, record: Any) -> NaceEvaluationResult:
    assert isinstance(result, NaceEvaluationResult)
    assert result.score is not None
    record(result.model or "", result.request_id)
    return result


def test_faithful_ranks_above_unfaithful(record: Any) -> None:
    evaluator = NaceFaithfulnessEvaluator(**live_options())
    good = seen(evaluator.evaluate(response=FAITHFUL, contexts=CONTEXT), record)
    bad = seen(evaluator.evaluate(response=UNFAITHFUL, contexts=CONTEXT), record)
    assert good.score is not None
    assert bad.score is not None
    assert good.passing
    assert not bad.passing
    assert good.score > bad.score


def test_relevancy(record: Any) -> None:
    evaluator = NaceRelevancyEvaluator(**live_options())
    on_topic = seen(evaluator.evaluate(QUERY, FAITHFUL, CONTEXT), record)
    off_topic = seen(
        evaluator.evaluate(
            QUERY, "Bananas are rich in potassium and grow in warm climates.", CONTEXT
        ),
        record,
    )
    assert on_topic.passing
    assert not off_topic.passing


def test_correctness_with_reference(record: Any) -> None:
    evaluator = NaceCorrectnessEvaluator(**live_options())
    reference = "Paris, France; it was completed in 1889."
    good = seen(evaluator.evaluate(QUERY, FAITHFUL, reference=reference), record)
    bad = seen(evaluator.evaluate(QUERY, UNFAITHFUL, reference=reference), record)
    assert good.score is not None
    assert bad.score is not None
    assert good.score >= 4
    assert bad.score <= 2.5
    assert good.probability is not None
    assert 0 <= good.probability <= 1


async def test_aevaluate(record: Any) -> None:
    evaluator = NaceFaithfulnessEvaluator(**live_options())
    result = seen(await evaluator.aevaluate(response=FAITHFUL, contexts=CONTEXT), record)
    assert result.passing


def bar_chart_png() -> bytes:
    """A bar chart of invented sales: Q1=20, Q2=35, Q3=90, Q4=50 (Q3 is clearly tallest)."""
    from PIL import Image, ImageDraw  # test-only dependency

    image = Image.new("RGB", (480, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.text((150, 10), "Sales per quarter (units)", fill="black")
    draw.line([(40, 280), (460, 280)], fill="black", width=2)
    for i, (label, value) in enumerate([("Q1", 20), ("Q2", 35), ("Q3", 90), ("Q4", 50)]):
        left = 70 + i * 100
        top = 280 - value * 2.5
        draw.rectangle([left, top, left + 60, 280], fill=(50, 120, 173))
        draw.text((left + 20, 288), label, fill="black")
        draw.text((left + 20, top - 14), str(value), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def chart_response(answer: str) -> Response:
    node = ImageNode(
        image=base64.b64encode(bar_chart_png()).decode(),
        image_mimetype="image/png",
        text="Bar chart of sales per quarter.",
    )
    return Response(response=answer, source_nodes=[NodeWithScore(node=node, score=0.9)])


def test_image_node_through_evaluate_response(record: Any) -> None:
    evaluator = NaceFaithfulnessEvaluator(model="drex-max", **live_options())
    query = "Which quarter had the highest sales?"
    good = seen(
        evaluator.evaluate_response(query, chart_response("Q3 had the highest sales.")), record
    )
    bad = seen(
        evaluator.evaluate_response(query, chart_response("Q1 had the highest sales.")), record
    )
    assert good.model == "drex-max"
    assert good.passing
    assert not bad.passing


def test_text_and_image_sources_together(record: Any) -> None:
    evaluator = NaceFaithfulnessEvaluator(**live_options())
    chart = chart_response("Q3 had the highest sales.").source_nodes[0]
    text = NodeWithScore(node=TextNode(text="The company sells widgets."), score=0.8)
    response = Response(response="Q3 had the highest sales.", source_nodes=[text, chart])
    result = seen(evaluator.evaluate_response("Which quarter sold most?", response), record)
    assert result.model == "drex-max-v1.1"  # the default model; it reads images
    assert result.passing


@pytest.mark.skipif(not OPENROUTER_KEY, reason="set OPENROUTER_API_KEY_LIVE to run")
def test_openrouter_faithfulness(record: Any) -> None:
    evaluator = NaceFaithfulnessEvaluator(
        api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1"
    )
    result = seen(evaluator.evaluate(response=FAITHFUL, contexts=CONTEXT), record)
    assert result.passing
