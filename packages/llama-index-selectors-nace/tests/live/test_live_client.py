"""Live contract matrix for `DrexClient`. Invented data only."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import nace_sdk
import pytest
from conftest import KEY, OPENROUTER_KEY
from llama_index.selectors.nace._drex.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from llama_index.selectors.nace._drex.client import DrexClient
from llama_index.selectors.nace._drex.errors import NaceError

pytestmark = pytest.mark.live

FIXTURES = Path(__file__).parents[1] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()
MODELS = ["drex-v2.0", "drex-max", "drex-max-v1.1"]
MAX_MODELS = ["drex-max", "drex-max-v1.1"]
TICKET = (
    "I was charged twice for my March invoice and I'm really annoyed. Please refund one of them."
)


def client(model: str | None, **kwargs: Any) -> DrexClient:
    base_url = os.environ.get("NACE_BASE_URL_LIVE") or None
    return DrexClient("llama-index-selectors-nace/live", model=model, api_key=KEY,
                      base_url=base_url, **kwargs)  # fmt: skip


def probabilities_sum_to_one(answer: ChoiceAnswer | ScoreAnswer) -> None:
    assert abs(sum(answer.probabilities.values()) - 1) < 1e-2


def test_default_model_is_drex_max_v1_1(record: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NACE_DEFAULT_DECISION_MODEL", raising=False)
    response = client(None).ask(
        TICKET, {"refund": nace_sdk.Noul(instructions="Is the customer asking for a refund?")}
    )
    assert response.model == "drex-max-v1.1"
    answer = response.answers["refund"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul > 0.7
    record(response.model, response.request_id)


@pytest.mark.parametrize("model", MODELS)
def test_single_noul(model: str, record: Any) -> None:
    response = client(model).ask(
        TICKET, {"refund": nace_sdk.Noul(instructions="Is the customer asking for a refund?")}
    )
    answer = response.answers["refund"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul > 0.7
    assert (response.usage.input_tokens or 0) > 0
    record(model, response.request_id)


@pytest.mark.parametrize("model", MODELS)
def test_single_choice(model: str, record: Any) -> None:
    question = nace_sdk.Choice(
        instructions="Which team should handle this?",
        criteria={
            "billing": "Payments, invoices and refunds",
            "technical": "Bugs and outages",
            "sales": "New purchases and quotes",
        },
    )
    response = client(model).ask(TICKET, {"team": question})
    answer = response.answers["team"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    probabilities_sum_to_one(answer)
    record(model, response.request_id)


@pytest.mark.parametrize("model", MODELS)
def test_single_score(model: str, record: Any) -> None:
    question = nace_sdk.Score(
        instructions="How urgent is this ticket?",
        criteria=["Not urgent", "Low", "Moderate", "High", "Critical"],
    )
    response = client(model).ask(TICKET, {"urgency": question})
    answer = response.answers["urgency"]
    assert isinstance(answer, ScoreAnswer)
    assert 0 <= answer.score <= 4
    probabilities_sum_to_one(answer)
    record(model, response.request_id)


@pytest.mark.parametrize("model", MODELS)
def test_mixed_batch_one_request(model: str, record: Any) -> None:
    questions: dict[str, Any] = {
        "refund": nace_sdk.Noul(instructions="Is the customer asking for a refund?"),
        "team": nace_sdk.Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payments and refunds", "technical": "Bugs and outages"},
        ),
        "anger": nace_sdk.Score(
            instructions="How angry is the customer?",
            criteria=["Calm", "Slightly annoyed", "Annoyed", "Angry", "Furious"],
        ),
    }
    response = client(model).ask(TICKET, questions)
    assert set(response.answers) == set(questions)
    assert isinstance(response.answers["refund"], NoulAnswer)
    assert isinstance(response.answers["team"], ChoiceAnswer)
    assert isinstance(response.answers["anger"], ScoreAnswer)
    record(model, response.request_id)


@pytest.mark.parametrize("model", MAX_MODELS)
def test_image_order_matters(model: str, record: Any) -> None:
    question = {"red": nace_sdk.Noul(instructions="Is Image 1 red?")}
    drex = client(model)
    forward = drex.ask("Two solid colour squares.", question, images=[RED, BLUE])
    reverse = drex.ask("Two solid colour squares.", question, images=[BLUE, RED])
    p_forward, p_reverse = forward.answers["red"], reverse.answers["red"]
    assert isinstance(p_forward, NoulAnswer)
    assert isinstance(p_reverse, NoulAnswer)
    assert p_forward.noul > 0.5
    assert p_reverse.noul < 0.5
    record(model, forward.request_id)
    record(model, reverse.request_id)


@pytest.mark.parametrize("model", MAX_MODELS)
def test_too_many_questions_refused(model: str) -> None:
    # The package leaves the cap to Drex and nace-sdk; over it, their message is raised.
    questions = {f"q{i}": nace_sdk.Noul(instructions="Is the sky blue?") for i in range(513)}
    with pytest.raises(NaceError, match="512 questions"):
        client(model).ask("The sky is blue.", questions)


@pytest.mark.skipif(not OPENROUTER_KEY, reason="set OPENROUTER_API_KEY_LIVE to run")
def test_openrouter_noul() -> None:
    drex = DrexClient(
        "llama-index-selectors-nace/live",
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    response = drex.ask(
        TICKET, {"refund": nace_sdk.Noul(instructions="Is the customer asking for a refund?")}
    )
    answer = response.answers["refund"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul > 0.7
    # OpenRouter answers with its own model id and doesn't pass Drex's request id through.
    assert response.model.startswith("nace-ai/")
