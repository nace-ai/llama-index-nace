from llama_index.selectors.nace._drex.answers import (
    ChoiceAnswer,
    DrexResponse,
    NoulAnswer,
    RawAnswer,
    ScoreAnswer,
)


def test_score_answer_int_keys() -> None:
    body = {
        "model": "drex-v2.0",
        "answers": {
            "q": {
                "type": "score",
                "score": 0.9,
                "legend": ["lo", "hi"],
                "probabilities": {"0": 0.1, "1": 0.9},
                "confidence": 0.8,
            }
        },
    }
    answer = DrexResponse.from_body(body, None).answers["q"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.probabilities == {0: 0.1, 1: 0.9}
    assert answer.legend == {0: "lo", 1: "hi"}


def test_noul_and_choice() -> None:
    body = {
        "model": "m",
        "answers": {
            "a": {"type": "noul", "noul": 0.7},
            "b": {"type": "choice", "choice": "x", "probabilities": {"x": 1.0}},
        },
        "request_id": "body-id",
        "evaluation_time_ms": 12.5,
    }
    response = DrexResponse.from_body(body, None)
    assert isinstance(response.answers["a"], NoulAnswer)
    choice = response.answers["b"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.confidence is None
    assert response.request_id == "body-id"
    assert DrexResponse.from_body(body, "hdr").request_id == "hdr"
    assert response.evaluation_time_ms == 12.5
    assert response.raw == body


def test_unknown_answer_type_kept_as_raw() -> None:
    body = {"model": "m", "answers": {"q": {"type": "ranking", "order": [2, 1]}}}
    answer = DrexResponse.from_body(body, None).answers["q"]
    assert isinstance(answer, RawAnswer)
    assert answer.type == "ranking"
    assert answer.data == {"order": [2, 1]}


def test_usage_image_tokens() -> None:
    body = {
        "model": "drex-max",
        "answers": {},
        "usage": {"input_tokens": 100, "output_tokens": 1, "image_tokens": 60},
    }
    usage = DrexResponse.from_body(body, None).usage
    assert (usage.input_tokens, usage.output_tokens, usage.image_tokens) == (100, 1, 60)
    assert DrexResponse.from_body({"model": "m"}, None).usage.image_tokens is None
