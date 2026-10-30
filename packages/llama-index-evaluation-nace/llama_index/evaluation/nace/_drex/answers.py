"""Parse Drex response bodies into typed answers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import NaceAPIResponseValidationError

Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class NoulAnswer(BaseModel):
    """Probability that the answer to a yes/no question is yes."""

    type: Literal["noul"] = "noul"
    noul: Probability


class ChoiceAnswer(BaseModel):
    """The selected label with every label's probability."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: Probability | None = None
    probabilities: dict[str, Probability] = Field(default_factory=dict)


class ScoreAnswer(BaseModel):
    """Expected score with its rubric and level probabilities."""

    type: Literal["score"] = "score"
    score: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    legend: dict[int, str] = Field(default_factory=dict)
    probabilities: dict[int, Probability] = Field(default_factory=dict)
    confidence: Probability | None = None


class RawAnswer(BaseModel):
    """An answer of a type this package doesn't model; its fields are kept in `data`."""

    type: str
    data: dict[str, Any] = Field(default_factory=dict)


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer, Field(union_mode="left_to_right")
]


class Usage(BaseModel):
    """Token usage reported by Drex."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    image_tokens: int | None = None


MAX_MESSAGE_CHARS = 200
# The only path segments an error message may name: this package's own schema fields. Every
# other segment (dict keys, list indexes) comes from the response and is replaced.
_SCHEMA_FIELDS = frozenset(
    name for model in (NoulAnswer, ChoiceAnswer, ScoreAnswer, Usage) for name in model.model_fields
)


def _safe_path(loc: tuple[int | str, ...]) -> str:
    """Render a pydantic error location naming schema fields only, never response keys."""
    return "".join(
        f".{part}" if isinstance(part, str) and part in _SCHEMA_FIELDS else ".<key>" for part in loc
    )


class DrexResponse(BaseModel):
    """Answers and metadata from one Drex request."""

    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer]
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None
    evaluation_time_ms: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @classmethod
    def from_body(cls, body: Mapping[str, Any], request_id: str | None) -> DrexResponse:
        """Build a response from a Drex response body, keeping unknown answer types."""
        answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer] = {}
        for name, raw in (body.get("answers") or {}).items():
            invalid: str | None = None
            try:
                answers[name] = _parse_answer(raw)
            except ValidationError as error:
                first = error.errors()[0]["loc"] if error.errors() else ()
                invalid = _safe_path(first)
            # Raised outside the `except` so no exception context (and no value) is attached.
            if invalid is not None:
                message = f"Drex returned an invalid answer at answers.<answer>{invalid}."
                raise NaceAPIResponseValidationError(message[:MAX_MESSAGE_CHARS])
        return cls(
            model=str(body.get("model", "")),
            answers=answers,
            usage=Usage.model_validate(body.get("usage") or {}),
            request_id=request_id or body.get("request_id"),
            evaluation_time_ms=body.get("evaluation_time_ms"),
            raw=dict(body),
        )


def _parse_answer(raw: Mapping[str, Any]) -> NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer:
    kind = raw.get("type")
    if kind == "noul":
        return NoulAnswer.model_validate(raw)
    if kind == "choice":
        return ChoiceAnswer.model_validate(raw)
    if kind == "score":
        data = dict(raw)
        legend = data.get("legend")
        if isinstance(legend, list):
            data["legend"] = dict(enumerate(legend))
        return ScoreAnswer.model_validate(data)
    return RawAnswer(
        type=str(kind), data={key: value for key, value in raw.items() if key != "type"}
    )


__all__ = [
    "Answer",
    "ChoiceAnswer",
    "DrexResponse",
    "NoulAnswer",
    "RawAnswer",
    "ScoreAnswer",
    "Usage",
]
