from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import nace_sdk
import pytest
from fakes import FakeDrex, error
from llama_index.selectors.nace._drex.answers import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from llama_index.selectors.nace._drex.client import (
    DEFAULT_MODEL,
    IMAGE_MODEL,
    MAX_IMAGES,
    OPENROUTER_DEFAULT_MODEL,
    TEXT_ONLY_MODELS,
    DrexClient,
)
from llama_index.selectors.nace._drex.errors import (
    NaceAPIConnectionError,
    NaceAPIResponseValidationError,
    NaceError,
    NaceImageError,
    NaceImageLimitError,
    NaceImageModelError,
    NaceUnprocessableEntityError,
)
from llama_index.selectors.nace._drex.images import to_drex_image
from llama_index.selectors.nace._version import __version__

FIXTURES = Path(__file__).parent / "fixtures"
RED = to_drex_image((FIXTURES / "red.png").read_bytes())
BLUE = to_drex_image((FIXTURES / "blue.png").read_bytes())
MakeClient = Callable[..., DrexClient]

QUESTIONS: dict[str, Any] = {
    "noul": nace_sdk.Noul(instructions="Is it fine?"),
    "choice": nace_sdk.Choice(instructions="Which?", criteria={"a": "first", "b": "second"}),
    "score": nace_sdk.Score(instructions="How good?", criteria=["bad", "ok", "good"]),
}
ANSWER_TYPES = {"noul": NoulAnswer, "choice": ChoiceAnswer, "score": ScoreAnswer}


def test_constants() -> None:
    assert DEFAULT_MODEL == "drex-max-v1.1"
    assert IMAGE_MODEL == "drex-max-v1.1"
    assert OPENROUTER_DEFAULT_MODEL == "drex-max"
    assert {"drex-v1.5", "drex-v2.0"} == TEXT_ONLY_MODELS
    assert MAX_IMAGES == {"drex-max": 8, "drex-max-v1.1": 8}


@pytest.mark.parametrize("model", ["drex-v2.0", "drex-max", "drex-max-v1.1"])
@pytest.mark.parametrize("kind", ["noul", "choice", "score"])
def test_single_question_each_type(
    make_client: MakeClient, drex: FakeDrex, kind: str, model: str
) -> None:
    response = make_client(model=model).ask("text", {"q": QUESTIONS[kind]})
    assert drex.bodies[-1]["model"] == model
    assert drex.bodies[-1]["questions"]["q"]["type"] == kind
    assert isinstance(response.answers["q"], ANSWER_TYPES[kind])


def test_mixed_batch_one_request(make_client: MakeClient, drex: FakeDrex) -> None:
    response = make_client().ask("text", QUESTIONS)
    assert len(drex.bodies) == 1
    assert set(response.answers) == {"noul", "choice", "score"}
    assert drex.bodies[-1]["state"] == "text"


def test_default_model_text(make_client: MakeClient, drex: FakeDrex) -> None:
    client = make_client()
    assert client.resolve_model(has_images=False) == "drex-max-v1.1"
    client.ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["model"] == "drex-max-v1.1"


def test_default_model_images_stays_max(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client().ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies[-1]["model"] == "drex-max-v1.1"


def test_default_model_env_text_only_v2(
    make_client: MakeClient, drex: FakeDrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-v2.0")
    client = make_client()
    assert client.resolve_model(has_images=True) == "drex-max-v1.1"
    assert client.resolve_model(has_images=False) == "drex-v2.0"
    client.ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies[-1]["model"] == "drex-max-v1.1"
    client.ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["model"] == "drex-v2.0"


def test_default_model_env_unknown_model_kept_with_images(
    make_client: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only known text-only models are swapped; a new release is sent as given.
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-latest")
    assert make_client().resolve_model(has_images=True) == "drex-latest"


def test_default_model_does_not_override_explicit_with_images(
    make_client: MakeClient, drex: FakeDrex
) -> None:
    with pytest.raises(NaceImageModelError):
        make_client(model="drex-v2.0").ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies == []


def test_default_model_env(
    make_client: MakeClient, drex: FakeDrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-v1.5")
    make_client().ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["model"] == "drex-v1.5"


def test_default_model_images(
    make_client: MakeClient, drex: FakeDrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-v1.5")
    client = make_client()
    assert client.resolve_model(has_images=True) == "drex-max-v1.1"
    assert client.resolve_model(has_images=False) == "drex-v1.5"
    client.ask("t", {"q": QUESTIONS["noul"]}, images=[(FIXTURES / "red.png").read_bytes()])
    assert drex.bodies[-1]["model"] == "drex-max-v1.1"


def test_two_images_in_order_and_reversed(make_client: MakeClient, drex: FakeDrex) -> None:
    client = make_client()
    client.ask("t", {"q": QUESTIONS["noul"]}, images=[RED, BLUE])
    assert drex.bodies[-1]["images"] == [RED, BLUE]
    client.ask("t", {"q": QUESTIONS["noul"]}, images=[BLUE, RED])
    assert drex.bodies[-1]["images"] == [BLUE, RED]


def test_images_on_text_model_refused(make_client: MakeClient, drex: FakeDrex) -> None:
    with pytest.raises(NaceImageModelError):
        make_client(model="drex-v2.0").ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies == []


@pytest.mark.parametrize("model", [None, "drex-max", "drex-max-v1.1"])
def test_question_count_left_to_drex(
    make_client: MakeClient, drex: FakeDrex, model: str | None
) -> None:
    # Drex enforces the per-request question cap; its 422 is shown as is.
    many = {f"q{i}": QUESTIONS["noul"] for i in range(17)}
    make_client(model=model).ask("t", many)
    assert len(drex.bodies[-1]["questions"]) == 17
    drex.replies.append(
        error(
            422,
            "Invalid request",
            issues=[{"path": "questions", "message": "Drex Max allows at most 16 questions."}],
        )
    )
    with pytest.raises(NaceUnprocessableEntityError) as raised:
        make_client(model=model).ask("t", many)
    assert "questions: Drex Max allows at most 16 questions." in str(raised.value)


def test_image_model_hint_plain(make_client: MakeClient, drex: FakeDrex) -> None:
    with pytest.raises(NaceImageModelError) as raised:
        make_client(model="drex-v2.0").ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert "model='drex-max-v1.1'" in str(raised.value)
    assert drex.bodies == []


def test_image_model_hint_keeps_openrouter_prefix(make_client: MakeClient, drex: FakeDrex) -> None:
    with pytest.raises(NaceImageModelError) as raised:
        make_client(base_url=OPENROUTER, model="nace-ai/drex-v2.0").ask(
            "t", {"q": QUESTIONS["noul"]}, images=[RED]
        )
    assert "model='nace-ai/drex-max'" in str(raised.value)
    assert drex.bodies == []


@pytest.mark.parametrize("model", [None, "drex-max", "drex-max-v1.1"])
def test_more_than_8_images_on_max_refused(
    make_client: MakeClient, drex: FakeDrex, model: str | None
) -> None:
    with pytest.raises(NaceImageLimitError, match="8"):
        make_client(model=model).ask("t", {"q": QUESTIONS["noul"]}, images=[RED] * 9)
    assert drex.bodies == []
    make_client(model=model).ask("t", {"q": QUESTIONS["noul"]}, images=[RED] * 8)
    assert len(drex.bodies[-1]["images"]) == 8


def test_prefixed_max_limits_apply(make_client: MakeClient, drex: FakeDrex) -> None:
    for model in ("nace-ai/drex-max", "~nace-ai/drex-max", "nace-ai/drex-max-v1.1"):
        with pytest.raises(NaceImageLimitError):
            make_client(model=model).ask("t", {"q": QUESTIONS["noul"]}, images=[RED] * 9)
    assert drex.bodies == []


def test_prefixed_text_model_refuses_images(make_client: MakeClient, drex: FakeDrex) -> None:
    with pytest.raises(NaceImageModelError):
        make_client(model="nace-ai/drex-v2.0").ask("t", {"q": QUESTIONS["noul"]}, images=[RED])


def test_unknown_model_sent_as_given(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(model="drex-latest").ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies[-1]["model"] == "drex-latest"


def test_no_questions_refused(make_client: MakeClient, drex: FakeDrex) -> None:
    with pytest.raises(ValueError, match="question"):
        make_client().ask("t", {})
    assert drex.bodies == []


def test_noul_without_instructions_sends_empty_string(
    make_client: MakeClient, drex: FakeDrex
) -> None:
    question = nace_sdk.Noul(criteria=nace_sdk.NoulCriteria(true="yes case", false="no case"))
    make_client().ask("t", {"q": question})
    assert drex.bodies[-1]["questions"]["q"]["instructions"] == ""


def test_choice_null_description(make_client: MakeClient, drex: FakeDrex) -> None:
    question = nace_sdk.Choice(criteria={"a": None, "b": "desc"})
    make_client().ask("t", {"q": question})
    sent = drex.bodies[-1]["questions"]["q"]
    assert sent["criteria"] == {"a": None, "b": "desc"}
    assert sent["instructions"] == ""


def test_422_surfaces_paths(make_client: MakeClient, drex: FakeDrex) -> None:
    drex.replies.append(
        error(
            422,
            "Invalid request",
            issues=[{"path": "questions.q.criteria", "message": "must not be empty"}],
        )
    )
    with pytest.raises(NaceUnprocessableEntityError) as info:
        make_client().ask("t", {"q": QUESTIONS["noul"]})
    assert "questions.q.criteria" in str(info.value)
    assert "must not be empty" in str(info.value)


def test_429_and_529_retried_after_retry_after_ms(
    make_client: MakeClient, drex: FakeDrex, sleeps: list[float]
) -> None:
    drex.replies += [
        error(429, "slow down", kind="rate_limit_error", headers={"retry-after-ms": "1500"}),
        error(529, "overloaded", kind="overloaded_error"),
    ]
    response = make_client().ask("t", {"q": QUESTIONS["noul"]})
    assert len(drex.requests) == 3
    assert sleeps[0] == 1.5
    assert len(sleeps) == 2
    assert response.request_id == "req_test"


def test_attribution_headers(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client().ask("t", {"q": QUESTIONS["noul"]})
    headers = drex.requests[-1].headers
    assert headers["user-agent"] == f"llama-index-selectors-nace/{__version__}"
    assert headers["x-nace-client"] == f"llama-index-selectors-nace/{__version__}"


def test_default_headers_sent(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(default_headers={"X-Team": "ml"}).ask("t", {"q": QUESTIONS["noul"]})
    assert drex.requests[-1].headers["x-team"] == "ml"


OPENROUTER = "https://openrouter.ai/api/v1"


def test_openrouter_shape(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(base_url=OPENROUTER, model="nace-ai/drex-max").ask(
        "describe", {"q": QUESTIONS["noul"]}, images=[RED]
    )
    request = drex.requests[-1]
    body = drex.bodies[-1]
    assert request.url.path.endswith("/api/v1/systemone")
    assert "images" not in body
    parts = body["state"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert request.headers["http-referer"] == "https://nace.ai"
    assert request.headers["x-openrouter-title"] == "Drex for LlamaIndex"


def test_openrouter_json_state_with_images(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(openrouter=True, base_url="https://gateway.test/api").ask(
        {"a": 1}, {"q": QUESTIONS["noul"]}, images=[RED]
    )
    assert drex.bodies[-1]["state"][0]["text"] == json.dumps({"a": 1})


def test_openrouter_text_only_keeps_plain_state(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(base_url=OPENROUTER).ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["state"] == "t"


def test_openrouter_workspace_file_refused(make_client: MakeClient, drex: FakeDrex) -> None:
    ref = nace_sdk.WorkspaceFileSource(workspace_id="ws_1", file_id="file_abc")
    with pytest.raises(NaceImageError, match="inline"):
        make_client(base_url=OPENROUTER, model="nace-ai/drex-max").ask(
            "t", {"q": QUESTIONS["noul"]}, images=[ref]
        )
    assert drex.bodies == []


def test_openrouter_default_model_prefixed(make_client: MakeClient, drex: FakeDrex) -> None:
    client = make_client(base_url=OPENROUTER)
    client.ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["model"] == "nace-ai/drex-max"
    client.ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies[-1]["model"] == "nace-ai/drex-max"


def test_openrouter_explicit_model_sent_as_given(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(base_url=OPENROUTER, model="drex-v2.0").ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["model"] == "drex-v2.0"


def test_timeout_below_60_rejected(make_client: MakeClient) -> None:
    with pytest.raises(ValueError, match="60"):
        make_client(timeout=30)


def test_empty_model_rejected(make_client: MakeClient) -> None:
    with pytest.raises(ValueError, match="model"):
        make_client(model="  ")


def test_missing_api_key_is_value_error() -> None:
    with pytest.raises(ValueError, match="NACE_API_KEY"):
        DrexClient("x/1")


def test_base_url_from_env_and_blank(
    make_client: MakeClient, drex: FakeDrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NACE_BASE_URL", "https://openrouter.ai/api/v1")
    make_client(base_url=None).ask("t", {"q": QUESTIONS["noul"]})
    assert drex.requests[-1].url.path.endswith("/api/v1/systemone")
    monkeypatch.setenv("NACE_BASE_URL", "  ")
    make_client(base_url=None).ask("t", {"q": QUESTIONS["noul"]})
    assert drex.requests[-1].url.host == "console.nace.ai"


def test_model_kwargs_merged_into_body(make_client: MakeClient, drex: FakeDrex) -> None:
    make_client(model_kwargs={"temperature": 0}).ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["temperature"] == 0


async def test_aask_same_body_as_ask(make_client: MakeClient, drex: FakeDrex) -> None:
    client = make_client(model="drex-max")
    sync = client.ask("t", QUESTIONS, images=[RED, BLUE])
    asynchronous = await client.aask("t", QUESTIONS, images=[RED, BLUE])
    assert drex.bodies[0] == drex.bodies[1]
    assert sync.answers == asynchronous.answers


async def test_aask_error_converted(make_client: MakeClient, drex: FakeDrex) -> None:
    drex.replies.append(error(422, "bad", issues=[{"path": "state", "message": "required"}]))
    with pytest.raises(NaceUnprocessableEntityError):
        await make_client().aask("t", {"q": QUESTIONS["noul"]})


async def test_aask_checks_limits(make_client: MakeClient, drex: FakeDrex) -> None:
    with pytest.raises(NaceImageModelError):
        await make_client(model="drex-v2.0").aask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies == []


def test_connection_error_converted(make_client: MakeClient) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = make_client(
        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    with pytest.raises(NaceAPIConnectionError):
        client.ask("t", {"q": QUESTIONS["noul"]})


SECRET = "PATIENT-SSN-123"


def test_non_json_body_is_validation_error_without_body(
    make_client: MakeClient, drex: FakeDrex
) -> None:
    def plain(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"not json {SECRET}")

    client = make_client(http_client=httpx.Client(transport=httpx.MockTransport(plain)))
    with pytest.raises(NaceAPIResponseValidationError) as info:
        client.ask("t", {"q": QUESTIONS["noul"]})
    assert SECRET not in str(info.value)


def test_malformed_answer_is_validation_error_without_body(
    make_client: MakeClient, drex: FakeDrex
) -> None:
    drex.replies.append((200, {}, {"answers": {"q": {"type": "noul", "noul": SECRET}}}))
    with pytest.raises(NaceAPIResponseValidationError) as info:
        make_client().ask("t", {"q": QUESTIONS["noul"]})
    assert SECRET not in str(info.value)
    assert SECRET not in repr(info.value)
    assert isinstance(info.value, NaceError)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__


def test_non_object_body_is_validation_error(make_client: MakeClient, drex: FakeDrex) -> None:
    drex.replies.append((200, {}, [1, 2]))
    with pytest.raises(NaceAPIResponseValidationError):
        make_client().ask("t", {"q": QUESTIONS["noul"]})


def test_unreadable_result_body_is_validation_error() -> None:
    import types

    from llama_index.selectors.nace._drex.client import _response

    for body in (ValueError(SECRET), [1], {"answers": {"q": {"type": "noul", "noul": SECRET}}}):

        def get_json(body: Any = body) -> Any:
            if isinstance(body, Exception):
                raise body
            return body

        raw = types.SimpleNamespace(json=get_json, headers={})
        result: Any = types.SimpleNamespace(raw_http_response=raw)
        with pytest.raises(NaceAPIResponseValidationError) as info:
            _response(result)
        assert SECRET not in str(info.value)
        assert info.value.__context__ is None


def test_retries_not_cut_short_by_slow_attempts(
    make_client: MakeClient, drex: FakeDrex, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    import types

    from nace_sdk._core import retry

    clock = [0.0]

    def slow_reply(body: dict[str, Any]) -> Any:
        clock[0] += 45.0  # each attempt takes 45 s: past the SDK's default 30 s budget
        return error(429, "slow down", kind="rate_limit_error")

    drex.replies += [slow_reply, slow_reply]
    fake_time = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if k[0] != "_"})
    fake_time.monotonic = lambda: clock[0]
    fake_time.sleep = sleeps.append
    monkeypatch.setattr(retry, "time", fake_time)
    make_client().ask("t", {"q": QUESTIONS["noul"]})
    assert len(drex.requests) == 3
    assert len(sleeps) == 2


def _raising_transport(status: int, content: bytes, content_type: str) -> Any:
    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            content=content,
            headers={"content-type": content_type, "x-request-id": "req_tb"},
        )

    async def areply(request: httpx.Request) -> httpx.Response:
        return reply(request)

    return httpx.MockTransport(reply), httpx.MockTransport(areply)


BODY_CASES = [
    pytest.param(502, f"<html><body>{SECRET}</body></html>".encode(), "text/html", id="502-html"),
    pytest.param(
        500,
        json.dumps({"state": SECRET, "trace": [SECRET]}).encode(),
        "application/json",
        id="500-json-no-message",
    ),
]


@pytest.mark.parametrize(("status", "content", "content_type"), BODY_CASES)
async def test_api_error_body_never_in_traceback(
    make_client: MakeClient, status: int, content: bytes, content_type: str
) -> None:
    import traceback

    sync_transport, async_transport = _raising_transport(status, content, content_type)
    client = make_client(
        max_retries=0,
        http_client=httpx.Client(transport=sync_transport),
        http_async_client=httpx.AsyncClient(transport=async_transport),
    )
    errors: list[BaseException] = []
    with pytest.raises(NaceError) as info:
        client.ask("t", {"q": QUESTIONS["noul"]})
    errors.append(info.value)
    with pytest.raises(NaceError) as ainfo:
        await client.aask("t", {"q": QUESTIONS["noul"]})
    errors.append(ainfo.value)
    for err in errors:
        assert getattr(err, "status", None) == status
        assert getattr(err, "request_id", None) == "req_tb"
        assert err.__cause__ is None
        text = "".join(traceback.format_exception(err))
        assert SECRET not in text
        assert f"Drex returned {status}" in text


def test_connection_error_keeps_cause(make_client: MakeClient) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = make_client(
        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    with pytest.raises(NaceAPIConnectionError) as info:
        client.ask("t", {"q": QUESTIONS["noul"]})
    assert isinstance(info.value.__cause__, nace_sdk.NaceAPIConnectionError)


async def test_shared_http_client_user_agent_per_drex_client(drex: FakeDrex) -> None:
    sync_http = httpx.Client(transport=drex.transport())
    async_http = httpx.AsyncClient(transport=drex.async_transport())
    clients = [
        DrexClient(
            name,
            api_key="nace_sk_test",
            base_url="https://drex.test",
            http_client=sync_http,
            http_async_client=async_http,
        )
        for name in ("first-pkg/1.0", "second-pkg/2.0")
    ]
    for client in clients:
        client.ask("t", {"q": QUESTIONS["noul"]})
        await client.aask("t", {"q": QUESTIONS["noul"]})
    names = [r.headers["x-nace-client"] for r in drex.requests]
    assert names == ["first-pkg/1.0"] * 2 + ["second-pkg/2.0"] * 2
    assert [r.headers["user-agent"] for r in drex.requests] == names
    assert len(sync_http.event_hooks["request"]) == 1
    assert len(async_http.event_hooks["request"]) == 1


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://openrouter.ai/api/v1", True),
        ("https://eu.openrouter.ai/api/v1", True),
        ("https://notopenrouter.ai/api/v1", False),
        ("https://openrouter.ai.evil.test/api/v1", False),
        (None, False),
    ],
)
def test_is_openrouter_host(url: str | None, expected: bool) -> None:
    from llama_index.selectors.nace._drex.client import _is_openrouter

    assert _is_openrouter(url) is expected


def test_openrouter_env_default_already_prefixed(
    make_client: MakeClient, drex: FakeDrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "nace-ai/drex-v2.0")
    client = make_client(base_url=OPENROUTER)
    client.ask("t", {"q": QUESTIONS["noul"]})
    assert drex.bodies[-1]["model"] == "nace-ai/drex-v2.0"
    client.ask("t", {"q": QUESTIONS["noul"]}, images=[RED])
    assert drex.bodies[-1]["model"] == "nace-ai/drex-max"


def test_openrouter_without_base_url_rejected(make_client: MakeClient) -> None:
    with pytest.raises(ValueError, match="base_url"):
        make_client(openrouter=True, base_url=None)


async def test_close_only_owned_http_clients() -> None:
    shared = httpx.Client()
    ashared = httpx.AsyncClient()
    borrowed = DrexClient(
        "x/1", api_key="nace_sk_test", http_client=shared, http_async_client=ashared
    )
    borrowed.close()
    await borrowed.aclose()
    assert not shared.is_closed
    assert not ashared.is_closed

    owned = DrexClient("x/1", api_key="nace_sk_test")
    owned.close()
    assert owned._client._http_client.is_closed  # type: ignore[attr-defined]
    await owned.aclose()
    assert owned._aclient._http_client.is_closed  # type: ignore[attr-defined]
    shared.close()
    await ashared.aclose()
