import httpx
import nace_sdk
import pytest
from llama_index.selectors.nace._drex.errors import (
    BILLING_URL,
    NaceAPIConnectionError,
    NaceAPIError,
    NaceAPITimeoutError,
    NaceAuthenticationError,
    NaceBadRequestError,
    NaceError,
    NaceImageError,
    NaceImageModelError,
    NaceInsufficientCreditError,
    NaceInternalServerError,
    NaceLowConfidenceError,
    NaceNotFoundError,
    NaceOverloadedError,
    NacePermissionDeniedError,
    NaceRateLimitError,
    NaceRedirectError,
    NaceUnprocessableEntityError,
    convert_error,
)

SDK = {
    400: nace_sdk.NaceBadRequestError,
    401: nace_sdk.NaceAuthenticationError,
    402: nace_sdk.NaceInsufficientCreditError,
    403: nace_sdk.NacePermissionDeniedError,
    404: nace_sdk.NaceNotFoundError,
    422: nace_sdk.NaceUnprocessableEntityError,
    429: nace_sdk.NaceRateLimitError,
    500: nace_sdk.NaceInternalServerError,
    529: nace_sdk.NaceOverloadedError,
}


def sdk_error(status: int, body: object = None, request_id: str | None = "req_1"):
    headers = httpx.Headers({"x-request-id": request_id} if request_id else {})
    return SDK[status](status, body if body is not None else {"error": "nope"}, headers)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, NaceBadRequestError),
        (401, NaceAuthenticationError),
        (402, NaceInsufficientCreditError),
        (403, NacePermissionDeniedError),
        (404, NaceNotFoundError),
        (422, NaceUnprocessableEntityError),
        (429, NaceRateLimitError),
        (500, NaceInternalServerError),
        (529, NaceOverloadedError),
    ],
)
def test_status_maps_to_class(status, expected):
    err = convert_error(sdk_error(status))
    assert isinstance(err, expected)
    assert isinstance(err, NaceAPIError)
    assert err.status == err.status_code == status
    assert err.request_id == "req_1"


def test_overloaded_is_an_internal_server_error():
    assert isinstance(convert_error(sdk_error(529)), NaceInternalServerError)


def test_402_message_has_billing_link():
    assert BILLING_URL in str(convert_error(sdk_error(402)))


def test_422_lists_every_issue():
    body = {
        "error": {
            "message": "invalid",
            "issues": [
                {"path": "questions.a.type", "message": "bad"},
                {"path": "model", "message": "unknown"},
            ],
        }
    }
    err = convert_error(sdk_error(422, body))
    assert "questions.a.type: bad" in str(err)
    assert "model: unknown" in str(err)


def test_message_never_contains_body():
    err = convert_error(sdk_error(500, {"state": "SECRET"}))
    assert "SECRET" not in str(err)
    assert "SECRET" not in repr(err)


def test_string_body_never_in_message():
    raw = nace_sdk.NaceAPIError(
        502, "<html>SECRET</html>", httpx.Headers({"x-request-id": "req_1"})
    )
    err = convert_error(raw)
    assert "SECRET" not in str(err)
    assert "SECRET" not in repr(err)
    assert "Drex returned 502" in str(err)
    assert "req_1" in str(err)


def test_unknown_status_falls_back():
    err = convert_error(nace_sdk.NaceAPIError(418, {"error": "teapot"}, httpx.Headers()))
    assert type(err) is NaceAPIError
    assert "teapot" in str(err)
    assert "teapot" in repr(err)


def test_other_5xx_is_internal_server_error():
    err = convert_error(nace_sdk.NaceAPIError(503, {"error": "down"}, httpx.Headers()))
    assert isinstance(err, NaceInternalServerError)


def test_transport_errors_are_converted():
    timeout = convert_error(nace_sdk.NaceAPITimeoutError(5.0))
    assert isinstance(timeout, NaceAPITimeoutError)
    assert isinstance(timeout, NaceAPIConnectionError)
    assert isinstance(
        convert_error(nace_sdk.NaceAPIConnectionError("down")), NaceAPIConnectionError
    )


def test_other_errors():
    assert type(convert_error(nace_sdk.NaceError("x"))) is NaceError
    other = ValueError("x")
    assert convert_error(other) is other


def test_client_side_errors():
    assert issubclass(NaceImageError, ValueError)
    assert issubclass(NaceImageModelError, NaceImageError)
    low = NaceLowConfidenceError("unsure", probabilities={"a": 0.4})
    assert low.probabilities == {"a": 0.4}


def test_sdk_redirect_refusal_maps_to_redirect_error():
    refusal = convert_error(nace_sdk.NaceError("Refusing an API redirect to a different origin"))
    assert isinstance(refusal, NaceRedirectError)
    assert "different origin" in str(refusal)
    other = convert_error(nace_sdk.NaceError("other"))
    assert type(other) is NaceError
    assert not isinstance(other, NaceRedirectError)
