"""A fake NDI server for unit tests: records requests, answers the document routes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

#: The API host the reader is pointed at, and the separate host an upload grant names.
UPLOAD_URL = "https://upload.test/uploads"
WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
FILE_ID = "22222222-2222-2222-2222-222222222222"
JOB_ID = "33333333-3333-3333-3333-333333333333"
UPLOAD_TOKEN = "upload-token-test"
CONTENT_URL = f"https://files.test/v1/documents/jobs/{JOB_ID}/document-content/markdown.md"

MARKDOWN = "# Test document\n\nMade-up Markdown for offline tests."
#: What `GET <content_url>` returns: the full text of a result that was too large to inline.
FULL_MARKDOWN = "# Test document\n\n" + "The full text that was too large to inline. " * 600
PAGE_CHUNKS = [
    {
        "id": "chk_1",
        "content": "# Test document\n\nMade-up page one.",
        "location": {"kind": "page_region", "page": 1, "polygons": []},
    },
    {
        "id": "chk_2",
        "content": "Made-up page two.",
        "location": {"kind": "page_region", "page": 2, "polygons": []},
    },
]

Reply = tuple[int, dict[str, str], Any]
JOBS_PATH = "/v1/documents/jobs/"


def parse_result(
    markdown: str = MARKDOWN,
    chunks: Sequence[dict[str, Any]] = PAGE_CHUNKS,
    content_url: str = CONTENT_URL,
    disclosures: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """A parse result the reader accepts, with optional truncation disclosures."""
    result: dict[str, Any] = {
        "document": {
            "markdown": markdown,
            "chunks": list(chunks),
            "content_url": content_url,
            "page_count": 2,
        }
    }
    if disclosures:
        result["disclosures"] = list(disclosures)
    return result


def job_body(status: str = "succeeded", result: Any = None, error: Any = None) -> dict[str, Any]:
    """A job as the document routes return it."""
    body: dict[str, Any] = {"job_id": JOB_ID, "kind": "parse", "status": status}
    if result is not None:
        body["result"] = result
    if error is not None:
        body["error"] = error
    return body


def error(
    status: int,
    message: str,
    *,
    kind: str = "invalid_request_error",
    headers: dict[str, str] | None = None,
) -> Reply:
    payload: dict[str, Any] = {"error": {"type": kind, "message": message}, "request_id": "req_err"}
    return status, headers or {}, payload


def route_of(request: httpx.Request) -> str:
    """The fake's name for a request's route."""
    method, url = request.method, request.url
    if method == "POST" and url.host == "upload.test":
        return "upload"
    path = url.path
    if method == "POST" and path == "/v1/documents/upload-grants":
        return "grant"
    if method == "POST" and path == "/v1/documents/parse":
        return "parse"
    if path.startswith(JOBS_PATH):
        if method == "DELETE":
            return "delete-job"
        if method == "GET":
            return "download" if "/" in path[len(JOBS_PATH) :] else "get"
    if method == "DELETE" and path.startswith("/v1/documents/workspaces/"):
        return "delete-file"
    return "unknown"


def _json_body(request: httpx.Request) -> dict[str, Any]:
    if not request.content or "application/json" not in request.headers.get("content-type", ""):
        return {}
    try:
        body = json.loads(request.content)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


@dataclass
class FakeNdi:
    """Answers every document route from made-up data; `replies` per route go first, in order.

    A `str` payload (the `download` route) is answered as `text/markdown`, everything else
    as JSON.
    """

    result: dict[str, Any] = field(default_factory=parse_result)
    replies: dict[str, list[Reply]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    @property
    def routes(self) -> list[str]:
        """The route name of each recorded request, in order."""
        return [route_of(request) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        route = route_of(request)
        self.requests.append(request)
        self.bodies.append(_json_body(request))
        queue = self.replies.get(route)
        status, extra, payload = queue.pop(0) if queue else self._default(route)
        headers = {"x-request-id": "req_test", **extra}
        if isinstance(payload, str):
            return httpx.Response(
                status,
                content=payload.encode(),
                headers={"content-type": "text/markdown; charset=utf-8", **headers},
            )
        return httpx.Response(status, json=payload, headers=headers)

    def _default(self, route: str) -> Reply:
        """The reply a route gets when `replies` holds none for it."""
        if route == "grant":
            return (
                200,
                {},
                {
                    "workspace_id": WORKSPACE_ID,
                    "upload_url": UPLOAD_URL,
                    "token": UPLOAD_TOKEN,
                    "expires_at": "2027-01-01T00:00:00Z",
                },
            )
        if route == "upload":
            return (
                200,
                {},
                {
                    "status": "succeeded",
                    "result": {
                        "file": {
                            "workspace_id": WORKSPACE_ID,
                            "file_id": FILE_ID,
                            "path": "report.pdf",
                        }
                    },
                },
            )
        if route in {"parse", "get"}:
            return 200, {}, job_body(status="succeeded", result=self.result)
        if route == "download":
            return 200, {}, FULL_MARKDOWN
        if route == "delete-job":
            return 200, {}, {}
        if route == "delete-file":
            return (
                200,
                {},
                {
                    "object": "file_deletion",
                    "workspace_id": WORKSPACE_ID,
                    "file_id": FILE_ID,
                    "status": "deleted",
                },
            )
        return 404, {}, {"error": {"type": "not_found", "message": f"no route for {route}"}}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)
