"""The reader's offline behaviour against a fake NDI. Invented data only."""

from __future__ import annotations

import re
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fsspec
import httpx
import pytest
from fakes import (
    CONTENT_URL,
    FILE_ID,
    FULL_MARKDOWN,
    JOB_ID,
    MARKDOWN,
    PAGE_CHUNKS,
    UPLOAD_TOKEN,
    WORKSPACE_ID,
    FakeNdi,
    error,
    job_body,
    parse_result,
    route_of,
)
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.nace import NaceReader
from llama_index.readers.nace import base as nace_base
from llama_index.readers.nace._drex.errors import (
    NaceAPIResponseValidationError,
    NaceAuthenticationError,
    NaceError,
    NaceRateLimitError,
)

URL = "https://example.test/files/report.pdf"
SIGNED_URL = "https://user:made-up@example.test/files/report.pdf?token=made-up-token#section"
MakeReader = Callable[..., NaceReader]

BAD_INPUTS = [
    ({"parse_mode": "turbo"}, URL, "parse_mode must be one of"),
    ({"job_timeout": float("inf")}, URL, "finite number above 0"),
    ({"chunking_strategy": "chapter"}, URL, "chunking_strategy must be one of"),
    ({"parse_options": {"chunking": {"strategy": "page"}}}, URL, "parse_options must not contain"),
    (
        {"parse_options": {"extra_body": {"formats": ["text"]}}},
        URL,
        "parse_options must not contain",
    ),
    ({}, "http://example.test/files/report.pdf", "https links only"),
    ({}, "https://example.test/files/report", "no file extension"),
]
UNREADABLE_RESULTS: list[Any] = [
    "not a mapping",
    {"document": "not a mapping"},
    {"document": {"markdown": 42, "chunks": [], "content_url": CONTENT_URL, "page_count": 1}},
    {
        "document": {
            "markdown": MARKDOWN,
            "chunks": [{"id": "chk_1", "content": 7}],
            "page_count": 1,
        }
    },
    {
        "document": {
            "markdown": MARKDOWN,
            "chunks": {"id": "chk_1", "content": "an object, not a list"},
            "page_count": 1,
        }
    },
    {"document": {"markdown": MARKDOWN, "chunks": "not a list", "page_count": 1}},
    {
        "document": {
            "markdown": MARKDOWN,
            "chunks": [],
            "content_url": CONTENT_URL,
            "page_count": 1,
        },
        "disclosures": {"code": "content_truncated"},
    },
    {
        "document": {"markdown": MARKDOWN, "chunks": [], "page_count": 1},
        "disclosures": ["content_truncated"],
    },
]
#: Every upload goes to `llama-index-readers-nace/<32 hex digits>/<file name>`.
UPLOAD_PATH = r"llama-index-readers-nace/[0-9a-f]{32}/report\.pdf"


def test_local_file_uploaded_parsed_and_cleaned_up(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    documents = make_reader().load_data(source, extra_info={"origin": "intranet"})
    assert [document.text for document in documents] == [MARKDOWN]
    metadata = documents[0].metadata
    assert metadata["file_name"] == "report.pdf"
    assert metadata["origin"] == "intranet"
    assert ndi.routes == ["grant", "upload", "parse", "delete-job", "delete-file"]
    assert re.fullmatch(UPLOAD_PATH, ndi.bodies[0]["path"])
    assert b"made-up pdf bytes" in ndi.requests[1].content
    upload = ndi.requests[1]
    assert upload.url.host == "upload.test"
    assert upload.headers["x-upload-token"] == UPLOAD_TOKEN
    assert upload.headers["user-agent"]
    assert "authorization" not in upload.headers
    assert ndi.bodies[2] == {
        "source": {"type": "workspace_file", "workspace_id": WORKSPACE_ID, "file_id": FILE_ID},
        "parse_mode": "low",
        "output": {"formats": ["markdown"]},
    }
    assert ndi.requests[2].url.params["wait_seconds"] == "60"


def test_https_url_needs_no_upload_and_hides_its_query(
    make_reader: MakeReader, ndi: FakeNdi
) -> None:
    documents = make_reader().load_data(SIGNED_URL)
    assert [document.text for document in documents] == [MARKDOWN]
    assert documents[0].metadata["url"] == "https://example.test/files/report.pdf"
    assert ndi.routes == ["parse", "delete-job"]
    assert ndi.bodies[0]["source"] == {"type": "url", "url": SIGNED_URL, "file_name": "report.pdf"}


def test_page_chunking_returns_one_document_per_chunk(
    make_reader: MakeReader, ndi: FakeNdi
) -> None:
    documents = make_reader(chunking_strategy="page").load_data(URL)
    assert [document.text for document in documents] == [chunk["content"] for chunk in PAGE_CHUNKS]
    assert [document.metadata["page_label"] for document in documents] == ["1", "2"]
    assert ndi.bodies[0]["chunking"] == {"strategy": "page"}


def test_queued_job_is_waited_for(make_reader: MakeReader, ndi: FakeNdi) -> None:
    ndi.replies["parse"] = [(202, {}, job_body(status="queued"))]
    documents = make_reader(job_timeout=5).load_data(URL)
    assert [document.text for document in documents] == [MARKDOWN]
    assert ndi.requests[0].url.params["wait_seconds"] == "5"  # held for the budget, not 60 s
    # The poll's HTTP timeout is the time the budget has left, not the SDK's 95-second floor.
    assert ndi.requests[1].extensions["timeout"]["read"] <= 5
    assert ndi.routes == ["parse", "get", "delete-job"]


def test_default_first_poll_read_is_not_the_whole_budget(
    make_reader: MakeReader, ndi: FakeNdi
) -> None:
    ndi.replies["parse"] = [(202, {}, job_body(status="queued"))]
    documents = make_reader().load_data(URL)  # the default job_timeout is 600 s
    assert [document.text for document in documents] == [MARKDOWN]
    # The first poll is held, and its read timeout is the SDK's 95-second floor — not a
    # ~600-second read that a stalled answer could hold open for the whole budget.
    assert ndi.requests[1].url.params["wait_seconds"] == "25"
    assert ndi.requests[1].extensions["timeout"]["read"] <= 95


def test_poll_refuses_a_result_that_arrives_after_the_budget(
    make_reader: MakeReader, ndi: FakeNdi, monkeypatch: pytest.MonkeyPatch
) -> None:
    ndi.replies["parse"] = [(202, {}, job_body(status="queued"))]
    clock: dict[str, float] = {"now": 1000.0}
    # A fake clock for the reader's wait loop: the poll answers only after 80 simulated seconds.
    fake_time = types.SimpleNamespace(
        monotonic=lambda: clock["now"],
        sleep=lambda delay: clock.__setitem__("now", clock["now"] + delay),
    )
    monkeypatch.setattr(nace_base, "time", fake_time, raising=False)

    def late_success(request: httpx.Request) -> httpx.Response:
        if route_of(request) == "get":
            clock["now"] += 80.0
        return ndi(request)

    reader = make_reader(
        job_timeout=5, http_client=httpx.Client(transport=httpx.MockTransport(late_success))
    )
    with pytest.raises(NaceError, match="did not come back within 5s"):
        reader.load_data(URL)
    assert ndi.routes == ["parse", "get", "delete-job"]  # the late job is still cleaned up


def test_poll_retry_waits_stay_within_the_job_budget(
    make_reader: MakeReader, ndi: FakeNdi, sleeps: list[float]
) -> None:
    ndi.replies["parse"] = [(202, {}, job_body(status="queued"))]
    ndi.replies["get"] = [
        error(429, "slow down a little", headers={"retry-after-ms": "2000"}),
        error(429, "slow down a lot", headers={"retry-after": "3600"}),
    ]
    with pytest.raises(NaceRateLimitError, match="Drex returned 429"):
        make_reader(job_timeout=5).load_data(URL)
    # The Retry-After that fits the budget is slept (a second attempt is made); the
    # hour-long one is refused instead of being slept.
    assert sleeps == [2.0]
    assert ndi.routes == ["parse", "get", "get", "delete-job"]


def test_stalled_full_text_download_is_bounded_by_the_job_budget(
    make_reader: MakeReader, ndi: FakeNdi
) -> None:
    ndi.result = parse_result(
        disclosures=[
            {"code": "content_truncated", "target": "document_content", "detail": "over 1 MiB"}
        ]
    )
    stalled = threading.Event()

    def stall_download(request: httpx.Request) -> httpx.Response:
        if route_of(request) == "download":
            stalled.wait(30.0)  # a transfer that never delivers a byte
        return ndi(request)

    reader = make_reader(
        job_timeout=1.0, http_client=httpx.Client(transport=httpx.MockTransport(stall_download))
    )
    began = time.monotonic()
    with pytest.raises(NaceError, match="did not download within 1s"):
        reader.load_data(URL)
    assert time.monotonic() - began < 10.0  # bounded: the stall cannot hold load_data open
    assert "delete-job" in ndi.routes  # cleanup is not blocked by the stalled transfer
    stalled.set()  # release the leftover daemon thread; its late result is discarded


def test_download_retry_wait_never_holds_the_caller_beyond_the_job_budget(
    make_reader: MakeReader, ndi: FakeNdi
) -> None:
    ndi.result = parse_result(
        disclosures=[
            {"code": "content_truncated", "target": "document_content", "detail": "over 1 MiB"}
        ]
    )
    ndi.replies["download"] = [error(429, "slow down", headers={"retry-after": "3600"})]
    began = time.monotonic()
    with pytest.raises(NaceError, match="did not download within 1s"):
        make_reader(job_timeout=1.0).load_data(URL)
    # The caller leaves within the budget although the SDK would sleep the hour the
    # Retry-After header asks for; the sleep is left to the leftover daemon thread.
    assert time.monotonic() - began < 10.0
    assert "delete-job" in ndi.routes  # cleanup is not blocked by the pending retry


def test_download_request_gets_a_finite_read_timeout(make_reader: MakeReader, ndi: FakeNdi) -> None:
    ndi.result = parse_result(
        disclosures=[
            {"code": "content_truncated", "target": "document_content", "detail": "over 1 MiB"}
        ]
    )
    documents = make_reader().load_data(URL)
    assert documents[0].text == FULL_MARKDOWN
    download = next(request for request in ndi.requests if route_of(request) == "download")
    read = download.extensions["timeout"]["read"]
    # SDK 0.9.0 streams a download with read=None: a stalled transfer would never end, and
    # neither would its thread. The reader caps the download thread's reads, so a stall ends.
    assert read is not None
    assert read <= 95


def test_failed_job_raises_and_cleans_up(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    ndi.replies["parse"] = [
        (
            200,
            {},
            job_body(
                status="failed",
                error={
                    "code": "unreadable_file",
                    "message": "could not fetch "
                    "Https://example.test/files/report.pdf?token=made-up-token for parsing "
                    "with the key nace_sk_test",
                },
            ),
        )
    ]
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    with pytest.raises(NaceError) as raised:
        make_reader().load_data(source)
    message = str(raised.value)
    assert "NDI parse failed for report.pdf: could not fetch <url> for parsing" in message
    assert "<url>" in message
    assert "made-up-token" not in message
    assert "example.test/files" not in message
    assert "nace_sk_test" not in message  # an echoed key is redacted like a signed link
    assert "<redacted>" in message
    assert ndi.routes == ["grant", "upload", "parse", "delete-job", "delete-file"]


def test_authentication_error_leaks_nothing(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    ndi.replies["parse"] = [
        (
            401,
            {},
            {
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key provided.",
                    "param": "raw-body-marker",
                },
                "request_id": "req_err",
            },
        )
    ]
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    with pytest.raises(NaceAuthenticationError) as raised:
        make_reader(api_key="nace_sk_test_secret").load_data(source)
    assert raised.value.status == 401
    message = str(raised.value)
    assert "nace_sk_test_secret" not in message
    assert "raw-body-marker" not in message
    # The file was still cleaned up; no job exists to delete.
    assert "delete-file" in ndi.routes


def test_an_echoed_api_key_never_reaches_the_error(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    ndi.replies["parse"] = [
        (
            401,
            {},
            {
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid API key nace_sk_echoed rejected.",
                    "issues": [
                        {"path": "authorization", "message": "the key nace_sk_echoed is invalid"}
                    ],
                },
                "request_id": "req_err",
            },
        )
    ]
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    with pytest.raises(NaceAuthenticationError) as raised:
        make_reader(api_key="nace_sk_echoed").load_data(source)
    echoed = raised.value
    assert "nace_sk_echoed" not in str(echoed)
    assert "nace_sk_echoed" not in repr(echoed)
    assert "Invalid API key <redacted> rejected." in str(echoed)
    assert "authorization: the key <redacted> is invalid" in str(echoed)
    assert echoed.body == {
        "error": {
            "type": "authentication_error",
            "message": "Invalid API key <redacted> rejected.",
            "issues": [{"path": "authorization", "message": "the key <redacted> is invalid"}],
        },
        "request_id": "req_err",
    }
    assert "delete-file" in ndi.routes  # the upload is still cleaned up


def test_truncated_result_downloads_full_text_or_fails_closed(
    make_reader: MakeReader, ndi: FakeNdi
) -> None:
    ndi.result = parse_result(
        disclosures=[
            {"code": "content_truncated", "target": "document_content", "detail": "over 1 MiB"}
        ]
    )
    documents = make_reader().load_data(URL)
    assert documents[0].text == FULL_MARKDOWN
    assert "download" in ndi.routes
    ndi.requests.clear()
    ndi.result = parse_result(
        disclosures=[
            {"code": "content_truncated", "target": "spreadsheet_rows", "detail": "over 1 MiB"}
        ]
    )
    with pytest.raises(NaceError, match="too large to return complete"):
        make_reader().load_data(URL)


@pytest.mark.parametrize("result", UNREADABLE_RESULTS)
def test_unreadable_result_is_refused(make_reader: MakeReader, ndi: FakeNdi, result: Any) -> None:
    ndi.result = result
    with pytest.raises(NaceAPIResponseValidationError, match="can't read"):
        make_reader(chunking_strategy="page").load_data(URL)
    assert "delete-job" in ndi.routes  # the job is still cleaned up


@pytest.mark.parametrize(("reader_kwargs", "file", "message"), BAD_INPUTS)
def test_invalid_input_is_refused_before_any_request(
    make_reader: MakeReader, ndi: FakeNdi, reader_kwargs: dict[str, Any], file: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_reader(**reader_kwargs).load_data(file)
    assert ndi.routes == []


def test_cleanup_failure_warns_and_still_returns_documents(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ndi.replies["delete-job"] = [error(404, "job already gone")]
    ndi.replies["delete-file"] = [error(404, "file already gone")]
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    with caplog.at_level("WARNING"):
        documents = make_reader().load_data(source)
    assert [document.text for document in documents] == [MARKDOWN]
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any(JOB_ID in text and "NaceNotFoundError" in text for text in warnings)
    assert any(FILE_ID in text and "NaceNotFoundError" in text for text in warnings)
    # A deletion the server is still running warns the same way and changes nothing else.
    ndi.requests.clear()
    caplog.clear()
    ndi.replies["delete-file"] = [
        (
            200,
            {},
            {
                "object": "file_deletion",
                "workspace_id": WORKSPACE_ID,
                "file_id": FILE_ID,
                "status": "deleting",
            },
        )
    ]
    documents = make_reader().load_data(source)
    assert [document.text for document in documents] == [MARKDOWN]
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert f"deletion of NDI file {FILE_ID} is still running" in warnings


def test_cleanup_does_not_honor_long_retry_after(
    make_reader: MakeReader,
    ndi: FakeNdi,
    tmp_path: Path,
    sleeps: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    throttled = [error(429, "slow down", headers={"retry-after": "3600"})] * 3
    ndi.replies["delete-job"] = throttled
    ndi.replies["delete-file"] = throttled
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    with caplog.at_level("WARNING"):
        documents = make_reader().load_data(source)
    # The documents still come back, and no hour-long cleanup sleep is ever scheduled.
    assert [document.text for document in documents] == [MARKDOWN]
    assert sleeps == []
    delete_job = next(request for request in ndi.requests if route_of(request) == "delete-job")
    assert delete_job.extensions["timeout"]["read"] <= 10  # a stalled deletion ends with the budget
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any(JOB_ID in text and "NaceRateLimitError" in text for text in warnings)
    assert any(FILE_ID in text and "NaceRateLimitError" in text for text in warnings)


def test_cleanup_budget_spent_skips_the_rest_with_a_warning(
    make_reader: MakeReader,
    ndi: FakeNdi,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nace_base, "CLEANUP_SECONDS", 0.2, raising=False)

    def slow_delete_job(request: httpx.Request) -> httpx.Response:
        if route_of(request) == "delete-job":
            time.sleep(0.3)  # longer than the whole cleanup budget
        return ndi(request)

    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    reader = make_reader(http_client=httpx.Client(transport=httpx.MockTransport(slow_delete_job)))
    with caplog.at_level("WARNING"):
        documents = reader.load_data(source)
    assert [document.text for document in documents] == [MARKDOWN]
    # The job deletion used up the budget, so the file deletion is skipped with a warning
    # instead of waiting past it.
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any(FILE_ID in text and "NaceError" in text for text in warnings)
    assert not any(JOB_ID in text for text in warnings)  # the job deletion itself succeeded


def test_job_deletion_crash_still_deletes_the_file(
    make_reader: MakeReader,
    ndi: FakeNdi,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    reader = make_reader()

    def crash(job_id: object, **kwargs: object) -> None:
        raise RuntimeError("made-up crash")

    monkeypatch.setattr(reader._sdk.jobs, "delete", crash)
    with caplog.at_level("WARNING"):
        documents = reader.load_data(source)
    assert [document.text for document in documents] == [MARKDOWN]
    assert "delete-file" in ndi.routes
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert any(JOB_ID in text and "RuntimeError" in text for text in warnings)


def test_fs_bytes_are_uploaded_with_the_file_name(make_reader: MakeReader, ndi: FakeNdi) -> None:
    fs = fsspec.filesystem("memory")
    with fs.open("/reports/report.pdf", "wb") as handle:
        handle.write(b"made-up pdf bytes")
    documents = make_reader().load_data("/reports/report.pdf", fs=fs)
    assert [document.text for document in documents] == [MARKDOWN]
    assert documents[0].metadata["file_name"] == "report.pdf"
    assert ndi.routes == ["grant", "upload", "parse", "delete-job", "delete-file"]
    assert re.fullmatch(UPLOAD_PATH, ndi.bodies[0]["path"])
    assert b"made-up pdf bytes" in ndi.requests[1].content


def test_simple_directory_reader_delegates_to_nace(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    documents = SimpleDirectoryReader(
        input_files=[source], file_extractor={".pdf": make_reader()}
    ).load_data()
    assert [document.text for document in documents] == [MARKDOWN]
    assert ndi.routes == ["grant", "upload", "parse", "delete-job", "delete-file"]


async def test_aload_data_returns_the_same_documents(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    reader = make_reader()
    expected = [document.text for document in reader.load_data(source)]
    ndi.requests.clear()
    documents = await reader.aload_data([source, URL])
    assert [document.text for document in documents[:1]] == expected
    assert len(documents) == 2
    assert ndi.routes == [
        "grant",
        "upload",
        "parse",
        "delete-job",
        "delete-file",
        "parse",
        "delete-job",
    ]
