"""The reader's offline behaviour against a fake NDI. Invented data only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import fsspec
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
)
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.nace import NaceReader
from llama_index.readers.nace._drex.errors import (
    NaceAPIResponseValidationError,
    NaceAuthenticationError,
    NaceError,
)

URL = "https://example.test/files/report.pdf"
SIGNED_URL = "https://user:made-up@example.test/files/report.pdf?token=made-up-token#section"
MakeReader = Callable[..., NaceReader]

BAD_INPUTS = [
    ({"parse_mode": "turbo"}, URL, "parse_mode must be one of"),
    ({"chunking_strategy": "chapter"}, URL, "chunking_strategy must be one of"),
    ({"parse_options": {"chunking": {"strategy": "page"}}}, URL, "parse_options must not contain"),
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
]


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
    assert ndi.bodies[0]["path"] == "report.pdf"
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
    documents = make_reader().load_data(URL)
    assert [document.text for document in documents] == [MARKDOWN]
    assert ndi.routes == ["parse", "get", "delete-job"]


def test_failed_job_raises_and_cleans_up(
    make_reader: MakeReader, ndi: FakeNdi, tmp_path: Path
) -> None:
    ndi.replies["parse"] = [
        (
            200,
            {},
            job_body(
                status="failed",
                error={"code": "unreadable_file", "message": "the document could not be read"},
            ),
        )
    ]
    source = tmp_path / "report.pdf"
    source.write_bytes(b"made-up pdf bytes")
    with pytest.raises(
        NaceError, match=r"NDI parse failed for report\.pdf: the document could not be read"
    ):
        make_reader().load_data(source)
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


def test_fs_bytes_are_uploaded_with_the_file_name(make_reader: MakeReader, ndi: FakeNdi) -> None:
    fs = fsspec.filesystem("memory")
    with fs.open("/reports/report.pdf", "wb") as handle:
        handle.write(b"made-up pdf bytes")
    documents = make_reader().load_data("/reports/report.pdf", fs=fs)
    assert [document.text for document in documents] == [MARKDOWN]
    assert documents[0].metadata["file_name"] == "report.pdf"
    assert ndi.routes == ["grant", "upload", "parse", "delete-job", "delete-file"]
    assert ndi.bodies[0]["path"] == "report.pdf"
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
