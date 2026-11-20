"""`NaceReader`: turn files into documents with NDI (Nace Document Intelligence) Parse."""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import nace_sdk
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from nace_sdk import DocumentJob, UploadedFile, WorkspaceFileSource

from ._drex.client import DrexClient, _cause
from ._drex.errors import (
    NaceAPIResponseValidationError,
    NaceError,
    _clip,
    _redact_text,
    convert_error,
)
from ._drex.origins import is_openrouter_host, resolve_destination
from ._version import __version__

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

logger = logging.getLogger(__name__)

CLIENT_NAME = f"llama-index-readers-nace/{__version__}"

PARSE_MODES = ("low", "medium", "high")
CHUNKING_STRATEGIES = ("page", "section")
#: `parse` parameters the reader owns, and the SDK's transport arguments; `extra_body` could
#: replace `source` or `formats`, and `extra_headers` the integration marker header.
RESERVED_PARSE_OPTIONS = frozenset(
    {
        "source",
        "parse_mode",
        "chunking",
        "wait_seconds",
        "extra_body",
        "extra_headers",
        "idempotency_key",
        "retry",
        "timeout",
    }
)
OPENROUTER_REFUSAL = "NDI is not available through OpenRouter; use https://console.nace.ai."
HTTPS_ONLY = "NDI reads https links only; download the file and pass its path instead."
NO_EXTENSION = (
    "the URL's last path segment has no file extension, so NDI can't pick a parser; "
    "download the file and pass its path instead."
)
UNREADABLE_RESULT = "NDI returned a parse result this package can't read."
TOO_LARGE = (
    "the NDI result is too large to return complete (for example a large workbook); "
    "nothing was returned"
)
NO_ERROR_DETAIL = "no error detail"
DETAIL_BUDGET = 300
DOWNLOAD_NAME = "document.md"
#: SDK 0.9.0 gives a held job read (`wait_seconds` above 0) an HTTP read timeout of at least
#: this many seconds, whatever the caller passes, so a hold is only asked for while that much
#: of the job budget is left; a smaller budget polls with plain reads instead.
HELD_READ_SECONDS = 95.0
MAX_HOLD_SECONDS = 25
INITIAL_POLL_SECONDS = 1.0
MAX_POLL_SECONDS = 10.0
#: A job-file read is never given longer than this, however much budget is left: the SDK's own
#: held reads use the same floor, and a stalled answer ends here instead of after the whole
#: (default 600 s) budget.
DOWNLOAD_READ_SECONDS = 95.0
DOWNLOAD_THREAD_NAME = "llama-index-readers-nace-download"
#: The whole job-and-file cleanup gets this many seconds; a deletion that would wait longer
#: (a long `Retry-After` header) is refused and warned about instead.
CLEANUP_SECONDS = 10.0

#: Thread-local read cap for the reader's download thread, set by `NaceReader._downloaded_text`.
_download_reads = threading.local()


def _cap_download_reads(request: httpx.Request) -> None:
    """Request hook: bound the reads of this thread's job-file downloads.

    `jobs.download` streams with no read timeout (SDK 0.9.0 sends `read=None`), so a stalled
    transfer would never end and neither would its thread. The reader's download thread caps
    its own reads here — every other thread, the reader's own polls included, is untouched,
    also on a caller-supplied `http_client` — and httpcore then ends a stalled read, so the
    transfer fails, the SDK's retry count runs out, and the thread finishes instead of
    holding its connection forever.
    """
    cap = getattr(_download_reads, "cap", None)
    if cap is None:
        return
    timeouts = request.extensions.get("timeout")
    if isinstance(timeouts, dict):
        read = timeouts.get("read")
        request.extensions["timeout"] = {**timeouts, "read": min(cap, read) if read else cap}


class _NdiClient(DrexClient):
    """The shared client's SDK connection, used for the document routes."""

    @property
    def sdk(self) -> nace_sdk.NaceClient:
        return self._client


def _url_without_secrets(url: str) -> str:
    """`url` without userinfo, query and fragment: a link can carry a password or a token there."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2], query="", fragment=""))


class NaceReader(BaseReader):
    """A reader that parses files with NDI Parse and returns them as `Document` objects.

    Each input is uploaded to Nace, parsed as a job, and — success or failure — the job and
    the uploaded file are deleted before the documents are returned. `load_data`,
    `aload_data` and `alazy_load_data` come from `BaseReader`; there is no separate async
    code path.

    Args:
        api_key: A Nace API key. Otherwise the environment is read (see *Backends and
            credentials* in the repository README).
        base_url: The API origin. The default is Nace production.
        parse_mode: Parse effort: `low`, `medium` or `high`.
        chunking_strategy: `page` or `section` for one `Document` per chunk; `None` for one
            `Document` per file.
        parse_options: Extra Parse options passed to NDI (for example `output`); `formats`
            is always `["markdown"]`, and `source`, `parse_mode`, `chunking`, `wait_seconds`
            and the SDK transport arguments are refused here.
        job_timeout: Seconds to wait for a parse job that is not finished when the create
            request returns. The create itself is held open for that time in whole
            seconds (rounded down), between 1 and 60. A finite number above 0. The same
            budget bounds everything after the create: every poll's HTTP timeout (capped at
            95 s) and retry (a `Retry-After` header included), and the download of the full
            text of a truncated result. A result that only arrives after the budget is
            refused, and the job is deleted. The bound is best-effort: a request already in
            flight, an SDK retry or the cleanup can run slightly past the deadline.
        timeout: HTTP timeout in seconds; at least 60, because a parse create is held open.
        max_retries: HTTP retries per SDK call.
        http_client: An `httpx.Client` to send the requests on. It stays the caller's:
            `close()` leaves it open. Its own default headers go out with every request it
            sends, the upload to the Nace document service host included.

    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        parse_mode: Literal["low", "medium", "high"] = "low",
        chunking_strategy: Literal["page", "section"] | None = None,
        parse_options: Mapping[str, Any] | None = None,
        job_timeout: float = 600.0,
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Create the reader.

        Raises:
            ValueError: `parse_mode`, `chunking_strategy`, `job_timeout` or
                `parse_options` is invalid, or the destination is OpenRouter, where NDI
                is not available.

        """
        if parse_mode not in PARSE_MODES:
            message = f"parse_mode must be one of {PARSE_MODES} (got {parse_mode!r})."
            raise ValueError(message)
        if chunking_strategy is not None and chunking_strategy not in CHUNKING_STRATEGIES:
            message = (
                f"chunking_strategy must be one of {CHUNKING_STRATEGIES} or None "
                f"(got {chunking_strategy!r})."
            )
            raise ValueError(message)
        if not (math.isfinite(job_timeout) and job_timeout > 0):
            message = f"job_timeout must be a finite number above 0 seconds (got {job_timeout!r})."
            raise ValueError(message)
        options = dict(parse_options or {})
        reserved = RESERVED_PARSE_OPTIONS & options.keys()
        if reserved:
            message = (
                f"parse_options must not contain {', '.join(sorted(reserved))}; "
                "the reader sets them from its own arguments."
            )
            raise ValueError(message)
        destination = resolve_destination(base_url)
        if is_openrouter_host(destination.url):
            raise ValueError(OPENROUTER_REFUSAL)
        self._parse_mode = parse_mode
        self._chunking = chunking_strategy
        self._job_timeout = job_timeout
        self._max_retries = max_retries
        self._output = dict(options.pop("output", None) or {})
        self._output["formats"] = ["markdown"]
        self._options = options
        self._client = _NdiClient(
            CLIENT_NAME,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            openrouter=False,
            http_client=http_client,
        )
        # Cap the reads of the download thread with a request hook on the sync httpx client
        # (a supplied one too: the hook is idempotent and thread-local, like the `_drex`
        # guard hook the client also installs there).
        if _cap_download_reads not in self._client._sync_http.event_hooks["request"]:
            self._client._sync_http.event_hooks["request"].append(_cap_download_reads)

    @property
    def _sdk(self) -> nace_sdk.NaceClient:
        return self._client.sdk

    def close(self) -> None:
        """Close the HTTP client this reader created; a supplied `http_client` stays open."""
        self._client.close()

    def lazy_load_data(
        self,
        file: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        extra_info: dict[str, Any] | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> Iterable[Document]:
        """Parse each input in sequence and yield its documents.

        Args:
            file: A local path, an `https://` URL, or a sequence of either. A URL must end
                in a file name with an extension.
            extra_info: Merged into every document's metadata, after the reader's own keys.
            fs: Read local paths from this fsspec filesystem instead of the local one.

        Raises:
            ValueError: An input is an `http://` URL, or a URL without a file extension.
            NaceError: The job failed, timed out, or the result was too large to return.

        """
        items: Sequence[str | os.PathLike[str]] = (
            [file] if isinstance(file, str | os.PathLike) else list(file)
        )
        for item in items:
            # Not a generator expression over `_read_one`: each item's cleanup must be
            # complete before its documents are yielded.
            yield from self._read_one(item, extra_info, fs)

    def _read_one(
        self,
        file: str | os.PathLike[str],
        extra_info: dict[str, Any] | None,
        fs: AbstractFileSystem | None,
    ) -> list[Document]:
        """Parse one input; clean up the job and the uploaded file before returning."""
        uploaded: UploadedFile | None = None
        job: DocumentJob | None = None
        try:
            source: Mapping[str, Any] | WorkspaceFileSource
            url: str | None
            if isinstance(file, str) and file.lower().startswith("https://"):
                source, file_name, url = self._url_source(file)
            else:
                if isinstance(file, str) and file.lower().startswith("http://"):
                    raise ValueError(HTTPS_ONLY)
                uploaded, file_name = self._upload(file, fs)
                source, url = uploaded.as_source(), None
            job = self._sdk.documents.parse(
                source,
                # Held open for at most the wait budget (1 to 60 s), so a small
                # `job_timeout` does not hold the create for 60 s.
                wait_seconds=max(1, min(60, int(self._job_timeout))),
                parse_mode=self._parse_mode,
                chunking={"strategy": self._chunking} if self._chunking else None,
                output=self._output,
                **self._options,
            )
            # One budget for everything after the create returns: waiting for the job and,
            # for a truncated result, downloading the full text.
            deadline = time.monotonic() + self._job_timeout
            if not job.is_terminal:
                job = self._await_job(job, file_name, deadline)
            if job.status != "succeeded":
                # A server message can echo a signed link or the configured key; scrub both
                # before they reach logs.
                detail = re.sub(
                    r"https?://\S+",
                    "<url>",
                    job.error_message or NO_ERROR_DETAIL,
                    flags=re.IGNORECASE,
                )
                detail = _redact_text(detail, self._client._secrets)
                message = f"NDI parse {job.status} for {file_name}: {_clip(detail, DETAIL_BUDGET)}"
                raise NaceError(message)
            return self._documents(job, file_name, url, extra_info, deadline)
        except nace_sdk.NaceError as error:
            raise convert_error(error, secrets=self._client._secrets) from _cause(error)
        finally:
            self._cleanup(job, uploaded)

    def _await_job(self, job: DocumentJob, file_name: str, deadline: float) -> DocumentJob:
        """Read the job until it is terminal, within `deadline`.

        `jobs.wait` is not used: its held reads carry a 95-second HTTP timeout, it retries
        under the client's unbounded policy (an hour-long `Retry-After` is slept), and it
        returns a job that only finished after its own deadline. Here every read's HTTP
        timeout and retry budget is the time the budget has left, and a read that only
        finishes after the deadline is not accepted, so the wait cannot outlast `job_timeout`.
        """
        interval = INITIAL_POLL_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._job_budget_error(job, file_name)
            # A held read's HTTP timeout is at least `HELD_READ_SECONDS` whatever is passed,
            # so one is asked for only while that much budget is left; a smaller budget polls
            # with plain reads, each bounded by the time it has left.
            hold = MAX_HOLD_SECONDS if remaining >= HELD_READ_SECONDS else 0
            began = time.monotonic()
            job = self._sdk.jobs.get(
                job.job_id,
                wait_seconds=hold,
                # The SDK raises a held read's timeout to at least 95 s whatever is passed,
                # so pass exactly that; a plain read takes the time left (under 95 s). A
                # poll's read timeout is thus never the whole remaining budget (600 s by
                # default), and a stalled answer ends at 95 s at the latest.
                timeout=HELD_READ_SECONDS if hold else remaining,
                retry=nace_sdk.RetryPolicy(max_retries=self._max_retries, timeout=remaining),
            )
            if job.is_terminal:
                if time.monotonic() > deadline:
                    raise self._job_budget_error(job, file_name)
                return job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._job_budget_error(job, file_name)
            # A held read already waited out its hold server-side; only make up the rest.
            time.sleep(max(0.0, min(interval, remaining) - (time.monotonic() - began)))
            interval = min(interval * 2, MAX_POLL_SECONDS)

    def _job_budget_error(self, job: DocumentJob, file_name: str) -> NaceError:
        """The error for a job that did not come back within its budget."""
        return NaceError(
            f"NDI parse for {file_name}: job {job.job_id} did not come back within "
            f"{self._job_timeout:g}s (its last status was {job.status})"
        )

    def _url_source(self, url: str) -> tuple[dict[str, Any], str, str]:
        """The parse source, the file name and the metadata URL of an `https://` source."""
        file_name = urlsplit(url).path.rsplit("/", 1)[-1]
        if "." not in file_name[1:-1]:
            raise ValueError(NO_EXTENSION)
        source = {"type": "url", "url": url, "file_name": file_name}
        return source, file_name, _url_without_secrets(url)

    def _upload(
        self, path: str | os.PathLike[str], fs: AbstractFileSystem | None
    ) -> tuple[UploadedFile, str]:
        """Upload a local path (or its open handle, from `fs`) and return the file and its name."""
        file_name = Path(os.fspath(path)).name
        # A unique workspace path, so Nace never answers with an existing file of the same
        # name — the reader could then delete the caller's own file.
        target = f"llama-index-readers-nace/{uuid.uuid4().hex}/{file_name}"
        if fs is None:
            uploaded = self._sdk.documents.upload(path, path=target)
        else:
            with fs.open(os.fspath(path), "rb") as handle:
                uploaded = self._sdk.documents.upload(handle, file_name=file_name, path=target)
        return uploaded, file_name

    def _documents(
        self,
        job: DocumentJob,
        file_name: str,
        url: str | None,
        extra_info: dict[str, Any] | None,
        deadline: float,
    ) -> list[Document]:
        """The `Document` objects of a succeeded parse job."""
        result = job.result
        if not isinstance(result, dict):
            raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
        document = result.get("document")
        if not isinstance(document, dict):
            raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
        disclosures = result.get("disclosures")
        truncated: list[dict[str, Any]] = []
        if disclosures is not None:
            # Fail closed on an unexpected shape: reading a malformed disclosures value as
            # "not truncated" could return partial content.
            if not isinstance(disclosures, list):
                raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
            for entry in disclosures:
                if not isinstance(entry, dict):
                    raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
                if entry.get("code") == "content_truncated":
                    truncated.append(entry)
        if any(entry.get("target") != "document_content" for entry in truncated):
            raise NaceError(f"NDI parse succeeded for {file_name}, but {TOO_LARGE}.")
        chunks = document.get("chunks")
        if self._chunking:
            # Fail closed on a malformed chunks value: reading it as "no chunks" would
            # silently change the output from one document per chunk to one unlabelled
            # document with the whole text.
            if chunks is not None and not isinstance(chunks, list):
                raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
            if chunks:
                metadata = self._metadata(file_name, url, extra_info)
                return [self._chunk_document(chunk, dict(metadata)) for chunk in chunks]
        markdown = document.get("markdown")
        if not isinstance(markdown, str):
            raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
        if truncated:
            markdown = self._downloaded_text(job, document, file_name, deadline)
        return [Document(text=markdown, metadata=self._metadata(file_name, url, extra_info))]

    def _chunk_document(self, chunk: Any, metadata: dict[str, Any]) -> Document:
        """One chunk as a `Document`, with `page_label` when the chunk names an int page."""
        if not isinstance(chunk, dict) or not isinstance(chunk.get("content"), str):
            raise NaceAPIResponseValidationError(UNREADABLE_RESULT)
        location = chunk.get("location")
        if isinstance(location, dict) and isinstance(location.get("page"), int):
            metadata["page_label"] = str(location["page"])
        return Document(text=chunk["content"], metadata=metadata)

    def _downloaded_text(
        self, job: DocumentJob, document: dict[str, Any], file_name: str, deadline: float
    ) -> str:
        """The full Markdown of a truncated result, downloaded before any cleanup.

        The SDK streams a download with no read timeout, so it runs on a daemon thread whose
        reads are capped (see `_cap_download_reads`): a stalled transfer ends on its own
        instead of lingering forever, and the caller is given only the time the job budget
        has left. A transfer that outlives the budget is left to finish or fail on its own;
        the connection it holds is released when it ends.
        """
        content_url = document.get("content_url")
        if not isinstance(content_url, str):
            raise NaceError(f"NDI parse succeeded for {file_name}, but {TOO_LARGE}.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._download_budget_error(job, file_name)
        outcome: dict[str, Any] = {}

        def download() -> None:
            # Bound this thread's reads (the request hook above), so a stall ends it.
            _download_reads.cap = min(remaining, DOWNLOAD_READ_SECONDS)
            try:
                with tempfile.TemporaryDirectory() as folder:
                    target = Path(folder) / DOWNLOAD_NAME
                    self._sdk.jobs.download(job.job_id, content_url, target, wait_timeout=remaining)
                    outcome["text"] = target.read_bytes().decode("utf-8")
            except Exception as error:  # the thread must never take the process down
                outcome["error"] = error
            finally:
                _download_reads.cap = None

        thread = threading.Thread(target=download, daemon=True, name=DOWNLOAD_THREAD_NAME)
        thread.start()
        thread.join(remaining)
        if thread.is_alive():
            raise self._download_budget_error(job, file_name)
        if "error" in outcome:
            raise outcome["error"]
        return str(outcome["text"])

    def _download_budget_error(self, job: DocumentJob, file_name: str) -> NaceError:
        """The error for a full-text download that did not finish within the job budget."""
        return NaceError(
            f"NDI parse for {file_name}: the full text of job {job.job_id} "
            f"did not download within {self._job_timeout:g}s"
        )

    def _metadata(
        self, file_name: str, url: str | None, extra_info: dict[str, Any] | None
    ) -> dict[str, Any]:
        """The metadata of a document: file name, the stripped URL, then `extra_info`."""
        metadata: dict[str, Any] = {"file_name": file_name}
        if url is not None:
            metadata["url"] = url
        if extra_info:
            metadata.update(extra_info)
        return metadata

    def _cleanup(self, job: DocumentJob | None, uploaded: UploadedFile | None) -> None:
        """Delete the job and the uploaded file within a short fixed budget.

        A failed deletion is a warning, never raised. Each deletion's HTTP timeout and retry
        budget — a long `Retry-After` header included — is the time that budget has left, so
        cleanup never waits long however the server answers; `delete_file` is held up to 25 s
        longer by the SDK, which waits for the erasure to be confirmed. Each deletion is
        guarded on its own, so a job deletion that raises anything (an SDK error, a crash in
        the SDK's plumbing) neither skips the file deletion nor replaces the first error.
        """
        deadline = time.monotonic() + CLEANUP_SECONDS
        if job is not None:
            try:
                self._sdk.jobs.delete(
                    job.job_id,
                    timeout=self._cleanup_left(deadline),
                    retry=self._cleanup_retry(deadline),
                )
            except Exception as error:
                logger.warning("could not delete NDI job %s: %s", job.job_id, type(error).__name__)
        if uploaded is not None:
            try:
                deletion = self._sdk.documents.delete_file(
                    uploaded.workspace_id,
                    uploaded.file_id,
                    timeout=self._cleanup_left(deadline),
                    retry=self._cleanup_retry(deadline),
                )
            except Exception as error:
                logger.warning(
                    "could not delete NDI file %s: %s", uploaded.file_id, type(error).__name__
                )
            else:
                if deletion.status != "deleted":
                    logger.warning("deletion of NDI file %s is still running", uploaded.file_id)

    def _cleanup_left(self, deadline: float) -> float:
        """The cleanup budget's time left; a spent budget refuses the deletion (so: a warning)."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NaceError(f"the {CLEANUP_SECONDS:g}s cleanup budget is spent")
        return remaining

    def _cleanup_retry(self, deadline: float) -> nace_sdk.RetryPolicy:
        """A retry policy bounded by the cleanup budget, so a long `Retry-After` is refused."""
        return nace_sdk.RetryPolicy(
            max_retries=self._max_retries, timeout=self._cleanup_left(deadline)
        )
