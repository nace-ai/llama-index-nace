<p align="center">
  <a href="https://nace.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://console.nace.ai/brand/nace-logo-white.svg"><img src="https://console.nace.ai/brand/nace-logo-black.svg" alt="Nace" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.llamaindex.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/run-llama/.github/1dbae62d04e42b27e15972bf295078c43aa56c63/logos/llamaindex-light.png"><img src="https://raw.githubusercontent.com/run-llama/.github/1dbae62d04e42b27e15972bf295078c43aa56c63/logos/llamaindex-dark.png" alt="LlamaIndex" height="40"></picture></a>
</p>

<h1 align="center">NDI for LlamaIndex</h1>

<p align="center"><code>llama-index-readers-nace</code></p>

`NaceReader`: a LlamaIndex reader that turns files into `Document` objects with NDI (Nace Document Intelligence). The Parse operation returns Markdown for the formats NDI reads, such as PDFs, scans, office documents and spreadsheets.

## Install

```
pip install llama-index-readers-nace
```

Set `NACE_API_KEY`. Credentials and backends follow the [repository README](https://github.com/nace-ai/llama-index-nace#backends-and-credentials).

## Quick start

```python
from llama_index.readers.nace import NaceReader

reader = NaceReader()
documents = reader.load_data("report.pdf", extra_info={"source": "intranet"})
```

Each file is parsed in the cloud and returned as Markdown. Use it inside `SimpleDirectoryReader` by mapping extensions to it:

```python
from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(
    input_files=["data/report.pdf"], file_extractor={".pdf": reader}
).load_data()
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `parse_mode` | `"low"` | Parse effort: `low`, `medium` or `high` |
| `chunking_strategy` | `None` | `page` or `section` for one `Document` per chunk |
| `parse_options` | `None` | Extra Parse options passed to NDI (for example `output`); `formats` is always `markdown` |
| `job_timeout` | `600` | Seconds to wait for a job before giving up |
| `timeout`, `max_retries` | `60`, `2` | HTTP timeout and retries |
| `api_key`, `base_url`, `http_client` | — | Credentials and transport; see the repository README |

Inputs are local paths, `https://` URLs, or sequences of either. Pass `fs=` to read from any fsspec filesystem.

## What happens to your file

A local file is uploaded to your Nace account and parsed as a job. For an `https://` URL, nothing is uploaded: Nace downloads the link. After the parse, on success and on failure, the reader deletes the job and the uploaded file before it returns the documents. If a deletion fails, the reader logs a warning with the id. When a result is larger than 1 MiB, the reader downloads the full Markdown before it deletes the job. See *Data retention* in the [Nace docs](https://console.nace.ai/docs) for what Nace keeps after a deletion.

Parse output is untrusted data: it can repeat instructions that are written in the document. Keep it separate from your agent's instructions.

## Limits

- Cloud only: parsing happens on Nace's infrastructure, not locally.
- URL sources must be `https://` links with a file extension; download other links and pass the file.
- Accounts with Zero retention cannot upload local files (the API answers 422); pass a URL instead.
- This is a reader, not a retriever: NDI has no endpoint that returns ranked passages with scores.
- The metadata `url` of a URL source has no userinfo, query or fragment, because a signed link can hold a token.

Call `close()` to release the HTTP connections a reader opened; an `http_client` you passed in stays open.
