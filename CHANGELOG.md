# Changelog

All notable changes to `llama-index-selectors-nace` and `llama-index-evaluation-nace` are listed here.
Both packages are released together with the same version number.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the packages follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-10-30

### Security

- A custom `base_url=` without `api_key=` is refused instead of receiving `NACE_API_KEY`.
- OpenRouter mode (`openrouter=True`, or an `openrouter.ai` host) never uses `NACE_API_KEY`; it uses `api_key=` or `OPENROUTER_API_KEY`.
- The ambient `OPENROUTER_API_KEY` is used only for `https://openrouter.ai` on port 443.
- Internal deployment headers and the matching cookie are sent only to `https://console.nace.ai` and `https://drex.nace.ai`, and removed from requests to other hosts.
- Redirects to another origin, including port 0 and a different IPv6 zone id, are refused before the request is sent, including on a caller-supplied HTTP client.
- The redirect guard applies to this integration's requests only, so other SDK traffic on a shared HTTP client is left alone.
- Base URLs are validated: http(s) only, a host, no credentials, query or fragment, a valid port other than 0.
- Hosts are checked as httpx will connect to them, so Unicode dots or other IDNA forms can't disguise an OpenRouter or Nace host.
- Response-controlled keys and the SDK's validation text no longer appear in error messages.
- These protections hold on every supported `nace-sdk` version (0.3.0 and later).

### Fixed

- Out-of-range or non-finite probabilities and scores raise `NaceAPIResponseValidationError`; a `NaN` could previously pass `min_confidence`.
- `default_index` is validated against the tools; an out-of-range value crashed `RouterQueryEngine`.
- Evaluator `threshold` and `score_threshold` are validated.
- A missing or wrong-type answer raises `NaceAPIResponseValidationError` instead of a bare `NaceError`.
- Error messages are capped at 500 characters, keeping the billing link and request id.
- An OpenRouter 402 points to OpenRouter credits instead of Nace billing.
- A string in `image_urls=` or `ImageNode.image_url` never reads a local file; use `image_paths` for files.

### Added

- Every Drex error class, including `NaceRedirectError`, is exported from both packages, and a `client=` from the other package is refused.
- `close()` and `aclose()` on selectors and evaluators release the HTTP connections they opened.

### Changed

- `openrouter.ai` is matched exactly; use `openrouter=True` for other OpenRouter-format hosts.
- The legacy `DREX_BASE_URL` is resolved by the integration, so OpenRouter detection and the key rules apply to it.
- CI tests both packages on the lowest and newest supported dependencies, and the release workflow runs the tests before publishing.

## [0.1.1] - 2026-10-29

### Added

- PyPI install instructions in both package READMEs.

## [0.1.0] - 2026-10-29

First release.

[Unreleased]: https://github.com/nace-ai/llama-index-nace/compare/selectors-v0.1.2...HEAD
[0.1.2]: https://github.com/nace-ai/llama-index-nace/compare/selectors-v0.1.1...selectors-v0.1.2
[0.1.1]: https://github.com/nace-ai/llama-index-nace/compare/selectors-v0.1.0...selectors-v0.1.1
[0.1.0]: https://github.com/nace-ai/llama-index-nace/releases/tag/selectors-v0.1.0
