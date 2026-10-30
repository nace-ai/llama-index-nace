<p align="center">
  <a href="https://nace.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://console.nace.ai/brand/nace-logo-white.svg"><img src="https://console.nace.ai/brand/nace-logo-black.svg" alt="Nace" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.llamaindex.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/run-llama/.github/1dbae62d04e42b27e15972bf295078c43aa56c63/logos/llamaindex-light.png"><img src="https://raw.githubusercontent.com/run-llama/.github/1dbae62d04e42b27e15972bf295078c43aa56c63/logos/llamaindex-dark.png" alt="LlamaIndex" height="40"></picture></a>
</p>

<h1 align="center">Drex for LlamaIndex</h1>

<p align="center"><code>llama-index-selectors-nace</code></p>

LlamaIndex selectors backed by Drex decision models from [Nace](https://nace.ai). Drex scores your options and returns probabilities instead of writing text for LlamaIndex to parse.

## Install

```
pip install llama-index-selectors-nace
```

Set `NACE_API_KEY`. Optional: `NACE_BASE_URL`, `NACE_DEFAULT_DECISION_MODEL`. For OpenRouter, pass `base_url="https://openrouter.ai/api/v1"` and set `OPENROUTER_API_KEY`. See *Backends and credentials* in the [repository README](https://github.com/nace-ai/llama-index-nace#backends-and-credentials).

The default model is `drex-max-v1.1`. An explicit `model=` is sent exactly as given; without one, `NACE_DEFAULT_DECISION_MODEL` is used when set, else `drex-max-v1.1`. If that default is a text-only model (`drex-v1.5`, `drex-v2.0`) and the request has images, `drex-max-v1.1` is used for that request. OpenRouter doesn't list v1.1 yet, so there the default is the original Drex Max, `nace-ai/drex-max`.

## Usage

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.selectors.nace import NaceSingleSelector

router = RouterQueryEngine(
    selector=NaceSingleSelector(min_confidence=0.6, default_index=0),
    query_engine_tools=tools,
    llm=llm,  # RouterQueryEngine resolves an LLM at construction, even if it only routes
)
```

Drex replaces the routing decision, not answer generation: the chosen engine still needs an LLM to write the answer, and with multi-select `RouterQueryEngine` uses an LLM to merge answers.

Selectors also work on their own, in a `Workflow` branch (see [`examples/workflow_branching.py`](https://github.com/nace-ai/llama-index-nace/blob/main/examples/workflow_branching.py)) and inside a running event loop such as Jupyter: `select()` is sync, `aselect()` is async.

## NaceSingleSelector

One `choice` question whose options are the tools, one winner. On the default model (`drex-max-v1.1`) a choice question takes at most 255 options (Drex's limit), so a single selector takes at most 255 tools. `min_confidence` rejects an unsure top choice: it returns `default_index` with `fell_back=True` if set, otherwise raises `NaceLowConfidenceError` (carrying the probabilities). With `default_index`, transient failures (5xx, rate limit, connection) also fall back; authentication and request errors always propagate. Without `min_confidence` the top option is always returned.

## NaceMultiSelector

One yes/no question per option, all in a single request. Options with `p >= threshold` (default 0.5) are returned best first, cut to `max_outputs`. The result is never empty: if nothing passes, the top option is returned with `fell_back=True`. Each option is one question, so the number of options is capped by Drex's per-request question limit. Drex enforces that limit itself and its refusal comes back as `NaceUnprocessableEntityError` with Drex's message; the package doesn't guess it, because it changes (Drex Max goes from 16 to 512 questions).

Known limitation: the default instruction asks whether each option is "needed". It is conservative on purely definitional questions (one live probe scored a docs tool 0.27 for "What does 429 mean?"), so the selector falls back to the top option. Lower `threshold`, or pass your own `instructions` (`{label}` and `{description}` are filled in).

## Results

Both return `NaceSelectorResult` with `probabilities` (by label), `labels`, `confidence`, `model`, `request_id`, `usage` and `fell_back`. Labels are tool names when all are present and unique, otherwise `option_1 ... option_n`.

## Closing

Call `close()` (or `await aclose()`) to release the HTTP connections a selector opened; a `client=` you passed in stays open.

## Images

If `QueryBundle.image_path` is set, the image is sent. The default model, `drex-max-v1.1`, reads images; when `NACE_DEFAULT_DECISION_MODEL` names a text-only model, image queries still go to `drex-max-v1.1`. An explicit text-only `model=` with an image raises `NaceImageModelError`. A missing file raises `NaceImageError` before any request.

See [`examples/`](https://github.com/nace-ai/llama-index-nace/tree/main/examples) for runnable scripts and notebooks.

## Security

Versions 0.1.0 and 0.1.1 could send `NACE_API_KEY` (or `DREX_API_KEY`) to a custom `base_url` or to OpenRouter, and could forward internal deployment headers to other hosts, including through an HTTP client that follows redirects. Upgrade to 0.1.2. See the [security note and changelog](https://github.com/nace-ai/llama-index-nace/blob/main/CHANGELOG.md) for details.
