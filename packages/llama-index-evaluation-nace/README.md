<p align="center">
  <a href="https://nace.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://console.nace.ai/brand/nace-logo-white.svg"><img src="https://console.nace.ai/brand/nace-logo-black.svg" alt="Nace" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.llamaindex.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/run-llama/.github/1dbae62d04e42b27e15972bf295078c43aa56c63/logos/llamaindex-light.png"><img src="https://raw.githubusercontent.com/run-llama/.github/1dbae62d04e42b27e15972bf295078c43aa56c63/logos/llamaindex-dark.png" alt="LlamaIndex" height="40"></picture></a>
</p>

<h1 align="center">Drex for LlamaIndex</h1>

<p align="center"><code>llama-index-evaluation-nace</code></p>

Replace your LLM judge. LlamaIndex evaluators backed by Drex decision models from [Nace](https://nace.ai): Drex returns probabilities, not text that has to be parsed.

## Install

```
pip install llama-index-evaluation-nace
```

Set `NACE_API_KEY`. Optional: `NACE_BASE_URL`, `NACE_DEFAULT_DECISION_MODEL`. For OpenRouter, pass `base_url="https://openrouter.ai/api/v1"` and set `OPENROUTER_API_KEY`. See *Backends and credentials* in the [repository README](https://github.com/nace-ai/llama-index-nace#backends-and-credentials).

The default model is `drex-max-v1.1`. An explicit `model=` is sent exactly as given; without one, `NACE_DEFAULT_DECISION_MODEL` is used when set, else `drex-max-v1.1`. If that default is a text-only model (`drex-v1.5`, `drex-v2.0`) and the request has images, `drex-max-v1.1` is used for that request. OpenRouter doesn't list v1.1 yet, so there the default is the original Drex Max, `nace-ai/drex-max`.

## Usage

```python
from llama_index.evaluation.nace import NaceFaithfulnessEvaluator

result = NaceFaithfulnessEvaluator().evaluate(
    query="When was the Eiffel Tower completed?",
    response="It was completed in 1889.",
    contexts=["The Eiffel Tower was completed in March 1889."],
)
print(result.passing, result.score, result.feedback)  # True 0.95 P(faithful)=0.95
```

`evaluate()` is sync and works in Jupyter; `aevaluate()` is async. `evaluate_response()` reads source nodes, so the evaluators work with `BatchEvalRunner`.

| Evaluator | Question | score / passing |
|---|---|---|
| `NaceFaithfulnessEvaluator` | Is every claim supported by the context? (strict) | P(yes); `passing = P >= threshold` (0.5) |
| `NaceRelevancyEvaluator` | Does the response answer the query, in line with the context? | P(yes); same threshold |
| `NaceCorrectnessEvaluator` | 5-level score, against `reference=` when given | 1 to 5; `passing = score >= score_threshold` (4.0) |

`raise_error=True` (faithfulness, relevancy) raises `ValueError` on a failing result, as LlamaIndex's evaluators do. The question text is editable through `update_prompts({"instructions": ...})`.

Results are `NaceEvaluationResult`: `probability`, `probabilities` (score levels), `model`, `request_id`, `usage`.

## Images

Evaluators read `ImageNode` sources (`image_path`, base64 `image`, or a `data:` URL) and send them to Drex. The default model, `drex-max-v1.1`, reads images; when `NACE_DEFAULT_DECISION_MODEL` names a text-only model, requests with images still go to `drex-max-v1.1`, and an explicit text-only `model=` with images raises `NaceImageModelError`. `http(s)` image URLs are refused, because Drex never fetches URLs. `max_images=N` keeps the N highest-scored image nodes, whatever their count; without it, more than 8 images (Drex's per-request limit) raise `NaceImageLimitError` rather than being trimmed silently. Strict faithfulness on image-only sources tends to score lower than on text; compare probabilities rather than trusting the default cut-off.

See [`examples/`](https://github.com/nace-ai/llama-index-nace/tree/main/examples) for runnable scripts and notebooks.
