# llama-index-nace

[![CI](https://img.shields.io/github/actions/workflow/status/nace-ai/llama-index-nace/ci.yml?branch=main&label=CI&logo=github)](https://github.com/nace-ai/llama-index-nace/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/nace-ai/llama-index-nace/blob/main/LICENSE)

LlamaIndex integrations for [Nace](https://nace.ai)'s Drex decision models. Drex scores the options and returns calibrated probabilities instead of writing text that gets parsed.

| Package | Provides |
|---|---|
| [`llama-index-evaluation-nace`](packages/llama-index-evaluation-nace) | `NaceFaithfulnessEvaluator`, `NaceRelevancyEvaluator`, `NaceCorrectnessEvaluator`: replace your LLM judge |
| [`llama-index-selectors-nace`](packages/llama-index-selectors-nace) | `NaceSingleSelector`, `NaceMultiSelector` for `RouterQueryEngine` and Workflows |

Evaluators: Drex gives each judgement as a probability, and Drex Max reads images (charts, screenshots) from `ImageNode` sources. Selectors: every answer is one of your options with a probability, so low confidence can fall back or go to a human.

## Install

```
pip install llama-index-evaluation-nace llama-index-selectors-nace
```

Each package installs on its own too.

Set `NACE_API_KEY`. The default model is `drex-max-v1.1` (text and images; `nace-ai/drex-max` on OpenRouter, which doesn't list v1.1 yet); pass `model=` or set `NACE_DEFAULT_DECISION_MODEL` to use another, such as `drex-v2.0`.

## Quick start

```python
from llama_index.evaluation.nace import NaceFaithfulnessEvaluator
from llama_index.selectors.nace import NaceSingleSelector
```

See [`examples/`](examples) for scripts and notebooks (`make examples`).

## Benchmark

Measured against LlamaIndex's own `FaithfulnessEvaluator` and `LLMSingleSelector`. Faithfulness uses 500 length-matched HaluEval QA examples
(HaluEval's right answers are much shorter than its hallucinated ones, so an "at most 4 words"
rule alone scores 93.6% on a random sample and 54.6% on the matched set). Routing uses 320 public
queries over four domains. Method, prices and caveats are in
[`benchmark/README.md`](benchmark/README.md).

| Task | System | Accuracy | $ per 1k decisions | p50 latency |
|---|---|---:|---:|---:|
| Faithfulness (matched) | Drex (drex-max-v1.1, default) | 93.8% (ROC-AUC 0.994) | $0.0076 | 559 ms |
| Faithfulness (matched) | Drex (drex-max-v1.1), question withheld | 89.8% (ROC-AUC 0.952) | $0.0067 | 550 ms |
| Faithfulness (matched) | Drex (drex-v2.0) | 89.0% (ROC-AUC 0.951) | $0.0115 | 481 ms |
| Faithfulness (matched) | Claude Sonnet 5.5 | 90.0% | $1.29 | 1256 ms |
| Faithfulness (matched) | gpt-5.4-nano | 76.6% | $0.0712 | 860 ms |
| Routing | Drex (drex-max-v1.1, default) | 98.1% | $0.0064 | 495 ms |
| Routing | Drex (drex-v2.0) | 97.2% | $0.0090 | 480 ms |
| Routing | Claude Sonnet 5.5 | 97.8% | $1.69 | 1672 ms |
| Routing | gpt-5.4-nano | 95.9% | $0.1064 | 1214 ms |

With the default model, faithfulness checks score 93.8% against 90.0% for LlamaIndex's
evaluator on Sonnet 5.5, at about 1/169 the cost and 2.2x faster. That gap comes from our
evaluator also reading the question, which LlamaIndex's does not send: given the same inputs,
drex-max-v1.1 is on par with Sonnet 5.5 (89.8% vs 90.0%). On routing, drex-max-v1.1 and Sonnet
5.5 are within sampling error (98.1% vs 97.8%), at about 1/264 the cost. Against gpt-5.4-nano,
drex-max-v1.1 is 17.2 points more accurate on matched faithfulness at about 1/9 the cost.
drex-v2.0 is less accurate and, at $0.06 against $0.03 per 1M tokens, about 1.5x the cost.
Latency is wall time from one machine, one run per cell (see the benchmark caveats).

## Releasing

When Drex ships a new default model, update the packages' default by hand:

1. Set `DEFAULT_MODEL` in both `_drex/client.py` copies
   (`packages/llama-index-selectors-nace/llama_index/selectors/nace/_drex/client.py` and
   `packages/llama-index-evaluation-nace/llama_index/evaluation/nace/_drex/client.py`).
   `scripts/check_drex_sync.py` (run in CI) fails unless the copies are identical.
2. In the same file, check the pinned facts for the new model: `TEXT_ONLY_MODELS`,
   `MAX_IMAGES`, `IMAGE_MODEL`, and `OPENROUTER_DEFAULT_MODEL` (move it once OpenRouter
   lists the new model). Question caps are left to Drex.
3. Update the unit and live tests that name the default model.
4. Add the model's list price to `benchmark/pricing.py`, add it as a system in
   `benchmark/run.py`, run it on every task, and update `benchmark/README.md` and the headline
   in this README.
5. Update the docs that name the default or quote the benchmark: both package READMEs, the
   LlamaIndex page in the Drex docs (https://console.nace.ai/docs), and re-run the notebooks in
   `examples/` so their saved outputs show the new model.
6. Bump `_version.py` in both packages: each ships its own `_drex/` copy, so a default-model
   change needs a release of both. The tests read the client name from `__version__`, so the
   bump needs no test change.
7. Tag `selectors-vX.Y.Z` and `evaluation-vX.Y.Z`; `.github/workflows/release.yml` checks the
   tag against `_version.py`, builds and publishes to PyPI.
