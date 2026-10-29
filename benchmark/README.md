# Benchmark: Drex vs LLM judges and selectors

Cost, latency and accuracy of the Drex evaluators and selectors against LlamaIndex's own
`FaithfulnessEvaluator` and `LLMSingleSelector` running on a strong and a cheap LLM. Every number
comes from `benchmark/results/*.json`; the per-example records (id, label, prediction, score,
tokens, latency, feedback snippet, error) are in the `.jsonl` next to each one.

`drex-max-v1.1` is the packages' default model, so its rows are the headline. `drex`
(`drex-v2.0` pinned explicitly) is kept for comparison.

## Results

### Faithfulness, length-matched (headline)

HaluEval QA items whose right and hallucinated answers differ by at most one word (both at least
two words), 250 items, 500 examples (250 faithful, 250 unfaithful; accuracy = balanced accuracy).

| System | Model | n | Accuracy | ROC-AUC | $ / 1k | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `drex-max-v1.1` (default) | drex-max-v1.1 | 500 | 93.8% | 0.994 | $0.0076 | 559 | 782 |
| `drex` | drex-v2.0 | 500 | 89.0% | 0.951 | $0.0115 | 481 | 751 |
| `llm-strong` | claude-sonnet-5-5 | 500 | 90.0% | n/a | $1.29 | 1256 | 2468 |
| `llm-cheap` | gpt-5.4-nano | 500 | 76.6% | n/a | $0.0712 | 860 | 1279 |
| `drex-max-v1.1-noquery` (reference) | drex-max-v1.1 | 500 | 89.8% | 0.952 | $0.0067 | 550 | 802 |
| `drex-noquery` (reference) | drex-v2.0 | 500 | 89.2% | 0.914 | $0.0097 | 479 | 828 |
| `length-rule` (reference) | none | 500 | 54.6% | n/a | $0 | n/a | n/a |

### Faithfulness, random HaluEval sample

250 items drawn at random, 500 examples. This set has an answer-length shortcut (below).

| System | Model | n | Accuracy | ROC-AUC | $ / 1k | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `drex-max-v1.1` (default) | drex-max-v1.1 | 500 | 86.6% | 0.987 | $0.0075 | 552 | 823 |
| `drex` | drex-v2.0 | 500 | 82.0% | 0.922 | $0.0113 | 480 | 837 |
| `llm-strong` | claude-sonnet-5-5 | 500 | 85.2% | n/a | $1.42 | 1384 | 7921 |
| `llm-cheap` | gpt-5.4-nano | 500 | 67.8% | n/a | $0.0709 | 883 | 1363 |
| `drex-noquery` (reference) | drex-v2.0 | 500 | 83.4% | 0.808 | $0.0096 | 475 | 852 |
| `length-rule` (reference) | none | 500 | 93.6% | n/a | $0 | n/a | n/a |

### Routing

320 queries, four tools (below). Accuracy only; balanced accuracy and AUC are not defined here.

| System | Model | n | Accuracy | $ / 1k | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| `drex-max-v1.1` (default) | drex-max-v1.1 | 320 | 98.1% | $0.0064 | 495 | 755 |
| `drex` | drex-v2.0 | 320 | 97.2% | $0.0090 | 480 | 949 |
| `llm-strong` | claude-sonnet-5-5 | 320 | 97.8% | $1.69 | 1672 | 2784 |
| `llm-cheap` | gpt-5.4-nano | 320 | 95.9% | $0.1064 | 1214 | 1657 |

No request failed or returned an unusable answer (`n_errors` is 0 in every JSON); a failure would
count as a wrong answer. LLM judges give a YES/NO verdict, so AUC is n/a for them.

### Cost and latency

Cost ratios (measured $/1k):

| Task | Sonnet 5.5 vs `drex-max-v1.1` | gpt-5.4-nano vs `drex-max-v1.1` | `drex-v2.0` vs `drex-max-v1.1` |
|---|---:|---:|---:|
| Faithfulness, matched | 169x | 9.4x | 1.5x |
| Faithfulness, random | 189x | 9.4x | 1.5x |
| Routing | 264x | 16.6x | 1.4x |

Median latency: Sonnet 5.5 is 2.2x to 3.4x slower than `drex-max-v1.1`, gpt-5.4-nano 1.5x to
2.5x. `drex-max-v1.1` costs $0.03 per 1M input tokens and `drex-v2.0` $0.06, so `drex-v2.0` is
both the more expensive and the less accurate Drex model here.

### Significance

The 95% interval on 500 examples is about plus or minus 2.6 points at 90% accuracy (3.3 at 82%),
and on 320 at 97% about plus or minus 1.9. Every system saw the same examples, so differences are
also checked paired, example by example (bootstrap 95% interval of the difference, and McNemar's
exact test):

- Matched faithfulness: our `drex-max-v1.1` evaluator scores 3.8 points above LlamaIndex's
  evaluator on Sonnet 5.5 (93.8% vs 90.0%; paired interval +2.0 to +5.6; 20 examples only
  `drex-max-v1.1` got right, 1 only Sonnet; p < 0.0001). **That gap comes from the question:**
  our evaluator sends it and LlamaIndex's does not (see the caveats). Given the same inputs,
  `drex-max-v1.1-noquery` and Sonnet 5.5 are not separable (89.8% vs 90.0%; paired interval -1.6
  to +1.0). So the fair reading is "on par with Sonnet 5.5 on the same inputs, and 93.8% as the
  package is used".
- Random-sample faithfulness: `drex-max-v1.1` and Sonnet 5.5 are not separable (86.6% vs 85.2%;
  paired interval -1.0 to +3.8).
- Routing: not separable (98.1% vs 97.8%; 1 example differs, in `drex-max-v1.1`'s favour).
- Against gpt-5.4-nano, `drex-max-v1.1` is ahead by 17.2 points on matched faithfulness (paired
  interval +13.6 to +21.0) and 18.8 on the random sample; on routing it is 2.2 points ahead
  (+0.0 to +4.4, p = 0.09), at the edge of sampling error.
- `drex-max-v1.1` against `drex-v2.0`: +4.8 points on matched faithfulness (+2.8 to +6.8) and
  +4.6 on the random sample, at two thirds of the cost; routing is not separable. `drex-v2.0`
  and Sonnet 5.5 are not separable on any task.

## The HaluEval length shortcut

In HaluEval QA the right answer is short and the hallucinated one is long: median 2 words against
9 in the 250-item random sample (the full file shows the same pattern). The rule "at most 4 words
means faithful" needs no model and no passage, and it scores 93.6% on the random sample (about
94.5% on the full file), more than any system here. The random-sample table therefore partly
measures answer length. The matched set keeps only items whose two answers differ by at most one
word; there the length rule drops to 54.6%, near chance. **The matched set is the faithfulness
result to quote.** The random-sample table is kept for reference.

## Systems

| System | Faithfulness | Routing |
|---|---|---|
| `drex-max-v1.1` | `NaceFaithfulnessEvaluator(model="drex-max-v1.1")`, query included (the default model) | `NaceSingleSelector(model="drex-max-v1.1")` |
| `drex` | `NaceFaithfulnessEvaluator(model="drex-v2.0")`, query included | `NaceSingleSelector(model="drex-v2.0")` |
| `llm-strong` | LlamaIndex `FaithfulnessEvaluator` + `Anthropic("claude-sonnet-5-5")` | LlamaIndex `LLMSingleSelector` + the same LLM |
| `llm-cheap` | LlamaIndex `FaithfulnessEvaluator` + `OpenAI("gpt-5.4-nano")` | LlamaIndex `LLMSingleSelector` + the same LLM |
| `drex-max-v1.1-noquery` | same evaluator on `drex-max-v1.1`, called without the query (matched set only) | n/a |
| `drex-noquery` | same evaluator on `drex-v2.0`, called without the query | n/a |
| `length-rule` | response has at most 4 words, so faithful; no API calls | n/a |

Baselines use LlamaIndex's stock prompts, unmodified, and default sampling. The LLM integrations
(`llama-index-llms-anthropic` 0.12.3, `llama-index-llms-openai` 0.8.3) are in the `benchmark`
dependency group only, not in either package.

## Datasets

- **Faithfulness**: [HaluEval](https://github.com/RUCAIBox/HaluEval) QA (`qa_data.json`), MIT licence.
  Each item gives two examples: the right answer (label: faithful) and the hallucinated answer
  (label: unfaithful), judged against the item's `knowledge` passage. Fixed seed 20261028.
  The random set draws 250 of the 10,000 items. The matched set draws 250 items from those whose
  two answers differ by at most one word and are both at least two words long.
- **Routing**: 320 queries, 80 from each of four public datasets, with one tool per dataset's
  domain (descriptions in `tasks/routing.py`). The tool order is shuffled per query.
  - `math_word_problems`: GSM8K test questions, MIT.
  - `python_programming`: MBPP task descriptions, CC BY 4.0.
  - `biomedical_literature`: PubMedQA (labelled set) questions, MIT.
  - `wikipedia_knowledge`: SQuAD v1.1 dev questions, CC BY-SA 4.0.

Raw datasets are downloaded to `benchmark/.cache` (git-ignored) and are not committed. The
per-example `.jsonl` files hold only ids, labels, predictions and short feedback snippets.

## Prices (USD per 1M tokens)

Cost = tokens reported in each API response x list price. Nothing was cached, so no cached-input
discount applies. Prices are in `pricing.py`.

| Model | Input | Output | Source | Checked |
|---|---:|---:|---|---|
| drex-max-v1.1 | $0.03 | free | https://console.nace.ai/docs/drex/reference/pricing | 2026-10-29 |
| drex-v2.0 | $0.06 | free | https://console.nace.ai/docs/drex/reference/pricing | 2026-10-28 |
| claude-sonnet-5-5 | $2.00 | $10.00 | https://www.anthropic.com/pricing | 2026-10-28 |
| gpt-5.4-nano | $0.20 | $1.25 | https://developers.openai.com/api/docs/pricing (standard tier) | 2026-10-28 |

Output tokens include thinking (Anthropic) and reasoning (OpenAI) tokens, as the APIs report them.
Drex bills input tokens only, so its reported output tokens are not priced.

## Caveats

- **Query asymmetry.** LlamaIndex's `FaithfulnessEvaluator` never sends the question to the LLM; it
  judges the bare answer against the passage. Our Drex evaluator includes it. The `-noquery`
  systems remove that difference. On `drex-max-v1.1` the question moves accuracy: matched 93.8%
  with it, 89.8% without (paired interval of the difference +2.2 to +5.8), and ROC-AUC 0.994 vs
  0.952. On `drex-v2.0` the accuracy is the same within noise with or without it (matched 89.0%
  vs 89.2%, random 82.0% vs 83.4%), but ROC-AUC is lower without it (0.914 vs 0.951 matched, 0.808
  vs 0.922 random).
- **One run per cell, no tuning.** Differences of about a point between runs are noise. LlamaIndex's
  prompts are not optimised for these models.
- **Latency** is wall time per decision from one machine, through the provider SDKs including their
  retries. Drex ran at concurrency 16 (its in-flight limit) and the LLMs at 8. The LLM cells ran
  concurrently with each other; the `drex-max-v1.1` cells ran one after another, so queueing is not
  identical. Sonnet 5.5's p95 varies a lot between runs (provider retries or thinking).
- **Routing is easy** (four clearly different domains), so every system is near the ceiling; it
  shows cost and latency, not a hard routing problem.
- **HaluEval** answers are short and the passages are short; it is not a long-document RAG test.
  Its hallucinated answers were generated by ChatGPT.
- LlamaIndex's stock `FaithfulnessEvaluator` passes any reply that contains "yes" (a substring
  check), so a verdict like "not yes" would pass. No example in the matched set was affected.
- LlamaIndex's refine step turns validation failures into the text "Empty Response", which the
  faithfulness evaluator reads as "unfaithful". `run.py` counts such a decision (no recorded LLM
  call, zero tokens, or that feedback) as an error instead.
- `llama-index-llms-anthropic` 0.12.3 calls `MessageParam(...)`, which `anthropic` 2.3.0 exports as
  a Union that cannot be instantiated, so `run.py` sets that name to `dict` before building the LLM.
  Thinking was left at the model default.
- Token counts for the LLM baselines come from LLM-call events inside LlamaIndex
  (`LLMChatEndEvent`), summed per decision. gpt-5.4-nano is listed as Deprecated by OpenAI.

## Reproduce

```bash
export NACE_API_KEY=...        # Drex
export ANTHROPIC_API_KEY=...   # llm-strong
export OPENAI_API_KEY=...      # llm-cheap
uv run --group benchmark python -m benchmark.run --task faithfulness-matched --system drex-max-v1.1 --concurrency 16
uv run --group benchmark python -m benchmark.run --task routing --system llm-strong --concurrency 8
uv run --group benchmark pytest benchmark   # metric tests
```

Tasks: `faithfulness`, `faithfulness-matched`, `routing`. Systems: `drex-max-v1.1` (the default
model), `drex` (pinned to `drex-v2.0`), `llm-strong`, `llm-cheap`, plus `drex-max-v1.1-noquery`,
`drex-noquery` and `length-rule` on the faithfulness tasks. Results are written as
`<task>-<system>.json`; `--limit N` runs the first N examples (written as
`<task>-<system>-limitN.json`).
