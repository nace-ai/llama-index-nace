"""Run one benchmark cell: `python -m benchmark.run --task faithfulness --system drex`."""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
)

from . import metrics
from .pricing import PRICES, Price
from .tasks import faithfulness, routing

RESULTS = Path(__file__).resolve().parent / "results"
# `drex` and `drex-noquery` are pinned to drex-v2.0; `drex-max-v1.1` is the package default.
# File names follow the system, so they never collide.
MODELS = {
    "drex": "drex-v2.0",
    "drex-noquery": "drex-v2.0",
    "drex-max-v1.1": "drex-max-v1.1",
    "drex-max-v1.1-noquery": "drex-max-v1.1",
    "llm-strong": "claude-sonnet-5-5",
    "llm-cheap": "gpt-5.4-nano",
    "length-rule": "none (no API calls)",
}
FAITHFULNESS_TASKS = ("faithfulness", "faithfulness-matched")
DREX_SYSTEMS = (
    "drex",
    "drex-noquery",
    "drex-max-v1.1",
    "drex-max-v1.1-noquery",
)
NOQUERY_SYSTEMS = ("drex-noquery", "drex-max-v1.1-noquery")
FAITHFULNESS_ONLY = (*NOQUERY_SYSTEMS, "length-rule")
LENGTH_RULE_MAX_WORDS = 4  # "at most 4 words => faithful"
FREE = Price(0.0, 0.0, "no API calls", "2026-10-28")
MAX_CONCURRENCY = {
    "drex": 16,
    "drex-noquery": 16,
    "drex-max-v1.1": 16,
    "drex-max-v1.1-noquery": 16,
    "llm-strong": 8,
    "llm-cheap": 8,
    "length-rule": 16,
}

_usage: contextvars.ContextVar[list[tuple[int, int]] | None] = contextvars.ContextVar(
    "usage", default=None
)


def _field(obj: Any, name: str) -> int:
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return int(value or 0)


def tokens_from_raw(raw: Any) -> tuple[int, int]:
    """(input, output) tokens from an Anthropic or OpenAI raw response.

    Anthropic's `output_tokens` includes thinking; OpenAI's `completion_tokens` includes
    reasoning. Anthropic cache tokens are billed apart from `input_tokens`, but nothing here
    is cached, so they are reported as zero and checked in `usage` below.
    """
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        has_prompt = "prompt_tokens" in usage
    else:
        has_prompt = hasattr(usage, "prompt_tokens")
    if has_prompt:
        return _field(usage, "prompt_tokens"), _field(usage, "completion_tokens")
    inp = (
        _field(usage, "input_tokens")
        + _field(usage, "cache_creation_input_tokens")
        + _field(usage, "cache_read_input_tokens")
    )
    return inp, _field(usage, "output_tokens")


class _UsageHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "UsageHandler"

    def handle(self, event: Any, **kwargs: Any) -> None:
        if isinstance(event, LLMChatEndEvent | LLMCompletionEndEvent):
            sink = _usage.get()
            if sink is not None:
                sink.append(tokens_from_raw(event.response.raw))  # type: ignore[union-attr]


get_dispatcher().add_event_handler(_UsageHandler())


@dataclass
class Outcome:
    """One decision: the prediction, an optional probability, tokens and wall time."""

    pred: Any = None
    score: float | None = None
    tokens: tuple[int, int] = (0, 0)
    latency_ms: float = 0.0
    error: str | None = None
    calls: int | None = None
    feedback: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def build_llm(system: str) -> Any:
    """The LlamaIndex LLM for a baseline system."""
    if system == "llm-strong":
        from llama_index.llms.anthropic import Anthropic, utils

        # llama-index-llms-anthropic 0.12.3 builds messages with `MessageParam(...)`, but the
        # anthropic SDK (2.3.0 here) exports `MessageParam` as a Union, which cannot be
        # instantiated. The SDK wants plain dicts anyway, so build those.
        utils.MessageParam = dict  # type: ignore[misc, assignment]
        return Anthropic(model=MODELS[system], max_tokens=16000)
    from llama_index.llms.openai import OpenAI

    return OpenAI(model=MODELS[system])


def build_faithfulness(system: str) -> Callable[[Any], Awaitable[Outcome]]:
    """A coroutine that judges one faithfulness example with the given system."""
    if system == "length-rule":

        async def rule(ex: faithfulness.FaithfulnessExample) -> Outcome:
            return Outcome(len(ex.response.split()) <= LENGTH_RULE_MAX_WORDS)

        return rule
    if system in DREX_SYSTEMS:
        from llama_index.evaluation.nace import NaceFaithfulnessEvaluator

        evaluator: Any = NaceFaithfulnessEvaluator(model=MODELS[system])
    else:
        from llama_index.core.evaluation import FaithfulnessEvaluator

        evaluator = FaithfulnessEvaluator(llm=build_llm(system))

    async def one(ex: faithfulness.FaithfulnessExample) -> Outcome:
        sink: list[tuple[int, int]] = []
        _usage.set(sink)
        # LlamaIndex's evaluator never sends the query; the `-noquery` systems match it.
        query = None if system in NOQUERY_SYSTEMS else ex.query
        start = time.perf_counter()
        result = await evaluator.aevaluate(query=query, response=ex.response, contexts=ex.contexts)
        ms = (time.perf_counter() - start) * 1000
        feedback = (result.feedback or "")[:80]
        if system in DREX_SYSTEMS:
            tokens = (result.usage.input_tokens or 0, result.usage.output_tokens or 0)
            return Outcome(bool(result.passing), result.probability, tokens, ms, feedback=feedback)
        tokens = (sum(t[0] for t in sink), sum(t[1] for t in sink))
        # LlamaIndex's refine step swallows validation errors and returns "Empty Response",
        # which the evaluator reads as "not faithful". That is a failure, not a verdict.
        if not sink or tokens == (0, 0) or (result.feedback or "").strip() == "Empty Response":
            msg = f"no usable LLM answer (calls={len(sink)}, feedback={feedback!r})"
            raise RuntimeError(msg)
        return Outcome(bool(result.passing), None, tokens, ms, calls=len(sink), feedback=feedback)

    return one


def build_routing(system: str) -> Callable[[Any], Awaitable[Outcome]]:
    """A coroutine that routes one query with the given system."""
    from llama_index.core.schema import QueryBundle

    if system in DREX_SYSTEMS:
        from llama_index.selectors.nace import NaceSingleSelector

        selector: Any = NaceSingleSelector(model=MODELS[system])
    else:
        from llama_index.core.selectors import LLMSingleSelector

        selector = LLMSingleSelector.from_defaults(llm=build_llm(system))

    async def one(ex: routing.RoutingExample) -> Outcome:
        sink: list[tuple[int, int]] = []
        _usage.set(sink)
        start = time.perf_counter()
        result = await selector.aselect(ex.tools, QueryBundle(ex.query))
        ms = (time.perf_counter() - start) * 1000
        pred = result.ind
        if system in DREX_SYSTEMS:
            tokens = (result.usage.input_tokens or 0, result.usage.output_tokens or 0)
            fell_back = bool(getattr(result, "fell_back", False))
            return Outcome(pred, None, tokens, ms, extra={"fell_back": fell_back})
        tokens = (sum(t[0] for t in sink), sum(t[1] for t in sink))
        if not sink or tokens == (0, 0):
            msg = f"no LLM call was recorded (calls={len(sink)})"
            raise RuntimeError(msg)
        return Outcome(pred, None, tokens, ms, calls=len(sink))

    return one


async def run_all(
    fn: Callable[[Any], Awaitable[Outcome]], examples: list[Any], concurrency: int
) -> list[Outcome]:
    """Run every example with bounded concurrency; failures become `Outcome(error=...)`."""
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def guarded(ex: Any) -> Outcome:
        nonlocal done
        async with sem:
            start = time.perf_counter()
            try:
                out = await fn(ex)
            except Exception as error:
                out = Outcome(
                    error=f"{type(error).__name__}: {str(error)[:160]}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(examples)}", flush=True)
            return out

    return await asyncio.gather(*(guarded(ex) for ex in examples))


def summarise(
    task: str, system: str, examples: list[Any], outcomes: list[Outcome]
) -> dict[str, Any]:
    """Aggregate outcomes into the result record. Errors count as wrong answers."""
    model = MODELS[system]
    price = FREE if system == "length-rule" else PRICES[model]
    ok = [o for o in outcomes if o.error is None]
    n = len(outcomes)
    if task in FAITHFULNESS_TASKS:
        labels = [ex.label for ex in examples]
        preds = [
            (o.pred if o.error is None else (not ex.label))
            for ex, o in zip(examples, outcomes, strict=True)
        ]
        acc = metrics.accuracy(labels, preds)
        bal = metrics.balanced_accuracy(labels, preds)
        scored = [
            (ex.label, o.score)
            for ex, o in zip(examples, outcomes, strict=True)
            if o.score is not None
        ]
        auc = (
            metrics.roc_auc([a for a, _ in scored], [b for _, b in scored]) if scored else math.nan
        )
    else:
        acc = (
            sum(
                o.error is None and o.pred == ex.label
                for ex, o in zip(examples, outcomes, strict=True)
            )
            / n
        )
        bal, auc = math.nan, math.nan
    lat = [o.latency_ms for o in ok] or [math.nan]
    if system == "length-rule":
        lat = [math.nan]
    cost = metrics.usd_per_1k(
        [o.tokens for o in ok] or [(0, 0)],
        input_per_m=price.input_per_m,
        output_per_m=price.output_per_m,
    )
    nn = lambda x: None if isinstance(x, float) and math.isnan(x) else x  # noqa: E731
    errors = [o.error for o in outcomes if o.error]
    return {
        "task": task,
        "system": system,
        "model": model,
        "n": n,
        "n_errors": len(errors),
        "accuracy": nn(acc),
        "balanced_accuracy": nn(bal),
        "roc_auc": nn(auc),
        "usd_per_1k": cost,
        "usd_total": cost * len(ok) / 1000,
        "input_tokens_mean": sum(o.tokens[0] for o in ok) / max(len(ok), 1),
        "output_tokens_mean": sum(o.tokens[1] for o in ok) / max(len(ok), 1),
        "p50_ms": nn(metrics.percentile(lat, 50)),
        "p95_ms": nn(metrics.percentile(lat, 95)),
        "date": date.today().isoformat(),
        "error_samples": sorted(set(errors))[:5],
    }


def example_row(ex: Any, o: Outcome) -> dict[str, Any]:
    """One per-example record for the JSONL next to the aggregate JSON."""
    return {
        "id": ex.id,
        "label": ex.label,
        "pred": o.pred,
        "score": o.score,
        "input_tokens": o.tokens[0],
        "output_tokens": o.tokens[1],
        "latency_ms": round(o.latency_ms, 1),
        "calls": o.calls,
        "feedback": o.feedback[:80],
        "error": o.error,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[*FAITHFULNESS_TASKS, "routing"], required=True)
    parser.add_argument("--system", choices=list(MODELS), required=True)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N examples")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    cap = MAX_CONCURRENCY[args.system]
    if args.concurrency > cap:
        parser.error(f"--concurrency for {args.system} is capped at {cap}")

    if args.task in FAITHFULNESS_TASKS:
        examples: list[Any] = faithfulness.load(matched=args.task == "faithfulness-matched")
        fn = build_faithfulness(args.system)
    else:
        if args.system in FAITHFULNESS_ONLY:
            parser.error(f"{args.system} is a faithfulness-only reference system")
        examples = routing.load()
        fn = build_routing(args.system)
    if args.limit:
        examples = examples[: args.limit]
    print(f"{args.task} / {args.system}: {len(examples)} examples")
    outcomes = asyncio.run(run_all(fn, examples, args.concurrency))
    record = summarise(args.task, args.system, examples, outcomes)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"-limit{args.limit}" if args.limit else ""
    path = args.out / f"{args.task}-{args.system}{suffix}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    rows = [example_row(ex, o) for ex, o in zip(examples, outcomes, strict=True)]
    path.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
