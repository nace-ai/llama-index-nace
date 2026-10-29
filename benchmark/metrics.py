"""Metrics for the benchmark: accuracy, ROC-AUC, cost per 1,000 decisions, percentiles."""

from __future__ import annotations

import math
from collections.abc import Sequence


def accuracy(labels: Sequence[object], preds: Sequence[object]) -> float:
    """Fraction of predictions equal to their labels."""
    return sum(a == b for a, b in zip(labels, preds, strict=True)) / len(labels)


def balanced_accuracy(labels: Sequence[bool], preds: Sequence[bool]) -> float:
    """Mean of recall on positives and recall on negatives."""
    pos = [p for label, p in zip(labels, preds, strict=True) if label]
    neg = [not p for label, p in zip(labels, preds, strict=True) if not label]
    if not pos or not neg:
        return math.nan
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Area under the ROC curve (Mann-Whitney U, ties count half); NaN with one class."""
    pos = [s for label, s in zip(labels, scores, strict=True) if label]
    neg = [s for label, s in zip(labels, scores, strict=True) if not label]
    if not pos or not neg:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def usd_per_1k(
    usages: Sequence[tuple[int, int]], *, input_per_m: float, output_per_m: float
) -> float:
    """Dollars per 1,000 decisions from (input_tokens, output_tokens) per decision."""
    total = sum(i * input_per_m + o * output_per_m for i, o in usages) / 1e6
    return total / len(usages) * 1000


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, `q` in 0..100."""
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
