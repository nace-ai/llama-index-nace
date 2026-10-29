"""Tests for the benchmark metrics."""

import math

import pytest

from benchmark.metrics import (
    balanced_accuracy,
    percentile,
    roc_auc,
    usd_per_1k,
)


def test_balanced_accuracy() -> None:
    # 4 positives (3 right), 2 negatives (1 right): (0.75 + 0.5) / 2
    labels = [True, True, True, True, False, False]
    preds = [True, True, True, False, False, True]
    assert balanced_accuracy(labels, preds) == pytest.approx(0.625)


def test_roc_auc_from_probabilities() -> None:
    assert roc_auc([True, True, False, False], [0.9, 0.8, 0.3, 0.1]) == 1.0
    assert roc_auc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == 0.0
    # ties count half
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5
    assert roc_auc([True, False, True, False], [0.9, 0.8, 0.75, 0.7]) == pytest.approx(0.75)
    assert math.isnan(roc_auc([True, True], [0.1, 0.2]))


def test_usd_per_1k_from_usage() -> None:
    # 1000 decisions, each 2000 input and 100 output tokens at $2 / $10 per 1M
    usages = [(2000, 100)] * 4
    got = usd_per_1k(usages, input_per_m=2.0, output_per_m=10.0)
    assert got == pytest.approx((4 * 2000 * 2.0 + 4 * 100 * 10.0) / 1e6 / 4 * 1000)


def test_percentiles() -> None:
    values = list(range(1, 101))
    assert percentile(values, 50) == pytest.approx(50.5)
    assert percentile(values, 95) == pytest.approx(95.05)
    assert percentile([7.0], 95) == 7.0
