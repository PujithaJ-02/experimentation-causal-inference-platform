"""A few small tests that double-check my A/B math on cases where I already know the answer."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ab_test import compare_groups


def _toy(rate_a_buyers, n_a, rate_b_buyers, n_b):
    rows = []
    rows += [{"treatment": 1, "conversion": 1}] * rate_a_buyers
    rows += [{"treatment": 1, "conversion": 0}] * (n_a - rate_a_buyers)
    rows += [{"treatment": 0, "conversion": 1}] * rate_b_buyers
    rows += [{"treatment": 0, "conversion": 0}] * (n_b - rate_b_buyers)
    return pd.DataFrame(rows)


def test_rates_are_computed_correctly():
    df = _toy(20, 100, 10, 100)
    r = compare_groups(df, "conversion")
    assert abs(r["rate_a"] - 0.20) < 1e-9
    assert abs(r["rate_b"] - 0.10) < 1e-9
    assert abs(r["gap"] - 0.10) < 1e-9


def test_no_difference_is_not_significant():
    df = _toy(50, 1000, 50, 1000)
    r = compare_groups(df, "conversion")
    assert abs(r["gap"]) < 1e-9
    assert r["significant"] is False


def test_big_clear_difference_is_significant():
    df = _toy(300, 1000, 50, 1000)
    r = compare_groups(df, "conversion")
    assert r["gap"] > 0
    assert r["significant"] is True


def test_confidence_interval_brackets_the_gap():
    df = _toy(120, 1000, 80, 1000)
    r = compare_groups(df, "conversion")
    assert r["ci_low"] <= r["gap"] <= r["ci_high"]
