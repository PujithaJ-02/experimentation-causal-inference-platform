"""CUPED: get a sharper answer from the same data by removing noise."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve_path  # noqa: E402

FEATURES = [f"f{i}" for i in range(12)]


def _ci_halfwidth(values_a: np.ndarray, values_b: np.ndarray) -> float:
    se = np.sqrt(values_a.var(ddof=1) / len(values_a) + values_b.var(ddof=1) / len(values_b))
    return 1.96 * se


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(resolve_path(cfg["paths"]["processed_file"]))
    metric = cfg["primary_metric"]

    y = df[metric].to_numpy(dtype=float)
    treated = df["treatment"].to_numpy() == 1

    ctrl = df[df["treatment"] == 0]
    model = LinearRegression().fit(ctrl[FEATURES].to_numpy(), ctrl[metric].to_numpy(dtype=float))
    x = model.predict(df[FEATURES].to_numpy())

    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adjusted = y - theta * (x - x.mean())

    gap_before = y[treated].mean() - y[~treated].mean()
    ci_before = _ci_halfwidth(y[treated], y[~treated])

    gap_after = y_adjusted[treated].mean() - y_adjusted[~treated].mean()
    ci_after = _ci_halfwidth(y_adjusted[treated], y_adjusted[~treated])

    reduction = (1 - ci_after / ci_before) * 100

    print("CUPED: same experiment, sharper estimate\n")
    print("Before CUPED (plain A/B test):")
    print(f"  ad lift = {gap_before:+.4%}   95% CI width = +/- {ci_before:.4%}")
    print("\nAfter CUPED (noise removed):")
    print(f"  ad lift = {gap_after:+.4%}   95% CI width = +/- {ci_after:.4%}")
    print("\n--- Plain-English summary ---")
    print(f"The estimated ad effect barely moved ({gap_before:.4%} vs {gap_after:.4%}),")
    print(f"which is the point: CUPED does not change the answer.")
    print(f"But the confidence interval got about {reduction:.1f}% narrower, so the same")
    print(f"result is now more precise. CUPED helps more when the f-columns predict buying")
    print(f"well, and less when they do not.")


if __name__ == "__main__":
    main()
