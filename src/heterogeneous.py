"""Find out WHO the ad works best on."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve_path  # noqa: E402


def make_tiers(df: pd.DataFrame) -> pd.Series:
    ranked = df["f0"].rank(method="first")
    return pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])


def effect_in_segment(seg: pd.DataFrame, metric: str, alpha: float) -> dict:
    a = seg.loc[seg["treatment"] == 1, metric]
    b = seg.loc[seg["treatment"] == 0, metric]
    buyers_a, n_a = int(a.sum()), int(a.shape[0])
    buyers_b, n_b = int(b.sum()), int(b.shape[0])
    rate_a = buyers_a / n_a if n_a else float("nan")
    rate_b = buyers_b / n_b if n_b else float("nan")
    _, pvalue = proportions_ztest([buyers_a, buyers_b], [n_a, n_b])
    lo, hi = confint_proportions_2indep(buyers_a, n_a, buyers_b, n_b,
                                        method="wald", alpha=alpha)
    return {
        "rate_a": rate_a, "rate_b": rate_b, "gap": rate_a - rate_b,
        "ci_low": lo, "ci_high": hi, "pvalue": float(pvalue),
        "n_a": n_a, "n_b": n_b,
    }


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(resolve_path(cfg["paths"]["processed_file"]))
    metric = cfg["primary_metric"]
    alpha = cfg["alpha"]

    df = df.copy()
    df["engagement_tier"] = make_tiers(df)

    tiers = ["Q1", "Q2", "Q3", "Q4"]
    strict_alpha = alpha / len(tiers)

    print("Ad effect on buying, split into 4 equal groups by f0 (Q1 low -> Q4 high):")
    print(f"(stricter bar for 'real' because I test {len(tiers)} groups: "
          f"p must beat {strict_alpha:.4f})\n")

    results = []
    for tier in tiers:
        seg = df[df["engagement_tier"] == tier]
        r = effect_in_segment(seg, metric, alpha)
        r["tier"] = tier
        r["real"] = r["pvalue"] < strict_alpha
        results.append(r)
        flag = "REAL" if r["real"] else "not clearly real"
        print(f"{tier}:  {r['n_a'] + r['n_b']:>9,} people   "
              f"ad lift = {r['gap']:+.3%}   "
              f"(from {r['rate_b']:.3%} up to {r['rate_a']:.3%})   {flag}")

    best = max(results, key=lambda r: r["gap"])
    print("\n--- Plain-English summary ---")
    print(f"The ad helps the most in group {best['tier']}, "
          f"where it lifts buying by {best['gap']:.3%}.")
    print("That group is where the store should focus the ad budget.")


if __name__ == "__main__":
    main()
