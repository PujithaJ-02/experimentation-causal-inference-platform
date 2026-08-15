"""Check whether the ad's gap is real, or just luck.

In this file I take the two groups and ask one honest question: the ad group bought
a bit more, but is that gap big enough (across this many people) to trust, or could it
easily happen by chance even if the ad did nothing?
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_config, resolve_path  # noqa: E402


def compare_groups(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> dict:
    a = df.loc[df["treatment"] == 1, metric]
    b = df.loc[df["treatment"] == 0, metric]

    buyers_a, n_a = int(a.sum()), int(a.shape[0])
    buyers_b, n_b = int(b.sum()), int(b.shape[0])
    rate_a, rate_b = buyers_a / n_a, buyers_b / n_b
    gap = rate_a - rate_b

    stat, pvalue = proportions_ztest([buyers_a, buyers_b], [n_a, n_b])
    lo, hi = confint_proportions_2indep(
        buyers_a, n_a, buyers_b, n_b, method="wald", alpha=alpha
    )

    return {
        "metric": metric,
        "rate_a": rate_a, "rate_b": rate_b, "gap": gap,
        "ci_low": lo, "ci_high": hi,
        "pvalue": float(pvalue),
        "n_a": n_a, "n_b": n_b,
        "significant": bool(pvalue < alpha),
    }


def _print_result(r: dict) -> None:
    print(f"\n=== {r['metric'].upper()} ===")
    print(f"Group A (shown ad):     {r['rate_a']:.3%}  ({r['n_a']:,} people)")
    print(f"Group B (not shown ad): {r['rate_b']:.3%}  ({r['n_b']:,} people)")
    print(f"Gap (A minus B):        {r['gap']:+.3%}")
    print(f"95% confidence interval for the gap: {r['ci_low']:+.3%} to {r['ci_high']:+.3%}")
    print(f"p-value:                {r['pvalue']:.2e}")
    if r["significant"]:
        print("Verdict: the gap is statistically significant (unlikely to be luck).")
    else:
        print("Verdict: not statistically significant (could be luck).")


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(resolve_path(cfg["paths"]["processed_file"]))
    alpha = cfg["alpha"]

    primary = compare_groups(df, cfg["primary_metric"], alpha)
    _print_result(primary)

    guardrail = compare_groups(df, cfg["secondary_metric"], alpha)
    _print_result(guardrail)

    print("\n--- Plain-English summary ---")
    if primary["significant"] and primary["gap"] > 0:
        print(f"The ad increased buying by {primary['gap']:.3%}, and that is a real effect,")
        print(f"not luck. The true lift is likely between {primary['ci_low']:.3%} and "
              f"{primary['ci_high']:.3%}.")
    elif primary["significant"] and primary["gap"] < 0:
        print("The ad actually LOWERED buying, and that is a real effect. That is bad.")
    else:
        print("We cannot say the ad moved buying. The gap could be luck.")


if __name__ == "__main__":
    main()
