"""Get the experiment data ready to work with.

In this file I do one job: produce a clean table of the experiment that the rest of
the project can use. I try two things, in order.

1. First I try to download the REAL Criteo file: about 14 million real users, each one
   randomly shown an ad (Group A) or held back from it (Group B), with columns for
   whether they later visited the site and whether they bought.

2. If that download does not work (for example my machine cannot reach the internet, or
   the file has moved), I fall back to MAKING fake data that has the exact same shape.
   I build it from a recipe I wrote out below, so it behaves like a real experiment.
   The bonus: because I invented the fake data, I know the true ad effect, so later I can
   check that my analysis recovers the right answer.

Either way, the columns are the same, so nothing downstream cares which path ran:
  f0..f11      -> 12 anonymous facts about each user (Criteo hid what they mean)
  treatment    -> 1 if the person was shown the ad (Group A), 0 if not (Group B)
  visit        -> 1 if they later visited the site
  conversion   -> 1 if they later bought / signed up   (this is the main thing I care about)

--------------------------------------------------------------------------------------
The recipe for the fake data (so I can explain it in an interview):
  - I draw 12 random facts per user (f0..f11), same as the real anonymous features.
  - I flip a biased coin: ~85% land in Group A (shown the ad), ~15% in Group B.
    The coin ignores the person's facts, which is what makes it a fair, valid split.
  - Each person has a baseline chance of buying that depends on their facts.
  - The ad adds a real bump to that chance. I make the bump BIGGER for more engaged
    users, so later I can show that targeting the right people beats treating everyone.
  - I then flip the "did they buy" coin using their chance, with the ad bump applied
    only to Group A. I do the same for "did they visit."
  - I save the true ad effect I baked in, so I can grade my own analysis against it.
--------------------------------------------------------------------------------------
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Let me import my helpers whether I run this as a script or as part of a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_rng, load_config, resolve_path, save_json, set_seed  # noqa: E402

# Where the real file lives online. This works from a normal machine with internet.
REAL_DATA_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/"
    "resolve/main/criteo-research-uplift-v2.1.csv.gz"
)
FEATURES = [f"f{i}" for i in range(12)]
EPS = 1e-6  # a tiny number so probabilities never hit exactly 0 or 1


def _quartile_tier(f0):
    """Sort users into four engagement tiers (Q1 lowest to Q4 highest) using one of
    their facts. I use this later to ask 'does the ad help some groups more than others'."""
    cutoffs = np.quantile(f0, [0.25, 0.5, 0.75])
    return np.array(["Q1", "Q2", "Q3", "Q4"])[np.digitize(f0, cutoffs)]


# ------------------------------------------------------------------------------------ #
# Path 1: the real data
# ------------------------------------------------------------------------------------ #
def load_real(cfg: dict) -> pd.DataFrame:
    """Try to download and tidy up the real Criteo file. If anything fails, I raise an
    error on purpose so the caller knows to fall back to the fake data."""
    df = pd.read_csv(REAL_DATA_URL, compression="gzip")
    needed = FEATURES + ["treatment", "conversion", "visit"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"The real file is missing columns I expected: {missing}")
    df = df[needed].copy()
    df["engagement_tier"] = _quartile_tier(df["f0"].to_numpy())
    df["source"] = "criteo_real"
    return df


# ------------------------------------------------------------------------------------ #
# Path 2: the fake data (only used if the download fails)
# ------------------------------------------------------------------------------------ #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _find_intercept(linear, target_rate):
    """Nudge the baseline up or down until the average buying chance matches the rate I
    asked for in my settings. It is a simple squeeze-from-both-sides search."""
    low, high = -20.0, 20.0
    for _ in range(100):
        mid = (low + high) / 2
        if _sigmoid(mid + linear).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def simulate(cfg: dict):
    """Build the fake experiment from the recipe in the docstring at the top."""
    seed = cfg["random_seed"]
    rng = get_rng(seed)
    s = cfg["sim"]
    n = int(s["n_users"])
    n_feat = int(s["n_features"])

    # 12 random facts per user.
    facts = rng.standard_normal((n, n_feat))
    df = pd.DataFrame(facts, columns=[f"f{i}" for i in range(n_feat)])
    df["engagement_tier"] = _quartile_tier(df["f0"].to_numpy())

    # The biased coin flip that assigns Group A vs Group B, ignoring the facts.
    df["treatment"] = (rng.random(n) < float(cfg["treatment_ratio"])).astype(int)

    # Each person's baseline chance of buying, tied to a few of their facts.
    weights = np.array([0.6, -0.4, 0.3] + [0.0] * (n_feat - 3))
    linear = facts @ weights
    intercept = _find_intercept(linear, s["base_conversion_rate"])
    base_chance = _sigmoid(intercept + linear)

    # The ad's real effect. I make it bigger for higher tiers so the ad clearly helps
    # engaged users more. The multipliers average to 1, so the overall effect equals the
    # target I set in config (true_ate_conversion).
    bump_by_tier = {"Q1": 0.0, "Q2": 0.5, "Q3": 1.0, "Q4": 2.5}
    bump = np.array([bump_by_tier[t] for t in df["engagement_tier"]]) * float(s["true_ate_conversion"])

    chance_if_A = np.clip(base_chance + bump, EPS, 1 - EPS)  # shown the ad
    chance_if_B = np.clip(base_chance, EPS, 1 - EPS)          # not shown the ad
    actual_chance = np.where(df["treatment"] == 1, chance_if_A, chance_if_B)
    df["conversion"] = (rng.random(n) < actual_chance).astype(int)

    # The visit signal: a smaller, steady ad effect, so it never looks harmful.
    weights_v = np.array([0.3, 0.2, 0.0] + [0.0] * (n_feat - 3))
    base_visit = _sigmoid(_find_intercept(facts @ weights_v, s["base_visit_rate"]) + facts @ weights_v)
    visit_bump = 0.01
    visit_chance = np.where(df["treatment"] == 1,
                            np.clip(base_visit + visit_bump, EPS, 1 - EPS),
                            np.clip(base_visit, EPS, 1 - EPS))
    df["visit"] = (rng.random(n) < visit_chance).astype(int)

    df["source"] = "simulated"

    # The answer key: the true effects I baked in, so I can grade my analysis later.
    truth = {
        "source": "simulated",
        "seed": seed,
        "n_users": n,
        "true_ad_effect_on_buying": float(bump.mean()),
        "true_effect_by_tier": {t: float(m * s["true_ate_conversion"]) for t, m in bump_by_tier.items()},
        "true_effect_on_visit": float(visit_bump),
    }
    return df, truth


# ------------------------------------------------------------------------------------ #
# A quick safety check before I trust the data
# ------------------------------------------------------------------------------------ #
def validate(df: pd.DataFrame, cfg: dict) -> None:
    """Fail loudly if the data is not shaped the way I expect. Catching a problem here
    saves me from quietly computing a wrong answer later."""
    for col in FEATURES + ["treatment", "conversion", "visit", "engagement_tier"]:
        assert col in df.columns, f"Missing column: {col}"
    for col in ["treatment", "conversion", "visit"]:
        assert set(pd.unique(df[col])).issubset({0, 1}), f"{col} should only be 0 or 1"
        assert df[col].isna().sum() == 0, f"{col} has empty values"
    n_A = int((df["treatment"] == 1).sum())
    n_B = int((df["treatment"] == 0).sum())
    assert n_A > 0 and n_B > 0, "Both groups must have people in them"


# ------------------------------------------------------------------------------------ #
# The main routine that ties it together
# ------------------------------------------------------------------------------------ #
def main() -> None:
    cfg = load_config()
    set_seed(cfg["random_seed"])

    try:
        print("Trying to download the real Criteo data ...")
        df = load_real(cfg)
        truth = None
        print(f"  Got the real data: {len(df):,} rows.")
    except Exception as exc:
        print(f"  Could not get the real data ({type(exc).__name__}). No problem.")
        print("Making realistic stand-in data instead ...")
        df, truth = simulate(cfg)
        print(f"  Built {len(df):,} rows of stand-in data.")

    validate(df, cfg)

    out = resolve_path(cfg["paths"]["processed_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved the data to {out}")
    if truth is not None:
        save_json(truth, cfg["paths"]["ground_truth_file"])
        print("Saved the answer key (true effects) alongside it.")

    # The friendly summary: this is the exact Group A vs Group B comparison we talked about.
    A = df[df["treatment"] == 1]
    B = df[df["treatment"] == 0]
    print("\n--- Quick look ---")
    print(f"Group A (shown the ad):     {len(A):,} people, {A['conversion'].mean():.2%} bought")
    print(f"Group B (not shown the ad): {len(B):,} people, {B['conversion'].mean():.2%} bought")
    print(f"Raw gap in buying rate:     {(A['conversion'].mean() - B['conversion'].mean()):+.2%}")
    print("(Later phases decide whether that gap is real, and who to target.)")


if __name__ == "__main__":
    main()
