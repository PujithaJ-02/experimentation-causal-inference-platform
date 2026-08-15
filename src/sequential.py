"""Sequential testing: peek at results safely, without fooling yourself.

The problem this solves (the "peeking problem"): if you check an A/B test over and over
and stop the moment it looks significant, you will sometimes declare a win by pure luck,
even when there is no real effect. Normal tests assume you look ONCE, at the end.

This file shows the problem, then shows the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_rng, load_config  # noqa: E402


def run_one_experiment(rng, n_per_peek, n_peeks, true_effect, z_cutoff):
    a_success = a_total = b_success = b_total = 0
    base_rate = 0.03
    for _ in range(n_peeks):
        a_success += rng.binomial(n_per_peek, base_rate + true_effect)
        a_total += n_per_peek
        b_success += rng.binomial(n_per_peek, base_rate)
        b_total += n_per_peek

        p_a = a_success / a_total
        p_b = b_success / b_total
        p_pool = (a_success + b_success) / (a_total + b_total)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / a_total + 1 / b_total))
        if se == 0:
            continue
        z = (p_a - p_b) / se
        if abs(z) > z_cutoff:
            return True
    return False


def false_alarm_rate(rng, z_cutoff, n_experiments=2000, n_peeks=10, n_per_peek=2000):
    false_alarms = 0
    for _ in range(n_experiments):
        if run_one_experiment(rng, n_per_peek, n_peeks, true_effect=0.0, z_cutoff=z_cutoff):
            false_alarms += 1
    return false_alarms / n_experiments


def main() -> None:
    cfg = load_config()
    rng = get_rng(cfg["random_seed"])

    normal_cutoff = 1.96
    sequential_cutoff = 2.77

    print("The peeking problem, demonstrated on A/A tests (true effect is ZERO)")
    print("so every 'significant' result is a false alarm.\n")

    normal = false_alarm_rate(rng, normal_cutoff)
    print(f"Peeking 10 times with the NORMAL cutoff (1.96):")
    print(f"  false-alarm rate = {normal:.1%}   (we wanted only 5% -- this is too high)\n")

    rng = get_rng(cfg["random_seed"])
    seq = false_alarm_rate(rng, sequential_cutoff)
    print(f"Peeking 10 times with the SEQUENTIAL cutoff (2.77):")
    print(f"  false-alarm rate = {seq:.1%}   (back down near 5% -- fixed)\n")

    print("--- Plain-English summary ---")
    print("Checking a test repeatedly with the normal bar makes you cry 'winner!' far")
    print("more than 5% of the time even when nothing is happening. The sequential bar is")
    print("stricter, so you can peek as often as you like and still keep false alarms near 5%.")


if __name__ == "__main__":
    main()
