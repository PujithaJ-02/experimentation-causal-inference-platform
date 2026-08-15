# Experimentation and Causal Inference Platform

I built this to answer a question that comes up on every product team I have seen:
we ran an experiment, now what do we actually do about it? This pipeline takes a
raw randomized experiment and gives back a clear decision (ship it, ship it to a
specific segment, or kill it), along with the statistics that decision should rest on.

Most portfolio projects stop at a t-test. I wanted this one to look like the work an
experimentation team actually does, so it goes further: it checks that the randomization
held before trusting anything, reduces variance with CUPED, stays honest when you peek at
results early, and models which users respond best rather than just asking "did it work."

## What it does

Point it at a randomized experiment and it will:

- confirm the treatment and control groups are actually comparable before anything else,
- run the primary A/B analysis and report a confidence interval, not just a p-value,
- check a guardrail metric so a "win" is not quietly doing harm somewhere,
- tighten the estimate with CUPED and keep repeated checking from inflating false positives,
- find which segments respond most and score users by predicted uplift,
- and end with a one-page memo a product lead can read in two minutes.

## The data

I use the Criteo Uplift v2.1 benchmark: about 14 million rows from a real ad-exposure
experiment, 12 anonymous features, a treatment flag, and two outcomes (visit and conversion).
Two things about it are worth knowing up front. The split is roughly 85% treated and 15%
control, not 50/50, and the conversion rate is very low (around 0.29%), which makes conversion
a low-power metric. I report it with a confidence interval and lean on visit as a
higher-power secondary.

If the real file cannot be downloaded (for example on a locked-down machine), the loader
falls back to a simulated experiment built from a documented data-generating process. The
nice side effect is that the simulation has a known true treatment effect, so I can check
that every estimator recovers the number it is supposed to.

Raw data never gets committed. The loader that produces it does.

## Running it

```bash
pip install -r requirements.txt
make data     # download the real data or simulate it, then write the processed file
make test     # run the unit tests
```

More `make` targets get wired in as each phase lands.

## Principles I am holding myself to

1. If a fancier method does not beat the simple one, I say so. I do not hide it.
2. Every effect gets a confidence interval. No bare point estimates.
3. One fixed random seed, set everywhere, so the numbers reproduce from a clean clone.
4. Raw data stays out of the repo; the code that makes it stays in.

## Status

Early. The scaffold and the plan are in place. Phases land one at a time, and this README
grows with them. The decision memo, not the code, is the thing to read first once it exists.
