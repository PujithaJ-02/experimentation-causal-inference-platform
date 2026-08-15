# Experimentation and Causal Inference Platform

I built this to answer a question that comes up on every product team: we ran an
experiment, now what do we actually do about it? This project takes a real randomized
ad experiment and turns it into a clear decision, backed by the statistics that decision
should rest on.

## The headline result

Using Criteo's real ad experiment (about 14 million users, randomly split into a group
shown the ad and a group held back), I found:

- **The ad works.** Buying went from 0.19% without the ad to 0.31% with it. Across 14
  million people that difference is far too steady to be luck, so it is a real effect.
- **It did no harm.** Site visits went up too, so nothing got worse while buying improved.
- **It only really helps one slice of users.** When I split users into four equal groups,
  almost all of the benefit landed in a single group. For the rest the ad barely moved.

So the recommendation is to target the ad at the group it actually works on, rather than
paying to show it to everyone. The full write-up is in
[reports/decision_memo.md](reports/decision_memo.md).

## What the project does, step by step

1. **Load the data** (`src/data_loading.py`) - downloads the real Criteo experiment, or
   builds realistic stand-in data if the download is not reachable.
2. **Run the A/B test** (`src/ab_test.py`) - compares the two groups, reports a confidence
   interval, and checks whether the difference is real or luck.
3. **Find who responds** (`src/heterogeneous.py`) - splits users into groups and measures
   the ad's effect in each, to see who it works best on.
4. **The decision memo** (`reports/decision_memo.md`) - the plain-English recommendation.

## How to run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/data_loading.py     # get the data
python3 src/ab_test.py          # is the ad effect real?
python3 src/heterogeneous.py    # who does it work best on?
```

## The data

Criteo Uplift v2.1: about 14 million users, 12 anonymous features (f0..f11), a treatment
flag (shown the ad or not), and two outcomes (visit and conversion). The split is roughly
85% shown the ad and 15% held back, and the buying rate is low (around 0.3%), so effects
are small but measurable at this scale. The raw data is not committed to the repo; the
loader reproduces it.

## Honest notes

- The user groups are built from an anonymized feature, so I cannot yet say in plain terms
  who the high-response group actually is. That would be the next step before real targeting.
- "Statistically real" is not the same as "worth the money." The ad clearly has an effect;
  whether it is worth its cost is a separate finance question.
- One fixed random seed is set everywhere, so results reproduce from a clean clone.
