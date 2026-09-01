# feature-drift

Detect feature drift between training and serving data, and report **which
features moved and by how much** — in metrics a reviewer can check, not a
blended score. Zero runtime dependencies — Python 3.10+ and nothing else.

```console
$ feature-drift train.csv serving.csv
training: train.csv
serving:  serving.csv

SEVERE  amount  PSI=0.412  distance=0.371
          mean moved 48.2 -> 96.5 (+2.6 training sd)
SEVERE  channel  PSI=0.310  distance=0.248
          1 category never seen in training carries 12.4% of serving rows
drifting latency_ms  PSI=0.121  distance=0.089
stable  country  PSI=0.004  distance=0.011

3 of 4 shared feature(s) moved
```

## What it measures, and why these

- **PSI** (Population Stability Index) per feature — the industry's shared
  vocabulary: < 0.1 stable, 0.1–0.25 drifting, > 0.25 severe. Numeric
  features are bucketed by the *training* sample's quantiles, so serving is
  judged against where training's mass actually sat.
- **KS statistic** (numeric) / **total variation distance** (categorical) —
  both bounded in [0, 1], so the number means the same thing for a latency
  column and a price column.
- **Missing-rate delta** — a feature that is suddenly null half the time has
  drifted in the way that matters most, even when its non-null values look
  identical. A ≥ 20-point jump is severe on its own.
- **Unseen categories** — values the model never trained on, with the share
  of serving rows they carry.
- **Schema changes** — a column that vanished or appeared is reported as
  exactly that, and is severe by itself; force-fitting it into a drift row
  would bury it.

**No p-values, deliberately.** At production sample sizes everything is
"significant"; the question a drift report answers is *how much moved*.

## Input

Two tables — training/reference and serving/current — as CSV (with header)
or JSON Lines. A column is numeric when every present value parses as a
finite number; `NA`/`null`/empty count as missing, not as demotion to
categorical. A column typed differently on the two sides is compared as
categories, with a note saying so.

## Usage

```console
feature-drift train.csv serve.csv
feature-drift train.jsonl serve.jsonl --format json   # machine-readable
feature-drift train.csv serve.csv --fail-on drifting  # stricter gate
```

Exit codes are the monitoring interface: `0` nothing at or above
`--fail-on` (default: `severe`), `1` bad input, `3` drift at or above it.

## Development

```console
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

MIT licensed.
