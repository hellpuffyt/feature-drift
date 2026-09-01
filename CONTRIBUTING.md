# Contributing

Thanks for looking into feature-drift.

## Setup

```console
python -m venv .venv && . .venv/bin/activate   # Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a pull request

```console
pytest            # the whole suite; new behaviour needs new tests
ruff check .      # lint
mypy              # strict — the src/ tree carries full annotations
```

## Ground rules

- **No runtime dependencies.** `dependencies = []` is a feature, not an
  oversight.
- **Every metric is named and checkable.** PSI, the KS statistic, total
  variation and missing-rate delta each report under their own name in
  `notes`/`to_dict`; nothing is blended into one opaque score.
- **No p-values.** At production sample sizes everything is "significant" -
  the report answers how much moved, not whether a null hypothesis holds.
- **Thresholds are named constants** (`PSI_DRIFTING`, `PSI_SEVERE`,
  `MISSING_DELTA_SEVERE`, ...) with a comment defending the number.
