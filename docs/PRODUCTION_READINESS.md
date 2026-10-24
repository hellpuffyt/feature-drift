# Production readiness — feature-drift

Audit date: 2026-10-22. Tested commit: `a96001d` (branch `audit/production-readiness`, based on `main` @ `6a71ae0`).

## Scope

Stdlib-only CLI: compare a training and a serving table (CSV or JSON Lines) per feature — PSI
(training-quantile buckets / category union), KS statistic (numeric), total variation distance
(categorical), missing-rate delta, unseen categories, schema changes. Exit 0 / 1 bad input / 3 drift.

## Platforms

CPython 3.10+, no runtime dependencies. Verified on macOS 27 (Apple M5), CPython 3.12.14.

## Clean-clone verification (commit `a96001d`)

```bash
git clone -b audit/production-readiness <local clone> feature-drift && cd feature-drift
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install .
feature-drift examples/train.csv examples/serving.csv     # README output, exit 3
feature-drift examples/train.csv examples/serving.csv --format json   # valid JSON
feature-drift examples/train.csv examples/train.csv       # all stable, exit 0
uv pip install -e ".[dev]" && pytest && ruff check . && mypy          # 67 passed / clean / clean
```

## Commands and results

| Command | Baseline (`6a71ae0`) | After (`a96001d`) |
| --- | --- | --- |
| `pytest` | exit 0 — 60 passed, 0 failed, 0 skipped | exit 0 — 67 passed, 0 failed, 0 skipped |
| `ruff check .` / `mypy` | exit 0 / exit 0 | exit 0 / exit 0 |
| `pytest --cov --cov-fail-under=95` (CI gate) | — | 97.4%, pass |
| `uvx pip-audit` | — | no known vulnerabilities |

CI status — default branch: success (`6a71ae0`, 2026-09-01); audit branch: pending until pushed.

## Statistics verification

Against SciPy 1.18.1 / NumPy in a throwaway environment, 1,000 random sample pairs with ties
(rounded values): KS vs `scipy.stats.ks_2samp(...).statistic` max |diff| 2.2e-16; numeric PSI vs an
independent NumPy implementation on the same training-quantile edges max |diff| 8.9e-16; total
variation vs hand formula exact. New hand-computed tests: KS 0.5, TV 0.5, categorical PSI 0.25·ln 3.

## Bugs found and fixed

| Severity | Bug | Regression test |
| --- | --- | --- |
| Medium (silent data loss) | header `a,a` → one column silently dropped by `DictReader` | `tests/test_io.py::test_repeated_header_is_refused` |
| Low (crash vs exit-1 contract) | `UnicodeDecodeError` (CSV and JSONL), `csv.Error` field > 128 KiB → traceback | `test_undecodable_csv_...`, `test_undecodable_jsonl_...`, `test_oversized_csv_field_...` |
| Docs | README console example did not match any real run (different numbers) | README now shows the real `examples/` output |
| Cosmetic | "1 category … carry" | covered by README example |

## Security review

No network, no `eval`/pickle; JSON via `json.loads`. History secret scan: no hits. `pip-audit`: clean.
Workflow permissions `contents: read`; actions tag-pinned (left as is).

## Known limitations

- Whole tables are loaded into memory (no streaming).
- PSI on tiny samples is dominated by the 1e-4 empty-bucket floor (documented in README).
- No p-values by design.

## Acceptance checklist

| # | Item | Expected | Test | Result |
| --- | --- | --- | --- | --- |
| 1 | KS correct incl. ties | = SciPy | comparison + hand test | PASS |
| 2 | PSI correct | = NumPy / hand | comparison + hand test | PASS |
| 3 | TV correct | = hand | hand test | PASS |
| 4 | Unseen categories reported with share | note shown | example run | PASS |
| 5 | Identical inputs → stable, exit 0 | exit 0 | clean clone | PASS |
| 6 | Drift → exit 3; bad input → exit 1 | codes | CLI tests | PASS |
| 7 | Clean clone | install/tests/example | above | PASS |

## Classification proposal

**Production-Ready for Its Documented Scope** (proposal for lead verification).
