# Changelog

## Unreleased

- Fixed: non-UTF-8 input and CSV fields over 128 KiB crashed with a traceback;
  they now exit 1 with `error: ...` as documented.
- Fixed: a CSV header repeating a column name silently dropped one column; it is
  now refused.
- Fixed: "1 category ... carry" wording.
- Docs: README example replaced with the real output for `examples/`; install
  from source; verification against SciPy/NumPy recorded. Added `SECURITY.md`,
  `.editorconfig`, `docs/PRODUCTION_READINESS.md`, hand-computed stats tests.

## 0.1.0 — 2026-09-01

Initial release.

- PSI, KS statistic (numeric) / total variation distance (categorical), and
  missing-rate delta per feature, computed from stdlib only. Numeric PSI
  buckets by the *training* sample's quantiles; categorical PSI is computed
  over the union of categories so unseen ones are counted, not dropped.
- Unseen-category detection with the share of serving rows affected;
  schema changes (columns present on only one side) reported separately
  and always treated as severe.
- `feature-drift train serve` CLI over CSV / JSON Lines: text and JSON
  output, `--fail-on severe|drifting|never` (default: `severe`). Exit codes:
  `0` clean, `1` input error, `3` drift at or above the threshold.
- Zero runtime dependencies; Python 3.10–3.13.
