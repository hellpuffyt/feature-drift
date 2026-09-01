# Changelog

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
