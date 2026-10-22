"""Compare two loaded datasets feature by feature and pass verdicts."""

from __future__ import annotations

from collections import Counter

from . import stats
from .types import FeatureColumn, FeatureDrift, FeatureKind, Verdict

#: The conventional PSI reading: under 0.1 stable, 0.1–0.25 worth a look,
#: above 0.25 the population has genuinely shifted.
PSI_DRIFTING = 0.1
PSI_SEVERE = 0.25

#: A missing-rate jump of this much is severe on its own — the values that
#: *are* present no longer describe the same population.
MISSING_DELTA_SEVERE = 0.2
MISSING_DELTA_DRIFTING = 0.05


def compare(
    train: dict[str, FeatureColumn],
    serve: dict[str, FeatureColumn],
) -> tuple[list[FeatureDrift], list[str], list[str]]:
    """(per-feature drift, features only in training, features only in serving).

    Features present on one side only are reported as lists rather than
    force-fitted into drift rows — a column that vanished is a schema change,
    not a distribution shift, and pretending otherwise buries it.
    """
    only_train = sorted(set(train) - set(serve))
    only_serve = sorted(set(serve) - set(train))

    drifts = []
    for name in sorted(set(train) & set(serve)):
        drifts.append(_feature(train[name], serve[name]))
    drifts.sort(key=lambda d: (-d.verdict.rank, -d.psi, d.name))
    return drifts, only_train, only_serve


def _feature(train: FeatureColumn, serve: FeatureColumn) -> FeatureDrift:
    notes: list[str] = []

    # A column numeric in training but not in serving (or vice versa) is
    # compared categorically — the honest common denominator — and noted.
    kind = train.kind
    if train.kind is not serve.kind:
        kind = FeatureKind.CATEGORICAL
        notes.append(
            f"typed {train.kind.value} in training but {serve.kind.value} in "
            "serving; compared as categories"
        )

    if kind is FeatureKind.NUMERIC:
        train_values = list(train.numbers)
        serve_values = list(serve.numbers)
        psi = stats.numeric_psi(train_values, serve_values)
        distance = stats.ks_statistic(train_values, serve_values)
        unseen: tuple[str, ...] = ()
        train_mean, train_std = stats.summary(train_values)
        serve_mean, _ = stats.summary(serve_values)
        if train_std > 0 and abs(serve_mean - train_mean) >= 0.1 * train_std:
            notes.append(
                f"mean moved {train_mean:g} -> {serve_mean:g} "
                f"({(serve_mean - train_mean) / train_std:+.1f} training sd)"
            )
    else:
        train_values_c = _as_categories(train)
        serve_values_c = _as_categories(serve)
        psi = stats.categorical_psi(train_values_c, serve_values_c)
        distance = stats.total_variation(train_values_c, serve_values_c)
        seen = set(train_values_c)
        serve_counts = Counter(serve_values_c)
        unseen = tuple(sorted(c for c in serve_counts if c not in seen))
        if unseen:
            share = sum(serve_counts[c] for c in unseen) / max(len(serve_values_c), 1)
            notes.append(
                f"{len(unseen)} categor{'y' if len(unseen) == 1 else 'ies'} "
                f"never seen in training {'carries' if len(unseen) == 1 else 'carry'} "
                f"{share:.1%} of serving rows"
            )

    missing_delta = serve.missing_rate - train.missing_rate
    if abs(missing_delta) >= MISSING_DELTA_DRIFTING:
        notes.append(
            f"missing rate {train.missing_rate:.1%} -> {serve.missing_rate:.1%}"
        )

    return FeatureDrift(
        name=train.name,
        kind=kind,
        verdict=_verdict(psi, missing_delta, bool(unseen)),
        psi=psi,
        distance=distance,
        missing_rate_delta=missing_delta,
        unseen_categories=unseen,
        notes=tuple(notes),
    )


def _as_categories(column: FeatureColumn) -> list[str]:
    if column.kind is FeatureKind.CATEGORICAL:
        return list(column.categories)
    return [format(v, "g") for v in column.numbers]


def _verdict(psi: float, missing_delta: float, has_unseen: bool) -> Verdict:
    if psi >= PSI_SEVERE or abs(missing_delta) >= MISSING_DELTA_SEVERE:
        return Verdict.SEVERE
    if psi >= PSI_DRIFTING or abs(missing_delta) >= MISSING_DELTA_DRIFTING or has_unseen:
        return Verdict.DRIFTING
    return Verdict.STABLE
