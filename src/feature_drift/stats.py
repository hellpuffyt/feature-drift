"""The statistics, each implemented plainly enough to check by hand.

Three measures, three reasons:

* **PSI** (Population Stability Index) is the industry's shared vocabulary
  for drift severity — its 0.1 / 0.25 rules of thumb are what an ML reviewer
  already knows how to read.
* **KS statistic** for numeric features is distribution-shape-agnostic and
  bounded in [0, 1], so "0.4" means the same thing for a latency column and
  a price column.
* **Total variation distance** is KS's natural categorical counterpart:
  half the L1 distance between the two category distributions, also [0, 1].

No p-values. At production sample sizes everything is "significant"; the
question a drift report answers is *how much moved*, not whether a null
hypothesis survives.
"""

from __future__ import annotations

import math
from collections import Counter

#: Quantile-bucket count for numeric PSI. Ten is the conventional choice —
#: fine enough to see shape, coarse enough that buckets stay populated.
PSI_BUCKETS = 10

#: Laplace-style floor so an empty bucket cannot send PSI to infinity; the
#: standard practical treatment, and it keeps PSI monotone in the shift.
_EPSILON = 1e-4


def quantile_edges(values: list[float], buckets: int = PSI_BUCKETS) -> list[float]:
    """Interior bucket edges from the *training* sample's quantiles.

    Edges come from training (the reference) so serving is judged against
    where training's mass actually sat, not against arbitrary round numbers.
    Duplicate edges collapse, so heavily-tied data yields fewer, honest
    buckets rather than empty ones.
    """
    if not values:
        return []
    ordered = sorted(values)
    edges = []
    for i in range(1, buckets):
        # Nearest-rank quantile: no interpolation to invent values.
        rank = math.ceil(i * len(ordered) / buckets) - 1
        edges.append(ordered[max(rank, 0)])
    deduped: list[float] = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def bucket_shares(values: list[float], edges: list[float]) -> list[float]:
    """Share of ``values`` in each bucket defined by ``edges`` (len+1 buckets)."""
    counts = [0] * (len(edges) + 1)
    for value in values:
        index = 0
        while index < len(edges) and value > edges[index]:
            index += 1
        counts[index] += 1
    total = len(values)
    return [count / total if total else 0.0 for count in counts]


def psi_from_shares(expected: list[float], actual: list[float]) -> float:
    """PSI between two share vectors of equal length."""
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e = max(e, _EPSILON)
        a = max(a, _EPSILON)
        total += (a - e) * math.log(a / e)
    return total


def numeric_psi(train: list[float], serve: list[float]) -> float:
    edges = quantile_edges(train)
    return psi_from_shares(bucket_shares(train, edges), bucket_shares(serve, edges))


def categorical_psi(train: list[str], serve: list[str]) -> float:
    """PSI over the union of categories, so unseen ones are counted."""
    vocabulary = sorted(set(train) | set(serve))
    train_counts = Counter(train)
    serve_counts = Counter(serve)
    expected = [train_counts[c] / len(train) if train else 0.0 for c in vocabulary]
    actual = [serve_counts[c] / len(serve) if serve else 0.0 for c in vocabulary]
    return psi_from_shares(expected, actual)


def ks_statistic(train: list[float], serve: list[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: max CDF gap.

    Tied values advance *both* CDFs together before the gap is measured —
    stepping one side at a time reports a phantom gap inside a tie, which
    is exactly the case (identical samples) that must score zero.
    """
    if not train or not serve:
        return 0.0
    a = sorted(train)
    b = sorted(serve)
    i = j = 0
    gap = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] == x:
            i += 1
        while j < len(b) and b[j] == x:
            j += 1
        gap = max(gap, abs(i / len(a) - j / len(b)))
    # Once one sample is exhausted its CDF sits at 1 while the other only
    # climbs toward it, so no larger gap can appear in the tail.
    return gap


def total_variation(train: list[str], serve: list[str]) -> float:
    """Half the L1 distance between the two category distributions."""
    if not train or not serve:
        return 0.0
    vocabulary = set(train) | set(serve)
    train_counts = Counter(train)
    serve_counts = Counter(serve)
    return 0.5 * sum(
        abs(train_counts[c] / len(train) - serve_counts[c] / len(serve))
        for c in vocabulary
    )


def summary(values: list[float]) -> tuple[float, float]:
    """(mean, population standard deviation) — for the human-readable notes."""
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (mean, math.sqrt(variance))
