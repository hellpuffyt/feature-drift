"""Core model: two datasets' worth of feature values, and drift verdicts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureKind(Enum):
    """How a feature's values are compared."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class Verdict(Enum):
    """The per-feature conclusion, in increasing order of concern."""

    STABLE = "stable"
    DRIFTING = "drifting"
    SEVERE = "severe"

    @property
    def rank(self) -> int:
        return {"stable": 0, "drifting": 1, "severe": 2}[self.value]


@dataclass(frozen=True)
class FeatureColumn:
    """One feature's values from one dataset, already typed.

    Missing entries are counted, not dropped silently: a feature that is
    suddenly null half the time has drifted in the way that matters most,
    even if the non-null values look identical.
    """

    name: str
    kind: FeatureKind
    numbers: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()
    missing: int = 0

    @property
    def present(self) -> int:
        return len(self.numbers) if self.kind is FeatureKind.NUMERIC else len(self.categories)

    @property
    def total(self) -> int:
        return self.present + self.missing

    @property
    def missing_rate(self) -> float:
        return self.missing / self.total if self.total else 0.0


@dataclass(frozen=True)
class FeatureDrift:
    """Everything measured about one feature, with the verdict last.

    Each metric is reported with its own name rather than blended into one
    opaque score — "PSI 0.31" is checkable, "drift score 7/10" is not.
    """

    name: str
    kind: FeatureKind
    verdict: Verdict
    #: Population Stability Index between the two samples.
    psi: float
    #: Kolmogorov–Smirnov statistic (numeric) or total variation distance
    #: (categorical) — both live in [0, 1].
    distance: float
    #: Change in missing rate, serving minus training.
    missing_rate_delta: float
    #: Categories present in serving but never seen in training.
    unseen_categories: tuple[str, ...] = ()
    #: Human-readable notes: what moved, in the data's own units.
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "feature": self.name,
            "kind": self.kind.value,
            "verdict": self.verdict.value,
            "psi": self.psi,
            "distance": self.distance,
            "missing_rate_delta": self.missing_rate_delta,
            "unseen_categories": list(self.unseen_categories),
            "notes": list(self.notes),
        }
