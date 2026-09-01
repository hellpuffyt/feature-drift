from __future__ import annotations

import random

from feature_drift import drift, io
from feature_drift.types import Verdict


def columns(rows: list[dict[str, str]]):
    return io.columns_from_rows(rows)


def numeric_rows(name: str, values: list[float]) -> list[dict[str, str]]:
    return [{name: str(v)} for v in values]


class TestStableFeatures:
    def test_same_distribution_is_stable(self) -> None:
        rng = random.Random(7)
        train = [rng.gauss(10, 2) for _ in range(500)]
        serve = [rng.gauss(10, 2) for _ in range(500)]
        drifts, _, _ = drift.compare(
            columns(numeric_rows("x", train)), columns(numeric_rows("x", serve))
        )
        assert drifts[0].verdict is Verdict.STABLE

    def test_same_categories_are_stable(self) -> None:
        rows = [{"c": "a" if i % 3 else "b"} for i in range(300)]
        drifts, _, _ = drift.compare(columns(rows), columns(list(rows)))
        assert drifts[0].verdict is Verdict.STABLE
        assert drifts[0].psi == 0.0


class TestNumericDrift:
    def test_large_mean_shift_is_severe_with_a_note_in_sd_units(self) -> None:
        rng = random.Random(7)
        train = [rng.gauss(10, 2) for _ in range(500)]
        serve = [rng.gauss(16, 2) for _ in range(500)]
        drifts, _, _ = drift.compare(
            columns(numeric_rows("x", train)), columns(numeric_rows("x", serve))
        )
        feature = drifts[0]
        assert feature.verdict is Verdict.SEVERE
        assert feature.psi > drift.PSI_SEVERE
        assert any("mean moved" in note and "sd" in note for note in feature.notes)

    def test_mild_shift_lands_in_drifting(self) -> None:
        rng = random.Random(11)
        train = [rng.gauss(10, 2) for _ in range(2000)]
        serve = [v + 0.8 for v in train]
        drifts, _, _ = drift.compare(
            columns(numeric_rows("x", train)), columns(numeric_rows("x", serve))
        )
        assert drifts[0].verdict is Verdict.DRIFTING


class TestCategoricalDrift:
    def test_unseen_category_is_at_least_drifting_and_named(self) -> None:
        train = [{"c": "a"} for _ in range(95)] + [{"c": "b"} for _ in range(5)]
        serve = [{"c": "a"} for _ in range(90)] + [{"c": "NEW"} for _ in range(10)]
        drifts, _, _ = drift.compare(columns(train), columns(serve))
        feature = drifts[0]
        assert feature.unseen_categories == ("NEW",)
        assert feature.verdict.rank >= Verdict.DRIFTING.rank
        assert any("never seen in training" in note for note in feature.notes)

    def test_rebalanced_categories_score_by_tv_distance(self) -> None:
        train = [{"c": "a"} for _ in range(75)] + [{"c": "b"} for _ in range(25)]
        serve = [{"c": "a"} for _ in range(25)] + [{"c": "b"} for _ in range(75)]
        drifts, _, _ = drift.compare(columns(train), columns(serve))
        assert drifts[0].distance == 0.5
        assert drifts[0].verdict is Verdict.SEVERE


class TestMissingness:
    def test_missing_rate_jump_is_severe_even_when_values_match(self) -> None:
        train = [{"x": str(i % 10)} for i in range(100)]
        serve = [{"x": str(i % 10) if i % 2 else ""} for i in range(100)]
        drifts, _, _ = drift.compare(columns(train), columns(serve))
        feature = drifts[0]
        assert feature.missing_rate_delta > drift.MISSING_DELTA_SEVERE
        assert feature.verdict is Verdict.SEVERE
        assert any("missing rate" in note for note in feature.notes)


class TestSchemaChanges:
    def test_disjoint_columns_are_reported_separately(self) -> None:
        drifts, only_train, only_serve = drift.compare(
            columns([{"a": "1", "b": "x"}]), columns([{"a": "1", "c": "y"}])
        )
        assert [d.name for d in drifts] == ["a"]
        assert only_train == ["b"]
        assert only_serve == ["c"]

    def test_type_disagreement_compares_as_categories_with_a_note(self) -> None:
        train = numeric_rows("x", [1.0, 2.0, 3.0])
        serve = [{"x": "one"}, {"x": "two"}]
        drifts, _, _ = drift.compare(columns(train), columns(serve))
        feature = drifts[0]
        assert any("compared as categories" in note for note in feature.notes)
        assert feature.unseen_categories == ("one", "two")


class TestOrdering:
    def test_worst_verdict_sorts_first(self) -> None:
        train = [{"stable": "a", "moved": "a"} for _ in range(50)]
        serve = [{"stable": "a", "moved": "b"} for _ in range(50)]
        drifts, _, _ = drift.compare(columns(train), columns(serve))
        assert [d.name for d in drifts] == ["moved", "stable"]
