from __future__ import annotations

import math

import pytest

from feature_drift import stats


class TestQuantileEdges:
    def test_uniform_range_gets_evenly_spread_edges(self) -> None:
        edges = stats.quantile_edges([float(i) for i in range(1, 101)])
        assert len(edges) == 9
        assert edges == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

    def test_heavily_tied_data_collapses_duplicate_edges(self) -> None:
        edges = stats.quantile_edges([1.0] * 95 + [2.0] * 5)
        assert edges == [1.0]

    def test_empty_input_yields_no_edges(self) -> None:
        assert stats.quantile_edges([]) == []


class TestBucketShares:
    def test_shares_sum_to_one(self) -> None:
        values = [float(i) for i in range(100)]
        shares = stats.bucket_shares(values, stats.quantile_edges(values))
        assert sum(shares) == pytest.approx(1.0)

    def test_values_beyond_the_last_edge_land_in_the_last_bucket(self) -> None:
        shares = stats.bucket_shares([100.0], [1.0, 2.0])
        assert shares == [0.0, 0.0, 1.0]

    def test_boundary_values_go_left(self) -> None:
        # <= edge goes into the bucket the edge closes, matching the
        # nearest-rank quantile construction.
        shares = stats.bucket_shares([1.0, 2.0], [1.0])
        assert shares == [0.5, 0.5]


class TestPsi:
    def test_identical_distributions_score_zero(self) -> None:
        values = [float(i % 10) for i in range(1000)]
        assert stats.numeric_psi(values, list(values)) == pytest.approx(0.0, abs=1e-9)

    def test_textbook_hand_computation(self) -> None:
        # Two buckets, 50/50 vs 70/30:
        # PSI = (0.7-0.5)ln(0.7/0.5) + (0.3-0.5)ln(0.3/0.5) ≈ 0.169457
        value = stats.psi_from_shares([0.5, 0.5], [0.7, 0.3])
        assert value == pytest.approx(0.169457, abs=1e-5)

    def test_shift_grows_monotonically(self) -> None:
        train = [float(i) for i in range(1000)]
        small = [v + 50 for v in train]
        large = [v + 400 for v in train]
        assert stats.numeric_psi(train, large) > stats.numeric_psi(train, small) > 0

    def test_empty_bucket_stays_finite(self) -> None:
        value = stats.psi_from_shares([1.0, 0.0], [0.0, 1.0])
        assert value == pytest.approx(2 * 0.9999 * (0.6931 + 9.2103), rel=0.5)
        assert value < 25

    def test_categorical_counts_unseen_categories(self) -> None:
        drifted = stats.categorical_psi(["a"] * 100, ["a"] * 50 + ["b"] * 50)
        assert drifted > stats.PSI_BUCKETS * 0  # sanity
        assert drifted > 1.0


class TestKs:
    def test_identical_samples_score_zero(self) -> None:
        values = [float(i) for i in range(50)]
        assert stats.ks_statistic(values, list(values)) == pytest.approx(0.0)

    def test_disjoint_samples_score_one(self) -> None:
        assert stats.ks_statistic([1.0, 2.0], [10.0, 11.0]) == pytest.approx(1.0)

    def test_known_half_overlap(self) -> None:
        # a = {1,2}, b = {2,3}: max CDF gap is 0.5 at value 1.
        assert stats.ks_statistic([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.5)

    def test_symmetry(self) -> None:
        a = [1.0, 5.0, 9.0, 12.0]
        b = [2.0, 5.0, 30.0]
        assert stats.ks_statistic(a, b) == pytest.approx(stats.ks_statistic(b, a))

    def test_empty_side_scores_zero(self) -> None:
        assert stats.ks_statistic([], [1.0]) == 0.0


class TestTotalVariation:
    def test_identical_distributions_score_zero(self) -> None:
        assert stats.total_variation(["a", "b"], ["b", "a"]) == pytest.approx(0.0)

    def test_disjoint_categories_score_one(self) -> None:
        assert stats.total_variation(["a"], ["b"]) == pytest.approx(1.0)

    def test_known_hand_computation(self) -> None:
        # train: a=0.75, b=0.25; serve: a=0.25, b=0.75 → TV = 0.5
        value = stats.total_variation(["a", "a", "a", "b"], ["a", "b", "b", "b"])
        assert value == pytest.approx(0.5)


class TestSummary:
    def test_mean_and_population_sd(self) -> None:
        mean, sd = stats.summary([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert mean == pytest.approx(5.0)
        assert sd == pytest.approx(2.0)

    def test_empty_is_zeroes(self) -> None:
        assert stats.summary([]) == (0.0, 0.0)


# --- audit: hand-computed reference values --------------------------------


def test_ks_matches_a_hand_computed_value() -> None:
    # train {1,2,3,4}, serve {3,4,5,6}: CDF gap is largest at x=2 (0.5 - 0)
    # and x=4 (1.0 - 0.5) -> 0.5. scipy.stats.ks_2samp gives 0.5 too.
    assert stats.ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == 0.5


def test_total_variation_matches_a_hand_computed_value() -> None:
    # train a:0.5 b:0.5, serve a:0.25 b:0.25 c:0.5 -> 0.5*(0.25+0.25+0.5)=0.5
    assert stats.total_variation(["a", "b"], ["a", "b", "c", "c"]) == 0.5


def test_categorical_psi_matches_a_hand_computed_value() -> None:
    # 50/50 -> 25/75: 0.25*ln(0.25/0.5)*-1 + 0.25*ln(0.75/0.5) = 0.25*ln(3)
    value = stats.categorical_psi(["a", "b"], ["a", "b", "b", "b"])
    assert abs(value - 0.25 * math.log(3)) < 1e-12
