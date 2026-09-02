"""Tests for the activity-cliff stratification (plan.md §7.2).

The stratification carries a real claim in §6.3, so its edge cases are pinned
here rather than trusted: in particular that `distant` is kept separate from
`smooth`, which is the whole reason the three-way split exists.
"""
import numpy as np
import pytest

from evapro.evaluation.cliffs import CLIFF, DISTANT, SMOOTH, cliff_census, stratify


def _sim(pairs, n):
    """Similarity matrix with 1.0 on the diagonal and `pairs` filled in."""
    s = np.eye(n)
    for i, j, v in pairs:
        s[i, j] = s[j, i] = v
    return s


def test_close_neighbour_with_big_label_gap_is_a_cliff():
    sim = _sim([(0, 1, 0.9)], 2)
    y = np.array([5.0, 7.0])           # gap 2.0 > delta
    assert stratify([0], [1], sim, y, threshold=0.7)[0] == CLIFF


def test_close_neighbour_with_small_label_gap_is_smooth():
    sim = _sim([(0, 1, 0.9)], 2)
    y = np.array([5.0, 5.5])           # gap 0.5 <= delta
    assert stratify([0], [1], sim, y, threshold=0.7)[0] == SMOOTH


def test_no_close_neighbour_is_distant_however_large_the_label_gap():
    """The case the three-way split exists for: a big label gap to a *dissimilar*
    training compound is not a cliff, it is simply extrapolation."""
    sim = _sim([(0, 1, 0.2)], 2)
    y = np.array([5.0, 7.0])
    assert stratify([0], [1], sim, y, threshold=0.7)[0] == DISTANT


def test_one_cliff_neighbour_outranks_many_smooth_ones():
    sim = _sim([(0, 1, 0.9), (0, 2, 0.9), (0, 3, 0.8)], 4)
    y = np.array([5.0, 5.1, 5.2, 7.0])
    assert stratify([0], [1, 2, 3], sim, y, threshold=0.7)[0] == CLIFF


def test_threshold_is_inclusive_and_delta_is_strict():
    sim = _sim([(0, 1, 0.7)], 2)
    y = np.array([5.0, 6.0])           # gap exactly 1.0, not > 1.0
    assert stratify([0], [1], sim, y, threshold=0.7, delta=1.0)[0] == SMOOTH
    y = np.array([5.0, 6.001])
    assert stratify([0], [1], sim, y, threshold=0.7, delta=1.0)[0] == CLIFF


def test_only_training_compounds_can_make_a_test_compound_a_cliff():
    """Two test compounds forming a cliff with each other must not count: the
    model never saw either label."""
    sim = _sim([(0, 1, 0.95), (0, 2, 0.3)], 3)
    y = np.array([5.0, 7.0, 5.0])
    strata = stratify([0, 1], [2], sim, y, threshold=0.7)
    assert list(strata) == [DISTANT, DISTANT]


def test_every_test_compound_lands_in_exactly_one_stratum():
    rng = np.random.default_rng(0)
    n = 40
    s = rng.random((n, n)); s = (s + s.T) / 2; np.fill_diagonal(s, 1.0)
    y = rng.normal(5, 1, n)
    strata = stratify(range(10), range(10, n), s, y, threshold=0.7)
    assert len(strata) == 10
    assert set(strata) <= {CLIFF, SMOOTH, DISTANT}


def test_census_counts_unordered_pairs_once():
    sim = _sim([(0, 1, 0.9)], 3)
    y = np.array([5.0, 7.0, 5.0])
    c = cliff_census(sim, y, threshold=0.7, delta=1.0)
    assert c["cliff_pairs"] == 1
    assert c["compounds_in_a_cliff"] == 2
    assert c["n_compounds"] == 3


def test_census_similar_pairs_is_a_superset_of_cliff_pairs():
    rng = np.random.default_rng(1)
    n = 30
    s = rng.random((n, n)); s = (s + s.T) / 2; np.fill_diagonal(s, 1.0)
    y = rng.normal(5, 1, n)
    c = cliff_census(s, y, threshold=0.5, delta=1.0)
    assert c["cliff_pairs"] <= c["similar_pairs"]
    assert 0.0 <= c["cliff_fraction_of_similar"] <= 1.0
