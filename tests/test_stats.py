"""Tests for the learning-curve / DER machinery (plan.md §6).

These run without RDKit, so the paper's headline statistic is checkable even on
a bare install.
"""
import numpy as np
import pytest

from evapro.evaluation.stats import data_efficiency_ratio, n_to_reach, paired_compare


def test_n_to_reach_interpolates_between_points():
    # RMSE falls 1.0 -> 0.8 -> 0.6 at n = 100, 200, 400.
    assert n_to_reach([100, 200, 400], [1.0, 0.8, 0.6], target=0.9) == pytest.approx(150.0)


def test_n_to_reach_is_inf_when_target_never_met():
    assert n_to_reach([100, 200], [1.0, 0.9], target=0.5) == float("inf")


def test_der_above_one_when_transfer_needs_less_data():
    baseline = ([100, 200, 400], [1.0, 0.9, 0.8])
    transfer = ([100, 200, 400], [0.9, 0.8, 0.7])
    der = data_efficiency_ratio(baseline, transfer, target=0.8)
    assert der > 1.0, "transfer reaching the target sooner must give DER > 1"


def test_der_is_one_for_identical_curves():
    curve = ([100, 200, 400], [1.0, 0.9, 0.8])
    assert data_efficiency_ratio(curve, curve, target=0.85) == pytest.approx(1.0)


def test_paired_compare_applies_holm_correction():
    rng = np.random.default_rng(0)
    base = list(rng.normal(1.0, 0.05, 10))
    noise = rng.normal(0.0, 0.05, 10)
    scores = {
        "B1": base,
        "T_better": [b - 0.5 for b in base],        # clearly better
        "T_noise": [b + n for b, n in zip(base, noise)],   # genuinely indistinguishable
    }
    results = {r.arm: r for r in paired_compare(scores, baseline="B1")}
    assert results["T_better"].median_delta > 0
    assert results["T_better"].p_holm >= results["T_better"].p_raw
    assert results["T_noise"].inconclusive


def test_wilcoxon_is_sign_based_so_tiny_consistent_deltas_are_significant():
    """A 1e-9 but perfectly consistent improvement is 'significant'.

    Wilcoxon signed-rank ranks signs, not magnitudes, so consistency alone
    drives p. This is precisely why plan.md §6 requires the effect size to be
    reported next to every p-value -- significance alone is not evidence of a
    difference that matters.
    """
    rng = np.random.default_rng(0)
    base = list(rng.normal(1.0, 0.05, 10))
    res = {r.arm: r for r in paired_compare({"B1": base, "T_tiny": [b - 1e-9 for b in base]},
                                            baseline="B1")}["T_tiny"]
    assert res.p_raw < 0.05                       # significant ...
    assert abs(res.median_delta) < 1e-6           # ... but the effect is nil


def test_paired_compare_rejects_mismatched_seed_counts():
    with pytest.raises(ValueError, match="seeds"):
        paired_compare({"B1": [1.0, 2.0, 3.0], "T1": [1.0]}, baseline="B1")
