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


def test_wilcoxon_can_detect_tiny_consistent_deltas():
    """A 1e-9 but perfectly consistent improvement is 'significant'.

    Wilcoxon combines signs with ranks of absolute differences, and is invariant
    to positive rescaling that preserves those ranks. This is why the effect size is
    reported next to every p-value -- significance alone is not evidence of a
    difference that matters.
    """
    rng = np.random.default_rng(0)
    base = list(rng.normal(1.0, 0.05, 10))
    res = {r.arm: r for r in paired_compare({"B1": base, "T_tiny": [b - 1e-9 for b in base]},
                                            baseline="B1")}["T_tiny"]
    assert res.p_raw < 0.05                       # significant ...
    assert abs(res.median_delta) < 1e-6           # ... but the effect is tiny


def test_paired_compare_rejects_mismatched_seed_counts():
    with pytest.raises(ValueError, match="seeds"):
        paired_compare({"B1": [1.0, 2.0, 3.0], "T1": [1.0]}, baseline="B1")


def test_precision_bounds_collapse_without_ties():
    from evapro.evaluation.metrics import precision_at_k_frac, precision_at_k_frac_bounds

    rng = np.random.default_rng(0)
    y, yhat = rng.normal(size=100), rng.normal(size=100)
    lo, hi = precision_at_k_frac_bounds(y, yhat)
    assert lo == hi == precision_at_k_frac(y, yhat)


def test_precision_bounds_span_a_tie_at_the_cutoff():
    from evapro.evaluation.metrics import precision_at_k_frac, precision_at_k_frac_bounds

    # k = 2. The top prediction is a hit; a hit and a miss tie for the last slot.
    y = np.array([9.0, 8.0, 1.0, 0.0] + [0.5] * 16)
    yhat = np.array([5.0, 3.0, 3.0 + 4e-7, 0.0] + [0.1] * 16)
    lo, hi = precision_at_k_frac_bounds(y, yhat)
    assert (lo, hi) == (0.5, 1.0)
    assert lo <= precision_at_k_frac(y, yhat) <= hi
    # A gap wider than eps is not a tie.
    yhat[2] = 3.0 + 1e-3
    assert precision_at_k_frac_bounds(y, yhat) == (0.5, 0.5)


def test_precision_bounds_span_a_tie_in_the_true_top_set():
    """The true top-k set is also chosen by argsort, so label ties matter too."""
    from evapro.evaluation.metrics import precision_at_k_frac_bounds

    # k = 1. One prediction is clearly top; two labels tie for the best and
    # only one of them is the predicted winner, so precision is 0 or 1. An
    # exact label tie is resolved both ways whatever eps_y is.
    y = np.array([5.0, 5.0] + [1.0] * 8)
    yhat = np.array([9.0, 2.0] + [1.0] * 8)
    assert precision_at_k_frac_bounds(y, yhat, frac=0.1) == (0.0, 1.0)
    assert precision_at_k_frac_bounds(y, yhat, frac=0.1, eps_y=1e-5) == (0.0, 1.0)

    # A label gap narrower than eps_y is a tie only once eps_y covers it.
    y[1] = 5.0 - 4e-7
    assert precision_at_k_frac_bounds(y, yhat, frac=0.1) == (1.0, 1.0)
    assert precision_at_k_frac_bounds(y, yhat, frac=0.1, eps_y=1e-5) == (0.0, 1.0)


def test_precision_bounds_reject_a_negative_label_tolerance():
    from evapro.evaluation.metrics import precision_at_k_frac_bounds

    with pytest.raises(ValueError):
        precision_at_k_frac_bounds([1.0, 2.0], [1.0, 2.0], eps_y=-1.0)
