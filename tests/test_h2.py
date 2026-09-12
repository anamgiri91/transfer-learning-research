"""The H2 analysis, fault-injected.

`plan.md` §6 decides H2 by an interaction term that was never run until
2026-09-11, and the correction it forced — H2 is *unanswered* for the frozen
probe rather than answered negatively — is now load-bearing in the abstract,
§5.3, §8.1 and §9. A statistic that carries that much has to be shown to move
in the right direction on data whose answer is known, not merely to run.

Every test below plants a curve with a known sign and asserts the estimator
recovers it, or plants a defect and asserts it is caught.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "scripts")
_spec = importlib.util.spec_from_file_location("analyse_h2", "scripts/analyse_h2.py")
h2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2)

SIZES = h2.SIZES
BASE = h2.BASELINE
# der() walks the real arm list, so fixtures borrow a real id.
ARM = "T1_chemberta_linear_probe"


def curves(arm_rmse: dict[int, list[float]], base_rmse: dict[int, list[float]]):
    """Build the arm -> seed -> n -> rmse structure `analyse_h2` consumes."""
    n_seeds = len(next(iter(arm_rmse.values())))
    M = {ARM: {}, BASE: {}}
    for s in range(n_seeds):
        M[ARM][s] = {n: arm_rmse[n][s] for n in SIZES}
        M[BASE][s] = {n: base_rmse[n][s] for n in SIZES}
    return M


def flat(v: float, n_seeds: int = 10):
    return {n: [v] * n_seeds for n in SIZES}


# --------------------------------------------------------------------------
# The estimator recovers a known sign
# --------------------------------------------------------------------------

def test_an_arm_whose_deficit_shrinks_as_data_shrinks_gives_a_positive_slope():
    """H2's prediction, planted. Arm is far behind at full data, level at n=50."""
    rng = np.random.default_rng(0)
    base = {n: [0.60 + rng.normal(0, 0.002) for _ in range(10)] for n in SIZES}
    # deficit grows with n: 0.00 at n=50 up to 0.15 at n=347
    deficit = dict(zip(SIZES, [0.0, 0.05, 0.12, 0.15]))
    arm = {n: [b + deficit[n] for b in base[n]] for n in SIZES}
    seeds = list(range(10))
    slopes = h2.per_seed_slopes(curves(arm, base), ARM, seeds)
    assert (slopes > 0).all(), slopes
    assert np.median(slopes) > 0.04


def test_an_arm_whose_deficit_grows_as_data_shrinks_gives_a_negative_slope():
    """The fine-tune's observed shape: catastrophic at n=50, close at full data."""
    rng = np.random.default_rng(1)
    base = {n: [0.60 + rng.normal(0, 0.002) for _ in range(10)] for n in SIZES}
    deficit = dict(zip(SIZES, [0.60, 0.30, 0.08, 0.05]))
    arm = {n: [b + deficit[n] for b in base[n]] for n in SIZES}
    slopes = h2.per_seed_slopes(curves(arm, base), ARM, list(range(10)))
    assert (slopes < 0).all(), slopes


def test_an_arm_with_a_constant_offset_gives_a_zero_slope():
    """The null H2 is tested against: worse everywhere, by the same amount.

    This is the case the DER cannot distinguish from the two above -- all three
    have DER = 0 if the offset never closes -- which is the whole reason the
    interaction term is not optional.
    """
    rng = np.random.default_rng(2)
    base = {n: [0.60 + rng.normal(0, 0.002) for _ in range(10)] for n in SIZES}
    arm = {n: [b + 0.05 for b in base[n]] for n in SIZES}
    slopes = h2.per_seed_slopes(curves(arm, base), ARM, list(range(10)))
    assert np.allclose(slopes, 0.0, atol=1e-12), slopes


def test_the_three_shapes_are_indistinguishable_by_DER_and_separated_by_the_slope():
    """The claim §5.3 makes, asserted rather than argued.

    Shrinking, growing and constant deficits all never reach the baseline's
    full-data score, so all three score DER = 0. Only the slope tells them
    apart.
    """
    base = flat(0.60)
    shapes = {
        "shrinking": dict(zip(SIZES, [0.02, 0.05, 0.12, 0.15])),
        "growing": dict(zip(SIZES, [0.60, 0.30, 0.08, 0.05])),
        "constant": dict(zip(SIZES, [0.05, 0.05, 0.05, 0.05])),
    }
    ders, signs = {}, {}
    for name, deficit in shapes.items():
        arm = {n: [b + deficit[n] for b in base[n]] for n in SIZES}
        M = curves(arm, base)
        ders[name] = h2.n_to_reach([float(n) for n in SIZES],
                                   [M[ARM][0][n] for n in SIZES], target=0.60)
        signs[name] = float(np.median(h2.per_seed_slopes(M, ARM, list(range(10)))))
    assert all(np.isinf(v) for v in ders.values()), ders      # every DER censored
    assert signs["shrinking"] > 0 > signs["growing"]
    assert abs(signs["constant"]) < 1e-12


# --------------------------------------------------------------------------
# Pairing, censoring and coverage
# --------------------------------------------------------------------------

def test_pairing_is_preserved_so_a_shared_shift_cancels():
    """A seed that is hard for BOTH arms must not register as an interaction.

    If the delta were not taken within a seed, per-seed difficulty would leak
    into the slope. Here every seed gets a large random level shift applied to
    both arms; the slope must be unmoved.
    """
    rng = np.random.default_rng(3)
    shift = rng.normal(0, 0.3, size=10)
    base = {n: list(0.60 + shift) for n in SIZES}
    deficit = dict(zip(SIZES, [0.0, 0.05, 0.12, 0.15]))
    arm = {n: list(0.60 + shift + deficit[n]) for n in SIZES}
    slopes = h2.per_seed_slopes(curves(arm, base), ARM, list(range(10)))
    assert np.allclose(slopes, slopes[0]), slopes     # identical across seeds


def test_a_seed_missing_a_size_is_dropped_not_silently_interpolated():
    """An incomplete curve must leave the seed out, not fabricate a point."""
    base, arm = flat(0.60), flat(0.65)
    M = curves(arm, base)
    del M[ARM][3][250]
    seeds = h2.complete_seeds(M, ARM, "scaffold")
    assert 3 not in seeds and len(seeds) == 9


def test_a_censored_curve_is_counted_as_censored_not_scored_as_zero():
    """The correction §5.3 makes to table2: never-reaching is not DER = 0."""
    base, arm = flat(0.60), flat(0.90)          # arm never reaches 0.60
    out = h2.der(curves(arm, base), "scaffold")
    row = out[out.arm == ARM].iloc[0]
    assert row.seeds_reaching_target == 0
    assert row.seeds_censored == 10
    assert np.isnan(row.median_der_where_defined)   # not 0.0


def test_a_partially_censored_arm_reports_both_halves():
    """The real shape: some seeds cross, most do not. Both numbers must show."""
    base = flat(0.60)
    arm = {n: [0.50 if s < 4 else 0.90 for s in range(10)] for n in SIZES}
    out = h2.der(curves(arm, base), "scaffold")
    row = out[out.arm == ARM].iloc[0]
    assert row.seeds_reaching_target == 4 and row.seeds_censored == 6
    assert not np.isnan(row.median_der_where_defined)


def test_non_monotonic_curves_are_counted_because_the_threshold_read_depends_on_them():
    rng = np.random.default_rng(4)
    base = {n: [0.60 + rng.normal(0, 0.001) for _ in range(10)] for n in SIZES}
    arm = {50: [0.80] * 10, 100: [0.70] * 10, 250: [0.55] * 10, 347: [0.65] * 10}
    out = h2.der(curves(arm, base), "scaffold")
    assert out[out.arm == ARM].iloc[0].curves_non_monotonic == 10


# --------------------------------------------------------------------------
# The committed result
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["scaffold", "random", "butina"])
def test_the_probes_H2_verdict_is_inconclusive_on_every_split(split):
    """The correction this analysis forced, pinned as a regression test.

    If a future change makes T1's interaction significant, that is a finding
    and this test should fail loudly rather than let the abstract go stale.
    """
    import pandas as pd
    df = pd.read_csv("results/tables/table15_h2_interaction.csv", comment="#")
    row = df[(df.split == split) & (df.arm == "T1_chemberta_linear_probe")]
    assert len(row) == 1
    assert row.iloc[0].h2_verdict == "inconclusive", row.iloc[0].to_dict()


def test_B0_is_the_sanity_check_and_has_the_expected_sign():
    """A constant predictor must lose relatively less ground as n falls.

    If this ever goes negative the estimator is wrong, not the world.
    """
    import pandas as pd
    df = pd.read_csv("results/tables/table15_h2_interaction.csv", comment="#")
    row = df[(df.split == "scaffold") & (df.arm == "B0_median")].iloc[0]
    assert row.median_slope > 0 and row.p_holm <= 0.05


def test_no_committed_DER_is_distinguishable_from_one():
    """§5.3 states this; it must stay true or the sentence must change."""
    import pandas as pd
    df = pd.read_csv("results/tables/table16_der_uncertainty.csv", comment="#")
    d = df[(df.split == "scaffold") & (df.arm != "B1_ecfp_histgb")
           & df.der_ci_lo.notna()]
    assert len(d) >= 2
    assert ((d.der_ci_lo <= 1.0) & (d.der_ci_hi >= 1.0)).all(), d.to_dict("records")


def test_the_scaffold_scoping_of_that_claim_is_load_bearing():
    """...and is false unscoped, which is why §5.3 names the split.

    B2's per-seed DER on the random split is [1.14, 3.78], excluding 1. An
    earlier draft of §5.3 said "no DER in this study" without qualification;
    this test exists so that sentence cannot drift back.
    """
    import pandas as pd
    df = pd.read_csv("results/tables/table16_der_uncertainty.csv", comment="#")
    row = df[(df.split == "random") & (df.arm == "B2_descriptors_rf")].iloc[0]
    assert row.der_ci_lo > 1.0
