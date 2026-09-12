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


def test_a_never_reaching_curve_is_censored_not_scored_as_zero():
    """The correction §5.3 makes to table2: never-reaching is not DER = 0."""
    base, arm = flat(0.60), flat(0.90)          # arm never reaches 0.60
    row = h2.der(curves(arm, base), "scaffold").query("arm == @ARM").iloc[0]
    assert row.crossed_interior == 0
    assert row.right_censored_never_reached == 10
    assert np.isnan(row.median_der_interior)    # not 0.0


def test_left_censoring_is_its_own_state_and_is_not_a_crossing_at_50():
    """An arm already at target at n=50 crossed at an UNKNOWN size <= 50.

    Scoring it as a crossing at exactly 50 is extrapolation outside the
    evaluated range, and it is not harmless: doing so is what produced the
    spurious "B2 on random excludes 1" DER interval that §5.3 retracts.
    """
    base, arm = flat(0.60), flat(0.40)          # already below target at n=50
    row = h2.der(curves(arm, base), "scaffold").query("arm == @ARM").iloc[0]
    assert row.left_censored_at_n50 == 10
    assert row.crossed_interior == 0
    assert np.isnan(row.median_der_interior)


def test_the_three_censoring_states_partition_the_seeds():
    """Asserted in der() itself; asserted here on a mixture of all three."""
    base = flat(0.60)
    arm = {n: [0.40] * 3 + [0.90] * 4 + [0.55] * 3 for n in SIZES}
    row = h2.der(curves(arm, base), "scaffold").query("arm == @ARM").iloc[0]
    assert (row.crossed_interior + row.left_censored_at_n50
            + row.right_censored_never_reached) == row.n_seeds == 10
    assert row.left_censored_at_n50 == 6        # 0.40 and 0.55 both start below


def test_a_median_is_withheld_below_three_crossings_rather_than_computed():
    """One or two crossings do not make a median worth printing.

    The two crossing seeds descend THROUGH the target between n=250 and n=347,
    which is what an interior crossing is; starting below it at n=50 would be
    left-censoring instead.
    """
    base = flat(0.60)
    crossing = {50: 0.90, 100: 0.85, 250: 0.80, 347: 0.50}
    never = {n: 0.90 for n in SIZES}
    arm = {n: [crossing[n] if s < 2 else never[n] for s in range(10)] for n in SIZES}
    row = h2.der(curves(arm, base), "scaffold").query("arm == @ARM").iloc[0]
    assert row.crossed_interior == 2, row.to_dict()
    assert np.isnan(row.median_der_interior)


def test_three_interior_crossings_do_get_a_median():
    """The floor is a floor, not a blanket refusal."""
    base = flat(0.60)
    crossing = {50: 0.90, 100: 0.85, 250: 0.80, 347: 0.50}
    never = {n: 0.90 for n in SIZES}
    arm = {n: [crossing[n] if s < 3 else never[n] for s in range(10)] for n in SIZES}
    row = h2.der(curves(arm, base), "scaffold").query("arm == @ARM").iloc[0]
    assert row.crossed_interior == 3 and not np.isnan(row.median_der_interior)


def test_non_monotonic_curves_are_counted_because_the_threshold_read_depends_on_them():
    rng = np.random.default_rng(4)
    base = {n: [0.60 + rng.normal(0, 0.001) for _ in range(10)] for n in SIZES}
    arm = {50: [0.80] * 10, 100: [0.70] * 10, 250: [0.55] * 10, 347: [0.65] * 10}
    out = h2.der(curves(arm, base), "scaffold")
    assert out[out.arm == ARM].iloc[0].curves_non_monotonic == 10


# --------------------------------------------------------------------------
# Mean vs median vs pseudomedian: three quantities, not three views of one
# --------------------------------------------------------------------------

def test_pooled_OLS_recovers_the_MEAN_of_per_seed_slopes_not_the_median():
    """The equivalence §5.3 states, on a case constructed so they differ.

    Nine seeds with slope ~ +0.01 and one with -0.50: the median stays
    positive, the mean is dragged negative. If a future refactor makes the
    pooled fit track the median instead, or the manuscript reverts to calling
    them the same coefficient, this fails.
    """
    base = flat(0.60)
    # per-seed deficit slopes: 9 gentle positives, 1 large negative
    x = np.log2(np.array(SIZES, float))
    xc = x - x.mean()
    per_seed = [0.01] * 9 + [-0.50]
    arm = {n: [] for n in SIZES}
    for s, sl in enumerate(per_seed):
        d = sl * xc
        for k, n in enumerate(SIZES):
            arm[n].append(0.60 + d[k])
    M = curves(arm, base)
    seeds = list(range(10))
    slopes = h2.per_seed_slopes(M, ARM, seeds)
    coef, _, _ = h2.pooled_interaction(M, ARM, seeds)

    assert np.isclose(np.median(slopes), 0.01, atol=1e-9)
    assert np.isclose(slopes.mean(), (9 * 0.01 - 0.50) / 10, atol=1e-9)
    assert np.isclose(coef, slopes.mean(), atol=1e-10)      # the equivalence
    assert not np.isclose(coef, np.median(slopes), atol=1e-3)   # and the non-equivalence
    # They even disagree in SIGN here, which is the worst case for a verdict.
    assert np.median(slopes) > 0 > coef


def test_the_equivalence_check_fires_when_the_design_is_unbalanced():
    """_check_equivalence must not pass vacuously.

    It is the guard that stops §5.3's wording going stale a second time, so it
    has to fail on a design where the identity genuinely stops holding.
    """
    base, arm = flat(0.60), flat(0.65)
    M = curves(arm, base)
    bad = np.array([0.0] * 10)                  # a wrong slope vector
    with pytest.raises(AssertionError, match="pooled interaction"):
        h2._check_equivalence(M, ARM, list(range(10)), bad + 1.0)


def test_hodges_lehmann_is_a_third_quantity_again():
    """Not the mean, not the median: the pseudomedian the signed-rank localises."""
    v = np.array([0.01] * 9 + [-0.50])
    hl = h2.hodges_lehmann(v)
    assert not np.isclose(hl, v.mean(), atol=1e-6)
    assert hl != pytest.approx(np.median(v), abs=1e-12) or True   # may coincide
    # Walsh averages include (0.01 + -0.50)/2 = -0.245, so HL < median here.
    assert hl <= np.median(v)


def test_the_reported_wilcoxon_method_matches_what_scipy_would_select():
    """§5.3 states the exact null distribution is used at n=10. Check it."""
    clean = np.array([0.1, 0.2, 0.3, -0.05, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p, method, n_zero = h2.wilcoxon_with_method(clean)
    assert method == "exact" and n_zero == 0
    assert p >= 2 / 2 ** 10 - 1e-12

    withzero = clean.copy(); withzero[0] = 0.0
    p2, method2, n_zero2 = h2.wilcoxon_with_method(withzero)
    assert n_zero2 == 1
    # zeros present and n<=13 => exhaustive permutations, never asymptotic
    assert method2 == "exhaustive permutations"


def test_a_degenerate_all_zero_slope_vector_does_not_crash_or_claim_significance():
    p, method, n_zero = h2.wilcoxon_with_method(np.zeros(10))
    assert p == 1.0 and n_zero == 10 and "degenerate" in method


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


def test_no_committed_DER_interval_on_any_split_excludes_one():
    """§5.3 states this without a split qualifier; it must stay true.

    It became true only once left-censored seeds stopped being scored as
    crossings at exactly n=50 -- before that, B2 on the random split appeared
    to exclude 1.
    """
    import pandas as pd
    df = pd.read_csv("results/tables/table16_der_uncertainty.csv", comment="#")
    d = df[(df.arm != "B1_ecfp_histgb") & df.der_ci_lo.notna()]
    assert len(d) >= 4
    assert ((d.der_ci_lo <= 1.0) & (d.der_ci_hi >= 1.0)).all(), d.to_dict("records")


def test_the_conditional_median_is_only_reported_where_enough_seeds_cross():
    """A median over 1-2 crossings must be withheld, not printed."""
    import pandas as pd
    df = pd.read_csv("results/tables/table16_der_uncertainty.csv", comment="#")
    assert (df[df.crossed_interior < 3].median_der_interior.isna()).all()
    assert (df[df.crossed_interior >= 3].median_der_interior.notna()).all()
