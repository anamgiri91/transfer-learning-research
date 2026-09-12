#!/usr/bin/env python
"""H2: does the transfer advantage grow as the training set shrinks?

`plan.md` §6 decides H2 by two things: "Interaction term in the learning-curve
model; DER". Only the DER half was ever run, and DER alone cannot answer H2 --
it asks whether an arm ever crosses a fixed threshold, which is a question
about the *full-data* end of the curve. An arm can be worse at full data and
still lose less ground at n = 50; nothing in DER = 0 rules that out, and this
study imposes no monotonicity assumption that would make it do so. This script
runs the half that was missing and puts uncertainty on the half that was not.

DISCLOSURE -- modelling choices made after results existed
----------------------------------------------------------
`plan.md` names an "interaction term in the learning-curve model" and stops
there: no model form, no error structure, no test. Everything below is a
post-hoc specification of a pre-registered intent, recorded as such in
docs/decision-log.md (2026-09-11) and as manuscript §7 limitation 11. Nothing
here was chosen by which version produced significance; §5.3 reports that the
two estimators disagree by a factor of seven on the point estimate and agree
on the verdict, which is the only reason the choice does not matter here.

  * Outcome is the PAIRED delta d(arm, seed, n) = RMSE_arm - RMSE_B1, positive
    meaning the arm is worse. Pairing is preserved at both levels a benchmark
    can break it: within a seed both arms see the same split AND the same
    training subsample, and every seed contributes all four sizes.
  * Size enters as log2(n). Learning curves are conventionally near-linear in
    log n, and the four sizes (50, 100, 250, 347) are roughly geometric.
  * Sign convention: H2 predicts the transfer deficit SHRINKS as n falls, i.e.
    d rises with n, i.e. slope > 0.

TWO ESTIMATORS, AND WHAT IS ACTUALLY EQUIVALENT TO WHAT
-------------------------------------------------------
An earlier draft claimed the per-seed-slope estimator and a pooled model with
seed blocking "give the same coefficient by construction". That is wrong as
stated, and the error was not harmless: on the scaffold split the frozen
probe's median per-seed slope is +0.0073 and the pooled coefficient is +0.0010.

What IS true, and is asserted in `_check_equivalence` on every run:

    pooled OLS interaction coefficient
      == mean of the per-seed OLS slopes            (exactly, to 1e-12)

and it holds in both parameterisations -- `delta ~ seed FE + log2 n` and the
protocol's `rmse ~ seed FE + arm * log2 n` -- because the design is balanced:
every seed contributes the same four log2(n) values to both arms. Balance is
what makes the pooled fit a simple average of within-seed fits.

What is NOT true:

  * The MEDIAN per-seed slope is not that coefficient. Here they differ by
    0.0063, which is larger than the median itself.
  * The tests are not interchangeable either, and point estimates agreeing
    would not make them so. The OLS t-test on the interaction is a statement
    about the MEAN slope and assumes homoscedastic independent residuals. The
    Wilcoxon signed-rank on the 10 per-seed slopes tests whether their
    distribution is SYMMETRIC ABOUT ZERO; under an added symmetry assumption
    it localises the pseudomedian (Hodges-Lehmann), which is a third quantity
    again, and is reported here beside the other two.

All three point estimates and all three intervals are therefore reported. The
Wilcoxon is primary because §4.4 fixes the seed as the unit of replication and
prefers non-parametric tests at this sample size, not because of its p-value.

WILCOXON: ASSUMPTIONS, TIES, ZEROS, AND WHAT SCIPY ACTUALLY COMPUTES
---------------------------------------------------------------------
Per the SciPy documentation for `scipy.stats.wilcoxon`
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html):

  * Null hypothesis: "the distribution of the differences x - y is symmetric
    about zero." It is not a test about a mean or a median as such. Read as a
    location test it localises the pseudomedian, and only under symmetry.
  * Assumes the differences "are independent and identically distributed
    observations, and all are distinct and nonzero."
  * `method='auto'` (the default used here): with no ties and no zeros it uses
    the exact distribution for len(d) <= 50; WITH ties or zeros it performs
    exhaustive permutations for len(d) <= 13 and the asymptotic normal
    approximation otherwise. At n = 10 seeds every test in this script is
    therefore exact or exhaustively permuted -- never the normal
    approximation. The method actually selected is recorded per row in
    `wilcoxon_method`, and the smallest attainable two-sided p at n = 10 is
    2/2**10 = 0.001953.
  * `zero_method='wilcox'` (the default) DISCARDS zero differences, reducing
    the effective n. Rows record `n_zero_slopes` so a p-value computed on
    fewer than 10 effective observations is visible rather than implied.

WHERE THE INDEPENDENCE ASSUMPTION IS STRAINED, AND WHAT THE SEEDS MEASURE
--------------------------------------------------------------------------
A seed here jointly controls the split draw, the training subsample at each
size, and model initialisation and fitting order. Resampling seeds therefore
estimates variability from those three sources **conditional on this fixed set
of 494 compounds**. It is not a bootstrap over compounds and says nothing about
sampling another 494. The 10 test folds are also 10 draws of 98 compounds from
the same 494, so they overlap -- a given compound appears in about two of them
-- and the per-seed slopes are consequently not strictly independent. The
Wilcoxon's iid assumption is approximate for that reason, equally at 10 seeds
and at 30, and no seed count repairs it.

DER, WITH THE CONDITIONING MADE EXPLICIT
-----------------------------------------
`table2` reports DER from the median curve and writes 0.0 where an arm never
reaches the target. That 0 is a censoring convention, not a measured ratio.
Here the DER is recomputed per seed (target = B1's own full-data RMSE *on that
seed*, so the comparison stays paired) and the seeds are partitioned into three
disjoint groups that are reported separately and never averaged together:

  crossed_interior     the curve crosses the target strictly between two
                       evaluated sizes; DER is defined by interpolation
  left_censored        the curve is already at or below the target at n = 50,
                       the smallest size evaluated. The true crossing size is
                       <= 50 and unknown, so DER is a BOUND, not a value
  right_censored       the curve never reaches the target by n = 347. The
                       crossing size is > 347 or does not exist; DER is
                       undefined and is NOT scored as zero

`median_der_interior` summarises only the first group and is labelled as
conditional on crossing. No value is interpolated outside [50, 347], and no
undefined value is turned into a number.

Writes results/tables/table15_h2_interaction.csv
       results/tables/table16_der_uncertainty.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evapro.evaluation.stats import n_to_reach

TABLES = Path("results/tables")
BASELINE = "B1_ecfp_histgb"
SIZES = [50, 100, 250, 347]
N_BOOT = 10000
BOOT_SEED = 0

ARMS = ["B0_median", "B2_descriptors_rf", "T0r_untrained_encoder_probe",
        "T1_chemberta_linear_probe", "T2_chemberta_full_finetune",
        "T4_indomain_probe", "T5_chained_probe"]


def load(split: str) -> dict[str, dict[int, dict[int, float]]]:
    """arm -> seed -> n_train -> RMSE."""
    out: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob("results/metrics/*.json"):
        d = json.load(open(f))
        if d["split"] != split:
            continue
        out[d["arm"]][d["seed"]][d["n_train"]] = d["metrics"]["rmse"]
    return out


def complete_seeds(M, arm: str, split: str) -> list[int]:
    """Seeds where BOTH the arm and the baseline have all four sizes."""
    return sorted(s for s in M[arm]
                  if all(n in M[arm][s] for n in SIZES)
                  and s in M[BASELINE]
                  and all(n in M[BASELINE][s] for n in SIZES))


def per_seed_slopes(M, arm: str, seeds: list[int]) -> np.ndarray:
    """OLS slope of the paired delta on log2(n), one per seed."""
    x = np.log2(np.array(SIZES, dtype=float))
    xc = x - x.mean()
    slopes = []
    for s in seeds:
        d = np.array([M[arm][s][n] - M[BASELINE][s][n] for n in SIZES])
        slopes.append(float((xc @ (d - d.mean())) / (xc @ xc)))
    return np.array(slopes)


def pooled_interaction(M, arm: str, seeds: list[int]) -> tuple[float, float, int]:
    """Interaction coefficient of `rmse ~ seed FE + arm * log2(n)`.

    The parameterisation `plan.md`'s wording points at. Returns the
    coefficient, the residual standard error, and the residual degrees of
    freedom -- the last two so the OLS interval can be formed without pulling
    in a modelling dependency for one two-parameter fit.
    """
    x = np.log2(np.array(SIZES, dtype=float))
    rows = []
    for si, s in enumerate(seeds):
        for k, n in enumerate(SIZES):
            rows.append((si, 0.0, x[k], M[BASELINE][s][n]))
            rows.append((si, 1.0, x[k], M[arm][s][n]))
    si_, a_, x_, y_ = (np.array(v) for v in zip(*rows))
    D = np.zeros((len(rows), len(seeds)))
    D[np.arange(len(rows)), si_.astype(int)] = 1.0
    X = np.column_stack([D, a_, x_, a_ * x_])
    coef, *_ = np.linalg.lstsq(X, y_, rcond=None)
    resid = y_ - X @ coef
    dof = len(rows) - np.linalg.matrix_rank(X)
    return float(coef[-1]), float(np.sqrt(resid @ resid / dof)), int(dof)


def _check_equivalence(M, arm: str, seeds: list[int], slopes: np.ndarray) -> None:
    """Assert the one equivalence that actually holds, on every run.

    Balanced design => the pooled interaction coefficient is the MEAN of the
    per-seed slopes. If a future change unbalances the design (an arm missing a
    size, say) this stops being true, and it should fail loudly rather than let
    §5.3's wording go stale a second time.
    """
    coef, _, _ = pooled_interaction(M, arm, seeds)
    if not np.isclose(coef, slopes.mean(), atol=1e-10):
        raise AssertionError(
            f"{arm}: pooled interaction {coef:.10f} != mean per-seed slope "
            f"{slopes.mean():.10f}. The design is no longer balanced, and "
            f"§5.3's statement of the equivalence must be re-derived.")


def hodges_lehmann(v: np.ndarray) -> float:
    """Pseudomedian: the median of all Walsh averages (v_i + v_j)/2, i <= j.

    This is the location parameter the signed-rank test actually localises,
    and it is neither the mean nor the median of the sample.
    """
    i, j = np.triu_indices(len(v))
    return float(np.median((v[i] + v[j]) / 2.0))


def boot_ci(v: np.ndarray, stat=np.median, alpha=0.05) -> tuple[float, float]:
    """Percentile bootstrap over SEEDS -- the unit of replication (§4.4).

    Conditional on this dataset: it resamples the 10 seeds, not the 494
    compounds, so it captures split/subsample/initialisation variability and
    not sampling variability in the compound set.
    """
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    draws = stat(v[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_with_method(v: np.ndarray) -> tuple[float, str, int]:
    """Signed-rank p, the method SciPy actually selected, and the zero count.

    `method='auto'` is left as the default and then interrogated, rather than
    pinned: pinning 'exact' would silently change behaviour if ties appear.
    """
    n_zero = int((v == 0).sum())
    if not np.any(v != 0):
        return 1.0, "degenerate (all zero)", n_zero
    p = float(stats.wilcoxon(v).pvalue)
    nz = v[v != 0]
    has_ties = len(np.unique(np.abs(nz))) < len(nz)
    if has_ties or n_zero:
        method = "exhaustive permutations" if len(nz) <= 13 else "asymptotic"
    else:
        method = "exact" if len(nz) <= 50 else "asymptotic"
    return p, method, n_zero


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (len(p) - r)))
        adj[i] = run
    return adj


def interaction(M, split: str) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        if arm not in M:
            continue
        seeds = complete_seeds(M, arm, split)
        if len(seeds) < 3:
            continue
        b = per_seed_slopes(M, arm, seeds)
        _check_equivalence(M, arm, seeds, b)
        coef, sigma, dof = pooled_interaction(M, arm, seeds)
        p, method, n_zero = wilcoxon_with_method(b)
        lo, hi = boot_ci(b)
        mlo, mhi = boot_ci(b, stat=np.mean)
        d50 = np.array([M[arm][s][50] - M[BASELINE][s][50] for s in seeds])
        d347 = np.array([M[arm][s][347] - M[BASELINE][s][347] for s in seeds])
        rows.append(dict(
            split=split, arm=arm, n_seeds=len(seeds),
            # --- the three location estimates, kept distinct -----------------
            median_slope=round(float(np.median(b)), 4),
            slope_ci_lo=round(lo, 4), slope_ci_hi=round(hi, 4),
            mean_slope=round(float(np.mean(b)), 4),
            mean_slope_ci_lo=round(mlo, 4), mean_slope_ci_hi=round(mhi, 4),
            pooled_interaction_coef=round(coef, 4),
            pooled_equals_mean=bool(np.isclose(coef, b.mean(), atol=1e-10)),
            hodges_lehmann=round(hodges_lehmann(b), 4),
            mean_minus_median=round(float(np.mean(b) - np.median(b)), 4),
            # --- the test ----------------------------------------------------
            slope_positive_in_seeds=int((b > 0).sum()),
            n_zero_slopes=n_zero,
            wilcoxon_method=method,
            p_raw=round(p, 4),
            # --- context -----------------------------------------------------
            median_delta_n50=round(float(np.median(d50)), 4),
            median_delta_n347=round(float(np.median(d347)), 4),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df.p_raw.to_numpy(float)).round(4)
        # The verdict is driven by the TEST, and the test is about the sign of
        # a location shift, so it is reported against the estimator the test
        # localises rather than against whichever point estimate is largest.
        df["h2_verdict"] = np.where(
            df.p_holm > 0.05, "inconclusive",
            np.where(df.hodges_lehmann > 0, "slope > 0: consistent with H2",
                     "slope < 0: contrary to H2"))
        df["estimators_agree_in_sign"] = (
            np.sign(df.median_slope) == np.sign(df.mean_slope))
    return df


def der(M, split: str) -> pd.DataFrame:
    """Per-seed DER, with the three censoring states kept apart."""
    rows = []
    for arm in [BASELINE] + ARMS:
        if arm not in M:
            continue
        seeds = complete_seeds(M, arm, split)
        if len(seeds) < 3:
            continue
        interior, left, right, nonmono = [], 0, 0, 0
        for s in seeds:
            target = M[BASELINE][s][347]          # paired: this seed's own target
            v = np.array([M[arm][s][n] for n in SIZES])
            nonmono += int(np.any(np.diff(v) > 0))
            if v[0] <= target:
                left += 1                          # crossing size is <= 50, unknown
                continue
            n_a = n_to_reach([float(n) for n in SIZES], list(v), target=target)
            n_b = n_to_reach([float(n) for n in SIZES],
                             [M[BASELINE][s][n] for n in SIZES], target=target)
            if not np.isfinite(n_a):
                right += 1                         # never reaches by n = 347
                continue
            if np.isfinite(n_b) and n_a > 0:
                interior.append(n_b / n_a)
        interior = np.array(interior)
        row = dict(split=split, arm=arm, n_seeds=len(seeds),
                   crossed_interior=len(interior),
                   left_censored_at_n50=left,
                   right_censored_never_reached=right,
                   curves_non_monotonic=nonmono)
        if len(interior) >= 3:
            lo, hi = boot_ci(interior)
            row |= dict(median_der_interior=round(float(np.median(interior)), 4),
                        der_ci_lo=round(lo, 4), der_ci_hi=round(hi, 4))
        else:
            row |= dict(median_der_interior=np.nan,
                        der_ci_lo=np.nan, der_ci_hi=np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        assert (df.crossed_interior + df.left_censored_at_n50
                + df.right_censored_never_reached == df.n_seeds).all(), \
            "the three censoring states must partition the seeds"
    return df


def write(df: pd.DataFrame, path: Path, note: str) -> None:
    path.write_text(
        f"# generated by scripts/analyse_h2.py on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# {note}\n" + df.to_csv(index=False))
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["scaffold", "random", "butina"])
    args = ap.parse_args()

    inter, ders = [], []
    for split in args.splits:
        M = load(split)
        inter.append(interaction(M, split))
        ders.append(der(M, split))
    idf, ddf = pd.concat(inter, ignore_index=True), pd.concat(ders, ignore_index=True)

    write(idf, TABLES / "table15_h2_interaction.csv",
          "paired delta RMSE (arm - B1) on log2(n_train). median/mean/pseudomedian of "
          "per-seed slopes reported separately; pooled OLS interaction == MEAN, not "
          "median. Wilcoxon vs 0 over seeds, Holm within the arm family")
    write(ddf, TABLES / "table16_der_uncertainty.csv",
          "per-seed DER vs that seed's own B1 full-data RMSE. Seeds partitioned into "
          "interior crossings / left-censored at n=50 / never reached; the median is "
          "conditional on interior crossing and censored seeds are never scored 0")

    pd.set_option("display.width", 240)
    print("\n-- H2 interaction (scaffold): three estimators, one test --")
    print(idf[idf.split == "scaffold"][
        ["arm", "median_slope", "mean_slope", "pooled_interaction_coef",
         "hodges_lehmann", "pooled_equals_mean", "slope_positive_in_seeds",
         "wilcoxon_method", "p_raw", "p_holm", "h2_verdict"]].to_string(index=False))
    print("\n-- DER, censoring states kept apart (scaffold) --")
    print(ddf[ddf.split == "scaffold"][
        ["arm", "crossed_interior", "left_censored_at_n50",
         "right_censored_never_reached", "curves_non_monotonic",
         "median_der_interior", "der_ci_lo", "der_ci_hi"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
