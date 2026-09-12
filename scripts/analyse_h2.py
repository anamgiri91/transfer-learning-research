#!/usr/bin/env python
"""H2: does the transfer advantage grow as the training set shrinks?

`plan.md` §6 decides H2 by two things: "Interaction term in the learning-curve
model; DER". Only the DER half was ever run, and DER alone cannot answer H2 --
it asks whether an arm ever crosses a fixed threshold, which is a question
about the *full-data* end of the curve. An arm can be worse at full data and
still be relatively better at n = 50; nothing in DER = 0 rules that out. This
script runs the half that was missing and puts uncertainty on the half that
was not.

DISCLOSURE -- modelling choices made after results existed
----------------------------------------------------------
`plan.md` names an "interaction term in the learning-curve model" and stops
there: no model form, no error structure, no test. Everything below is
therefore a post-hoc specification of a pre-registered intent, and is recorded
as such in docs/decision-log.md (2026-09-11). The choices, and why:

  * Outcome is the PAIRED delta d(arm, seed, n) = RMSE_arm - RMSE_B1, positive
    meaning the arm is worse. Pairing is preserved at both levels a benchmark
    can break it: within a seed both arms see the same split AND the same
    training subsample, and every seed contributes all four sizes.
  * Size enters as log2(n). Learning curves are conventionally near-linear in
    log n, and the four sizes (50, 100, 250, 347) are roughly geometric.
  * Primary test: per-seed OLS slope of d on log2(n), then a one-sample
    Wilcoxon signed-rank on the 10 slopes. This matches §4.4's discipline
    (seed is the unit of replication; non-parametric; effect size beside p)
    and avoids fitting a mixed model with 10 clusters.
  * The single-model form the protocol's wording suggests -- RMSE ~ arm x
    log2(n) with seed fixed effects -- is also reported. Its interaction
    coefficient is the mean per-seed slope; the CI comes from a seed-level
    (cluster) bootstrap, so the two rows agree by construction and neither
    leans on a normality assumption. No new dependency was added for this.

  Sign convention: H2 predicts the transfer deficit SHRINKS as n falls, i.e.
  d rises with n, i.e. slope > 0. A slope <= 0 is evidence against H2, not
  merely an absence of evidence for it -- and that distinction is why this
  test is worth running even though the DER is already 0.

DER, with the censoring made explicit
-------------------------------------
`table2` reports DER from the median curve and writes 0.0 where an arm never
reaches the target. That 0 is a censoring convention, not a measured ratio,
and it carries no uncertainty. Here the DER is recomputed per seed (target =
B1's own full-data RMSE *on that seed*, so the comparison stays paired), and
three things are reported that the collapse to 0 hides: how many seeds the arm
crosses in at all, the DER among those seeds, and a seed-level bootstrap CI.
Curves that never cross are reported as censored, never as DER = 0.

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

# Every arm with full curve coverage on the split being analysed. B0 is kept:
# a flat median predictor is the one arm whose interaction slope has a known
# sign, so it doubles as a sanity check on the model.
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


def boot_ci(v: np.ndarray, stat=np.median, alpha=0.05) -> tuple[float, float]:
    """Percentile bootstrap over SEEDS -- the unit of replication (§4.4)."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    draws = stat(v[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def interaction(M, split: str) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        if arm not in M:
            continue
        seeds = complete_seeds(M, arm, split)
        if len(seeds) < 3:
            continue
        b = per_seed_slopes(M, arm, seeds)
        p = float(stats.wilcoxon(b).pvalue) if np.any(b != 0) else 1.0
        lo, hi = boot_ci(b)
        mlo, mhi = boot_ci(b, stat=np.mean)
        # Paired delta at the extremes, so the slope can be read against the
        # thing it is a slope of.
        d50 = np.array([M[arm][s][50] - M[BASELINE][s][50] for s in seeds])
        d347 = np.array([M[arm][s][347] - M[BASELINE][s][347] for s in seeds])
        rows.append(dict(
            split=split, arm=arm, n_seeds=len(seeds),
            median_slope=round(float(np.median(b)), 4),
            slope_ci_lo=round(lo, 4), slope_ci_hi=round(hi, 4),
            mean_slope=round(float(np.mean(b)), 4),
            mean_slope_ci_lo=round(mlo, 4), mean_slope_ci_hi=round(mhi, 4),
            slope_positive_in_seeds=int((b > 0).sum()),
            p_raw=round(p, 4),
            median_delta_n50=round(float(np.median(d50)), 4),
            median_delta_n347=round(float(np.median(d347)), 4),
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        # Holm within the arm family, as every other family in the paper is.
        p = df.p_raw.to_numpy(float)
        order = np.argsort(p)
        adj = np.empty(len(p))
        run = 0.0
        for r, i in enumerate(order):
            run = max(run, min(1.0, p[i] * (len(p) - r)))
            adj[i] = run
        df["p_holm"] = adj.round(4)
        df["h2_verdict"] = np.where(
            df.p_holm > 0.05, "inconclusive",
            np.where(df.median_slope > 0, "slope > 0: consistent with H2",
                     "slope < 0: contrary to H2"))
    return df


def der(M, split: str) -> pd.DataFrame:
    """Per-seed DER with censoring reported rather than collapsed to zero."""
    rows = []
    for arm in [BASELINE] + ARMS:
        if arm not in M:
            continue
        seeds = complete_seeds(M, arm, split)
        if len(seeds) < 3:
            continue
        ders, crossed, nonmono = [], 0, 0
        for s in seeds:
            target = M[BASELINE][s][347]          # paired: this seed's own target
            base_c = ([float(n) for n in SIZES], [M[BASELINE][s][n] for n in SIZES])
            arm_c = ([float(n) for n in SIZES], [M[arm][s][n] for n in SIZES])
            v = np.array([M[arm][s][n] for n in SIZES])
            nonmono += int(np.any(np.diff(v) > 0))
            n_b, n_a = n_to_reach(*base_c, target=target), n_to_reach(*arm_c, target=target)
            if np.isfinite(n_a) and np.isfinite(n_b) and n_a > 0:
                ders.append(n_b / n_a)
                crossed += 1
        ders = np.array(ders)
        row = dict(split=split, arm=arm, n_seeds=len(seeds),
                   seeds_reaching_target=crossed,
                   seeds_censored=len(seeds) - crossed,
                   curves_non_monotonic=nonmono)
        if crossed >= 3:
            lo, hi = boot_ci(ders)
            row |= dict(median_der_where_defined=round(float(np.median(ders)), 4),
                        der_ci_lo=round(lo, 4), der_ci_hi=round(hi, 4))
        else:
            row |= dict(median_der_where_defined=np.nan,
                        der_ci_lo=np.nan, der_ci_hi=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


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
          "paired delta RMSE (arm - B1) regressed on log2(n_train); per-seed slopes, "
          "Wilcoxon vs 0, Holm within the arm family; slope > 0 supports H2")
    write(ddf, TABLES / "table16_der_uncertainty.csv",
          "per-seed DER against that seed's own B1 full-data RMSE; censored seeds "
          "counted, never scored as DER = 0")

    pd.set_option("display.width", 200)
    print("\n-- H2 interaction (scaffold) --")
    print(idf[idf.split == "scaffold"][
        ["arm", "median_slope", "slope_ci_lo", "slope_ci_hi",
         "slope_positive_in_seeds", "p_raw", "p_holm", "h2_verdict"]].to_string(index=False))
    print("\n-- DER with censoring (scaffold) --")
    print(ddf[ddf.split == "scaffold"][
        ["arm", "seeds_reaching_target", "seeds_censored", "curves_non_monotonic",
         "median_der_where_defined", "der_ci_lo", "der_ci_hi"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
