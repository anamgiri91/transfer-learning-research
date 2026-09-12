#!/usr/bin/env python
"""Analysis for the Amendment 4 fine-tuning arms (T2v, T4ft, T5ft).

Amended experiments, designed after the earlier results were seen. Nothing here
may be described as the pre-registered analysis or as independent confirmation.

It refuses to report a comparison until every planned cell for it exists. That
is not fastidiousness: `plan.md` Amendment 4 fixes ten seeds with no stopping
rule, and a table that quietly summarises the six seeds that happen to have
finished is how a stopping rule gets introduced by accident. Partial state is
printed as a completeness report instead, with no test statistics at all.

Three questions, and each is reported against the arms that can answer it:

  H2   does the corrected generic fine-tune's deficit shrink as data shrinks?
       The same paired-slope estimator as §5.3, on T2v's four sizes.
  H3   in-domain vs generic pretraining, architecture and adaptation matched:
       T4ft vs T2v and T5ft vs T2v at n = 347. This is the comparison form
       plan.md §1 names and that the study had never run.
  --   T2v vs T2: two conditions, NOT an isolation of the schedule, because
       they differ in readout as well (Amendment 4).

Writes results/tables/table17_amended_finetune.csv
       results/tables/table18_amended_contrasts.csv
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

TABLES = Path("results/tables")
FT = Path("results/metrics_ft")
BASE = Path("results/metrics")
SIZES = [50, 100, 250, 347]
SEEDS = list(range(10))
PLANNED = {"T2v": SIZES, "T4ft": [347], "T5ft": [347]}


def load_ft() -> dict:
    out = defaultdict(dict)
    for f in FT.glob("*.json"):
        if f.name.endswith(".FAILED.json"):
            continue
        d = json.loads(f.read_text())
        out[d["arm"]][(d["seed"], d["n_train"])] = d
    return out


def load_published(arm: str) -> dict:
    out = {}
    for f in glob.glob(f"results/metrics/{arm}__scaffold__*.json"):
        d = json.loads(open(f).read())
        out[(d["seed"], d["n_train"])] = d
    return out


def failures() -> list[dict]:
    return [json.loads(f.read_text()) for f in FT.glob("*.FAILED.json")]


def completeness() -> pd.DataFrame:
    M = load_ft()
    rows = []
    for arm, sizes in PLANNED.items():
        for n in sizes:
            have = sum(1 for s in SEEDS if (s, n) in M.get(arm, {}))
            rows.append(dict(arm=arm, n_train=n, planned=len(SEEDS), complete=have,
                             missing=len(SEEDS) - have))
    return pd.DataFrame(rows)


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Paired signed-rank p at full precision, and the method SciPy selected.

    The statistic uses the SIGNS of the paired differences together with the
    RANKS OF THEIR ABSOLUTE VALUES -- it is not a sign test, and it is not
    indifferent to magnitude; it is indifferent to the *scale* of the
    magnitudes, responding only to their rank order.
    """
    d = a - b
    if not np.any(d != 0):
        return 1.0, "degenerate (all differences zero)"
    nz = d[d != 0]
    has_ties = len(np.unique(np.abs(nz))) < len(nz)
    has_zeros = len(nz) < len(d)
    if has_ties or has_zeros:
        method = "exhaustive permutations" if len(nz) <= 13 else "asymptotic"
    else:
        method = "exact" if len(nz) <= 50 else "asymptotic"
    return float(stats.wilcoxon(a, b).pvalue), method


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (len(p) - r)))
        adj[i] = run
    return adj


def curve_table(M) -> pd.DataFrame:
    """Per-size medians for every amended arm, plus the comparators."""
    rows = []
    b1 = load_published("B1_ecfp_histgb")
    t2 = load_published("T2_chemberta_full_finetune")
    for arm, sizes in PLANNED.items():
        for n in sizes:
            cells = [M[arm][(s, n)] for s in SEEDS if (s, n) in M.get(arm, {})]
            if len(cells) < len(SEEDS):
                continue
            r = np.array([c["metrics"]["rmse"] for c in cells])
            rows.append(dict(
                arm=arm, n_train=n, n_seeds=len(cells),
                median_rmse=round(float(np.median(r)), 4),
                iqr=round(float(np.subtract(*np.percentile(r, [75, 25]))), 4),
                median_spearman=round(float(np.median(
                    [c["metrics"]["spearman"] for c in cells])), 4),
                median_selected_lr=float(np.median([c["selected_lr"] for c in cells])),
                median_selected_epoch=float(np.median(
                    [c["selected_epoch"] for c in cells])),
                median_optimizer_steps=float(np.median(
                    [c["optimizer_steps"] for c in cells])),
                median_val_rmse=round(float(np.median(
                    [c["selected_val_rmse"] for c in cells])), 4),
                median_seconds=round(float(np.median([c["seconds"] for c in cells])), 1),
                n_train_fitted=int(cells[0]["n_train_fitted"]),
                n_internal_val=int(cells[0]["n_internal_val"]),
                b1_median_rmse=round(float(np.median(
                    [b1[(s, n)]["metrics"]["rmse"] for s in SEEDS])), 4),
                t2_median_rmse=(round(float(np.median(
                    [t2[(s, n)]["metrics"]["rmse"] for s in SEEDS])), 4)
                    if all((s, n) in t2 for s in SEEDS) else np.nan),
            ))
    return pd.DataFrame(rows)


# The amended RMSE family, fixed by plan.md Amendment 4's erratum BEFORE any
# H3 result was analysed. Seven contrasts. The size is 7 whether or not all
# seven have finished: correcting over however many happen to be complete would
# shrink the family as a side effect of scheduling.
AMENDED_FAMILY = [
    ("T2v", "B1", 50), ("T2v", "B1", 100), ("T2v", "B1", 250), ("T2v", "B1", 347),
    ("T4ft", "T2v", 347), ("T5ft", "T2v", 347), ("T2v", "T2", 347),
]
FAMILY_M = len(AMENDED_FAMILY)


def paired_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over seeds for the median paired difference."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    draws = np.median(d[idx], axis=1)
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def contrasts(M) -> pd.DataFrame:
    """The amended family: paired, corrected within itself at m = 7, reported
    apart from the pre-registered families rather than merged into them.

    SIGN CONVENTION, stated once and used everywhere in this file:

        delta(seed) = RMSE_arm(seed) - RMSE_reference(seed)

    so **positive delta means the arm is WORSE** (RMSE is lower-is-better).
    `median_delta` is the median of those per-seed differences -- a paired
    quantity -- and is NOT the difference between the two marginal medians,
    which is reported separately as `marginal_median_diff` because the two can
    disagree in sign (§5.4 documents a case where they do).

    P-values are carried at full precision into the correction and rounded
    only for display: rounding a p to 4 dp before Holm can change which
    hypothesis is rejected.
    """
    b1 = load_published("B1_ecfp_histgb")
    t2 = load_published("T2_chemberta_full_finetune")
    rows = []

    def series(arm, n):
        if arm == "B1":
            return np.array([b1[(s, n)]["metrics"]["rmse"] for s in SEEDS])
        if arm == "T2":
            return np.array([t2[(s, n)]["metrics"]["rmse"] for s in SEEDS])
        if all((s, n) in M.get(arm, {}) for s in SEEDS):
            return np.array([M[arm][(s, n)]["metrics"]["rmse"] for s in SEEDS])
        return None

    what = {
        ("T2v", "B1"): "corrected generic fine-tune vs the baseline (H1)",
        ("T4ft", "T2v"): "in-domain vs generic pretraining, adaptation matched "
                         "(H3, the pre-registered comparison form)",
        ("T5ft", "T2v"): "chained vs generic pretraining, adaptation matched (H3)",
        ("T2v", "T2"): "two conditions: schedule AND readout differ, not an isolation",
    }
    for arm, ref, n in AMENDED_FAMILY:
        a, b = series(arm, n), series(ref, n)
        if a is None or b is None:
            rows.append(dict(arm=arm, reference=ref, n_train=n,
                             what=what[(arm, ref)], n_seeds=0, status="pending"))
            continue
        d = a - b                       # positive => arm worse
        p, method = wilcoxon(a, b)
        lo, hi = paired_ci(d)
        rows.append(dict(
            arm=arm, reference=ref, n_train=n, what=what[(arm, ref)],
            n_seeds=len(SEEDS), status="complete",
            median_paired_delta=float(np.median(d)),
            paired_ci_lo=lo, paired_ci_hi=hi,
            marginal_median_diff=float(np.median(a) - np.median(b)),
            arm_better_in_seeds=int((d < 0).sum()),
            arm_worse_in_seeds=int((d > 0).sum()),
            ties=int((d == 0).sum()),
            wilcoxon_method=method, p_raw=float(p)))

    df = pd.DataFrame(rows)
    done = df.status == "complete"
    df["family_m"] = FAMILY_M
    df["family_complete"] = bool(done.all())
    if done.any():
        if done.all():
            adj = np.full(len(df), np.nan)
            adj[done.to_numpy()] = holm(df.loc[done, "p_raw"].to_numpy(float))
            df["p_adjusted"] = adj
            df["adjustment"] = np.where(done, "Holm, m=7 (family complete)", "")
        else:
            # Valid upper bound on the Holm adjustment with the family
            # incomplete: no assignment of the missing p-values can make the
            # Holm value exceed p_raw * m.
            df["p_adjusted"] = np.where(
                done, np.minimum(1.0, df.p_raw.astype(float) * FAMILY_M), np.nan)
            df["adjustment"] = np.where(
                done, f"Bonferroni bound, m={FAMILY_M} (family incomplete)", "")
        df["verdict"] = np.where(
            ~done, "pending",
            np.where(df.p_adjusted > 0.05, "no detectable difference",
                     np.where(df.median_paired_delta < 0, "arm better", "arm worse")))
    return df


def per_seed_table(M) -> pd.DataFrame:
    """Per-seed RMSE for B1, T2 and T2v at full data, with both differences.

    Written because the two "differences" in this paper are different
    quantities and a reader has to be able to see them side by side:

        delta_T2v_B1(seed) = RMSE_T2v(seed) - RMSE_B1(seed)   -- PAIRED
        positive => T2v worse on that seed

    The median of that column is the paired effect the Wilcoxon tests. The
    difference between the two columns' medians is the MARGINAL difference, and
    the two are not the same number -- §5.4 already documents a case where they
    disagree in sign.
    """
    if not all((s, 347) in M.get("T2v", {}) for s in SEEDS):
        return pd.DataFrame()
    b1 = load_published("B1_ecfp_histgb")
    t2 = load_published("T2_chemberta_full_finetune")
    have_h3 = all((s, 347) in M.get(a, {}) for a in ("T4ft", "T5ft") for s in SEEDS)
    rows = []
    for s in SEEDS:
        r_b1 = b1[(s, 347)]["metrics"]["rmse"]
        r_t2 = t2[(s, 347)]["metrics"]["rmse"]
        r_t2v = M["T2v"][(s, 347)]["metrics"]["rmse"]
        row = dict(seed=s, B1=r_b1, T2=r_t2, T2v=r_t2v,
                   delta_T2_minus_B1=r_t2 - r_b1,
                   delta_T2v_minus_B1=r_t2v - r_b1,
                   delta_T2v_minus_T2=r_t2v - r_t2)
        if have_h3:
            r_t4 = M["T4ft"][(s, 347)]["metrics"]["rmse"]
            r_t5 = M["T5ft"][(s, 347)]["metrics"]["rmse"]
            row |= dict(T4ft=r_t4, T5ft=r_t5,
                        delta_T4ft_minus_T2v=r_t4 - r_t2v,
                        delta_T5ft_minus_T2v=r_t5 - r_t2v)
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {"seed": "median"}
    for c in df.columns[1:]:
        summary[c] = float(np.median(df[c]))
    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def h2_slope(M) -> pd.DataFrame:
    """T2v's interaction slope, by the same estimator as §5.3.

    The reason this arm exists. §5.3's only significant H2 result belongs to
    the untuned T2, measured under a schedule §6.2 shows to be an artefact; if
    the corrected arm's slope points the other way, that result was the
    artefact talking.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("h2", "scripts/analyse_h2.py")
    h2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h2)

    if not all((s, n) in M.get("T2v", {}) for s in SEEDS for n in SIZES):
        return pd.DataFrame()
    b1 = load_published("B1_ecfp_histgb")
    # Shape the amended arm into what analyse_h2's estimators consume, so the
    # two sections use one implementation rather than two that can diverge.
    MM = {"T2v": {s: {n: M["T2v"][(s, n)]["metrics"]["rmse"] for n in SIZES}
                  for s in SEEDS},
          h2.BASELINE: {s: {n: b1[(s, n)]["metrics"]["rmse"] for n in SIZES}
                        for s in SEEDS}}
    slopes = h2.per_seed_slopes(MM, "T2v", SEEDS)
    h2._check_equivalence(MM, "T2v", SEEDS, slopes)
    coef, _, _ = h2.pooled_interaction(MM, "T2v", SEEDS)
    p, method, n_zero = h2.wilcoxon_with_method(slopes)
    lo, hi = h2.boot_ci(slopes)

    # Multiplicity, both ways, because neither alone is the whole story.
    #
    # Within the amended set this is the ONLY slope test, so there is nothing
    # to correct for -- plan.md Amendment 4's erratum fixes that. Reporting
    # only that would be convenient, so the pooled figure is given too: Holm
    # across this slope and the seven published slope tests of §5.3, as a
    # sensitivity analysis. It is NOT merged into §5.3's family, because
    # re-correcting pre-registered results retroactively is the failure
    # recorded on 2026-09-02.
    pub = pd.read_csv("results/tables/table15_h2_interaction.csv", comment="#")
    pub = pub[pub.split == "scaffold"]
    pooled_in = np.concatenate([pub.p_raw.to_numpy(float), [p]])
    pooled = h2.holm(pooled_in)[-1]

    return pd.DataFrame([dict(
        arm="T2v", split="scaffold", n_seeds=len(SEEDS),
        median_slope=round(float(np.median(slopes)), 4),
        slope_ci_lo=round(lo, 4), slope_ci_hi=round(hi, 4),
        mean_slope=round(float(slopes.mean()), 4),
        pooled_interaction_coef=round(coef, 4),
        hodges_lehmann=round(h2.hodges_lehmann(slopes), 4),
        slope_positive_in_seeds=int((slopes > 0).sum()),
        n_zero_slopes=n_zero, wilcoxon_method=method,
        p_raw=float(p),
        amended_family_m=1,
        p_adjusted_amended=float(p),      # m = 1: adjustment is the identity
        p_holm_pooled_with_published=float(pooled),
        pooled_family_m=int(len(pooled_in)),
        h2_verdict=("inconclusive" if pooled > 0.05 else
                    "slope > 0: consistent with H2" if np.median(slopes) > 0
                    else "slope < 0: contrary to H2"),
        untuned_T2_median_slope=-0.1764,
    )])


def write(df: pd.DataFrame, path: Path, note: str) -> None:
    path.write_text(
        f"# generated by scripts/analyse_amended.py on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# {note}\n" + df.to_csv(index=False))
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="write tables from whatever is complete. Off by default: "
                         "a partial table is a stopping rule by accident.")
    args = ap.parse_args()

    comp = completeness()
    # Always written, complete or not. §6.6 renders it, so the manuscript's
    # progress figures update with `make report` instead of being hand-typed
    # into prose that goes stale every time a cell lands.
    comp_out = comp.assign(failures=len(failures()))
    write(comp_out, TABLES / "table19_amended_progress.csv",
          "Amendment 4 sweep completeness. 10 seeds per cell, no stopping rule; "
          "every planned cell is reported on completion, favourable or not")
    print("== completeness (Amendment 4 fixes 10 seeds, no stopping rule) ==")
    print(comp.to_string(index=False))
    fails = failures()
    print(f"\nrecorded failures: {len(fails)}")
    for f in fails:
        print(f"  {f['tag']}: {f['error'].strip().splitlines()[-1][:100]}")

    # The gate is per COMPARISON, not global. A contrast whose own ten seeds
    # are all present is complete and is reported; one still missing a cell is
    # withheld entirely and named. Gating globally would withhold a finished
    # sub-experiment because an unrelated one is still running; gating on
    # "whatever has finished" would be a stopping rule. Neither is this.
    ready = {f"{r.arm}@{int(r.n_train)}" for r in comp.itertuples() if r.missing == 0}
    withheld = {f"{r.arm}@{int(r.n_train)}" for r in comp.itertuples() if r.missing}
    print(f"\ncomplete and reportable: {sorted(ready) or 'none'}")
    if withheld:
        print(f"withheld until every planned seed exists: {sorted(withheld)}")
    if not ready and not args.allow_partial:
        print("\nNothing is complete yet; no statistics computed.")
        return 2

    M = load_ft()
    cur, con, slope = curve_table(M), contrasts(M), h2_slope(M)
    seeds_tbl = per_seed_table(M)
    if not seeds_tbl.empty:
        write(seeds_tbl, TABLES / "table21_per_seed_finetune.csv",
              "per-seed RMSE at n=347. delta_X_minus_Y = RMSE_X - RMSE_Y on the "
              "same seed; POSITIVE means X is worse. The median of a delta column "
              "is the PAIRED effect; the difference of two arms' medians is the "
              "MARGINAL difference, and they are not the same quantity")
    write(cur, TABLES / "table17_amended_finetune.csv",
          "Amendment 4 arms: validation-selected lr and checkpoint, scaffold split, "
          "seeds 0-9. AMENDED, not pre-registered")
    write(con, TABLES / "table18_amended_contrasts.csv",
          "paired Wilcoxon over seeds within the amended family, Holm-corrected "
          "within that family only and reported apart from the pre-registered ones")
    if not slope.empty:
        write(slope, TABLES / "table20_amended_h2_slope.csv",
              "T2v interaction slope by the §5.3 estimator; positive supports H2")
    pd.set_option("display.width", 220)
    print("\n-- amended curve --")
    print(cur.to_string(index=False))
    print("\n-- amended contrasts --")
    print(con.to_string(index=False))
    if not slope.empty:
        print("\n-- H2 interaction slope, corrected fine-tune --")
        print(slope.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
