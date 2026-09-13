#!/usr/bin/env python
"""Arm B3 (D-MPNN) against B1 and T2v — plan.md Amendment 5's five contrasts.

The arm is pre-registered (`plan.md` §4); the implementation is post-hoc
(Amendment 5). The family is the five contrasts Amendment 5 enumerates, m = 5,
held at 5 whether or not all five have finished so that completion cannot
shrink it — the same discipline Amendment 4's erratum imposes.

Sign convention, as everywhere else in this study:

    delta(seed) = RMSE_arm(seed) - RMSE_reference(seed)   POSITIVE => arm worse

Writes results/tables/table23_b3_curve.csv
       results/tables/table24_b3_contrasts.csv
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TABLES = Path("results/tables")
SIZES = [50, 100, 250, 347]
SEEDS = list(range(10))

# Fixed in Amendment 5 before any B3 result existed.
B3_FAMILY = [("B3", "B1", n) for n in SIZES] + [("B3", "T2v", 347)]
FAMILY_M = len(B3_FAMILY)


def load(pattern: str, key="arm") -> dict:
    out = {}
    for f in glob.glob(pattern):
        if f.endswith(".FAILED.json"):
            continue
        d = json.loads(open(f).read())
        out[(d["seed"], d["n_train"])] = d
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (len(p) - r)))
        adj[i] = run
    return adj


def paired_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return tuple(float(v) for v in np.percentile(np.median(d[idx], axis=1), [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    b3 = load("results/metrics_b3/B3__scaffold__*.json")
    b1 = load("results/metrics/B1_ecfp_histgb__scaffold__*.json")
    b2 = load("results/metrics/B2_descriptors_rf__scaffold__*.json")
    t2v = load("results/metrics_ft/T2v__scaffold__*.json")

    n_have = sum(1 for s in SEEDS for n in SIZES if (s, n) in b3)
    n_fail = len(glob.glob("results/metrics_b3/*.FAILED.json"))
    print(f"B3 cells: {n_have}/40 complete, {n_fail} failed")

    # ---- curve -----------------------------------------------------------
    rows = []
    for n in SIZES:
        if not all((s, n) in b3 for s in SEEDS):
            continue
        cells = [b3[(s, n)] for s in SEEDS]
        r = np.array([c["metrics"]["rmse"] for c in cells])
        rows.append(dict(
            arm="B3", n_train=n, n_seeds=len(cells),
            median_rmse=round(float(np.median(r)), 4),
            iqr=round(float(np.subtract(*np.percentile(r, [75, 25]))), 4),
            median_spearman=round(float(np.median(
                [c["metrics"]["spearman"] for c in cells])), 4),
            # Labelled budget vs what was actually fitted -- Amendment 5 asks
            # for both, because they differ and B1 does not report the split.
            n_train_labelled=n,
            n_train_fitted=int(cells[0]["n_train_fitted"]),
            n_internal_val=int(cells[0]["n_internal_val"]),
            median_selected_lr=float(np.median([c["selected_lr"] for c in cells])),
            median_selected_epoch=float(np.median([c["selected_epoch"] for c in cells])),
            median_seconds=round(float(np.median([c["seconds"] for c in cells])), 1),
            b1_median_rmse=round(float(np.median(
                [b1[(s, n)]["metrics"]["rmse"] for s in SEEDS])), 4),
            b2_median_rmse=round(float(np.median(
                [b2[(s, n)]["metrics"]["rmse"] for s in SEEDS])), 4),
            t2v_median_rmse=(round(float(np.median(
                [t2v[(s, n)]["metrics"]["rmse"] for s in SEEDS])), 4)
                if all((s, n) in t2v for s in SEEDS) else np.nan),
        ))
    curve = pd.DataFrame(rows)

    # ---- the five contrasts ----------------------------------------------
    src = {"B1": b1, "T2v": t2v, "B3": b3}
    crows = []
    for arm, ref, n in B3_FAMILY:
        A, B = src[arm], src[ref]
        if not (all((s, n) in A for s in SEEDS) and all((s, n) in B for s in SEEDS)):
            crows.append(dict(arm=arm, reference=ref, n_train=n, status="pending"))
            continue
        a = np.array([A[(s, n)]["metrics"]["rmse"] for s in SEEDS])
        b = np.array([B[(s, n)]["metrics"]["rmse"] for s in SEEDS])
        d = a - b
        p = 1.0 if not np.any(d != 0) else float(stats.wilcoxon(a, b).pvalue)
        lo, hi = paired_ci(d)
        crows.append(dict(
            arm=arm, reference=ref, n_train=n, status="complete", n_seeds=len(SEEDS),
            median_paired_delta=float(np.median(d)),
            paired_ci_lo=lo, paired_ci_hi=hi,
            marginal_median_diff=float(np.median(a) - np.median(b)),
            arm_better_in_seeds=int((d < 0).sum()),
            arm_worse_in_seeds=int((d > 0).sum()),
            p_raw=float(p)))
    con = pd.DataFrame(crows)
    done = con.status == "complete"
    con["family_m"] = FAMILY_M
    con["family_complete"] = bool(done.all())
    if done.any():
        if done.all():
            adj = np.full(len(con), np.nan)
            adj[done.to_numpy()] = holm(con.loc[done, "p_raw"].to_numpy(float))
            con["p_adjusted"], con["adjustment"] = adj, np.where(
                done, f"Holm, m={FAMILY_M} (complete)", "")
        else:
            con["p_adjusted"] = np.where(
                done, np.minimum(1.0, con.p_raw.astype(float) * FAMILY_M), np.nan)
            con["adjustment"] = np.where(
                done, f"Bonferroni bound, m={FAMILY_M} (incomplete)", "")
        con["verdict"] = np.where(
            ~done, "pending",
            np.where(con.p_adjusted > 0.05, "no detectable difference",
                     np.where(con.median_paired_delta < 0, "arm better", "arm worse")))

    for df, path, note in (
        (curve, "table23_b3_curve.csv",
         "B3 D-MPNN (chemprop 2.3.1), scaffold split, seeds 0-9. Pre-registered arm, "
         "post-hoc implementation (plan.md Amendment 5)"),
        (con, "table24_b3_contrasts.csv",
         "Amendment 5's five contrasts. delta = arm - reference, POSITIVE means arm "
         "worse. Family held at m=5 whether or not complete")):
        (TABLES / path).write_text(
            f"# generated by scripts/analyse_b3.py on "
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"# {note}\n" + df.to_csv(index=False))
        print(f"wrote {TABLES / path}")

    pd.set_option("display.width", 230)
    print("\n-- B3 curve --")
    print(curve.to_string(index=False))
    print("\n-- Amendment 5 contrasts --")
    print(con.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
