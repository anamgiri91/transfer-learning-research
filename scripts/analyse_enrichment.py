#!/usr/bin/env python
"""precision@10% robustness check — plan.md Amendment 6's five enumerated tests.

§5.7 reports the paper's only positive result: on precision@10% the original
fine-tune `T2` beats `B1` within that endpoint's six-arm scaffold family. The
manuscript said it "does not replicate" on the other splits. It had never been
**evaluated** there — the probe arms had, and that is evidence about the probes.

Amendment 6 fixes five tests and their family (m = 5) before computation:

  1  T2   vs B1, random   -- the arm that produced §5.7's claim, unchanged recipe
  2  T2   vs B1, Butina
  3  T2v  vs B1, scaffold -- additional analysis of runs that already exist
  4  T4ft vs B1, scaffold -- likewise
  5  T5ft vs B1, scaffold -- likewise

Tests 3-5 are an ADDITIONAL ANALYSIS of stored predictions, not new
experiments, and they do not substitute for `T2`. The endpoint is used exactly
as defined in §4.3 and implemented in `evapro.evaluation.metrics`: same top
decile, same k = round(0.10 * n_test), same `np.argsort(-y)` tie handling. No
re-definition, no re-thresholding.

This is a **robustness check on the same 494 compounds, re-partitioned**. It is
not an independent or external replication and is never described as one.

Writes results/tables/table25_enrichment_robustness.csv
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
SEEDS = list(range(10))
N = 347
METRIC = "precision_at_10pct"

# Fixed in Amendment 6 before any of these cells was run.
TESTS = [
    ("T2", "random", "results/metrics/T2_chemberta_full_finetune__random__*n347.json",
     "original T2 recipe, never previously evaluated on this split"),
    ("T2", "butina", "results/metrics/T2_chemberta_full_finetune__butina__*n347.json",
     "original T2 recipe, never previously evaluated on this split"),
    ("T2v", "scaffold", "results/metrics_ft/T2v__scaffold__*n347.json",
     "additional analysis of existing runs; NOT a substitute for T2"),
    ("T4ft", "scaffold", "results/metrics_ft/T4ft__scaffold__*n347.json",
     "additional analysis of existing runs"),
    ("T5ft", "scaffold", "results/metrics_ft/T5ft__scaffold__*n347.json",
     "additional analysis of existing runs"),
]
FAMILY_M = len(TESTS)


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        if f.endswith(".FAILED.json"):
            continue
        d = json.loads(open(f).read())
        out[d["seed"]] = d["metrics"][METRIC]
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


def main() -> int:
    argparse.ArgumentParser().parse_args()
    rows = []
    for arm, split, pat, note in TESTS:
        a = load(pat)
        b = load(f"results/metrics/B1_ecfp_histgb__{split}__*n347.json")
        if not all(s in a for s in SEEDS) or not all(s in b for s in SEEDS):
            rows.append(dict(arm=arm, split=split, status="pending", note=note,
                             n_seeds=len([s for s in SEEDS if s in a])))
            continue
        x = np.array([a[s] for s in SEEDS])
        y = np.array([b[s] for s in SEEDS])
        d = x - y                     # positive => arm BETTER (higher is better)
        p = 1.0 if not np.any(d != 0) else float(stats.wilcoxon(x, y).pvalue)
        rows.append(dict(
            arm=arm, split=split, status="complete", note=note, n_seeds=len(SEEDS),
            arm_median=float(np.median(x)), b1_median=float(np.median(y)),
            median_paired_delta=float(np.median(d)),
            arm_better_in_seeds=int((d > 0).sum()),
            arm_worse_in_seeds=int((d < 0).sum()),
            ties=int((d == 0).sum()), p_raw=float(p)))
    df = pd.DataFrame(rows)
    done = df.status == "complete"
    df["family_m"] = FAMILY_M
    df["family_complete"] = bool(done.all())
    if done.any():
        if done.all():
            adj = np.full(len(df), np.nan)
            adj[done.to_numpy()] = holm(df.loc[done, "p_raw"].to_numpy(float))
            df["p_adjusted"] = adj
            df["adjustment"] = np.where(done, f"Holm, m={FAMILY_M} (complete)", "")
        else:
            df["p_adjusted"] = np.where(
                done, np.minimum(1.0, df.p_raw.astype(float) * FAMILY_M), np.nan)
            df["adjustment"] = np.where(
                done, f"Bonferroni bound, m={FAMILY_M} (incomplete)", "")
        df["verdict"] = np.where(
            ~done, "pending",
            np.where(df.p_adjusted > 0.05, "no detectable difference",
                     np.where(df.median_paired_delta > 0, "arm better", "arm worse")))

    # Pooled sensitivity: §5.7's 31 endpoint tests plus these five.
    if done.all():
        t14 = pd.read_csv("results/tables/table14_all_endpoints.csv", comment="#")
        pooled_in = np.concatenate([t14[t14.p_raw.notna()].p_raw.to_numpy(float),
                                    df.loc[done, "p_raw"].to_numpy(float)])
        pooled = holm(pooled_in)[-FAMILY_M:]
        df.loc[done, "p_holm_pooled_36"] = pooled
        df["pooled_family_m"] = len(pooled_in)

    (TABLES / "table25_enrichment_robustness.csv").write_text(
        f"# generated by scripts/analyse_enrichment.py on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# Amendment 6's five enumerated tests, family m=5. precision@10%, "
        f"n=347, delta = arm - B1 and POSITIVE means arm better. Robustness check "
        f"on the same 494 compounds re-partitioned, not external replication\n"
        + df.to_csv(index=False))
    print(f"wrote {TABLES}/table25_enrichment_robustness.csv")
    pd.set_option("display.width", 230)
    print(df.drop(columns=["note"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
