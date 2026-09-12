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
    d = a - b
    if not np.any(d != 0):
        return 1.0, "degenerate"
    nz = d[d != 0]
    ties = len(np.unique(np.abs(nz))) < len(nz)
    method = ("exhaustive permutations" if (ties or len(nz) < len(d))
              else "exact") if len(nz) <= 13 else "asymptotic"
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


def contrasts(M) -> pd.DataFrame:
    """The amended family: paired, Holm-corrected within itself, reported apart
    from the pre-registered families rather than merged into them."""
    b1 = load_published("B1_ecfp_histgb")
    t2 = load_published("T2_chemberta_full_finetune")
    rows = []

    def add(arm, ref_name, get_ref, n, what):
        if not all((s, n) in M.get(arm, {}) for s in SEEDS):
            return
        a = np.array([M[arm][(s, n)]["metrics"]["rmse"] for s in SEEDS])
        try:
            b = np.array([get_ref(s, n) for s in SEEDS])
        except KeyError:
            return
        p, method = wilcoxon(a, b)
        rows.append(dict(arm=arm, reference=ref_name, n_train=n, what=what,
                         n_seeds=len(SEEDS),
                         median_delta=round(float(np.median(b - a)), 4),
                         arm_better_in_seeds=int((a < b).sum()),
                         wilcoxon_method=method, p_raw=round(p, 4)))

    for n in SIZES:
        add("T2v", "B1", lambda s, k: b1[(s, k)]["metrics"]["rmse"], n,
            "corrected generic fine-tune vs the baseline (H2/H1)")
    add("T4ft", "T2v", lambda s, k: M["T2v"][(s, k)]["metrics"]["rmse"], 347,
        "in-domain vs generic pretraining, adaptation matched (H3, pre-registered form)")
    add("T5ft", "T2v", lambda s, k: M["T2v"][(s, k)]["metrics"]["rmse"],
        347, "chained vs generic pretraining, adaptation matched (H3)")
    add("T2v", "T2", lambda s, k: t2[(s, k)]["metrics"]["rmse"], 347,
        "two conditions: schedule AND readout differ, not an isolation")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df.p_raw.to_numpy(float)).round(4)
        df["verdict"] = np.where(df.p_holm > 0.05, "inconclusive",
                                 np.where(df.median_delta > 0, "arm better",
                                          "arm worse"))
    return df


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

    if comp.missing.sum() and not args.allow_partial:
        print(f"\n{int(comp.missing.sum())} planned cells are still missing. "
              f"No test statistics are computed until they exist; re-run when "
              f"the sweep completes, or pass --allow-partial to override "
              f"(which changes what the numbers mean).")
        return 2

    M = load_ft()
    cur, con = curve_table(M), contrasts(M)
    write(cur, TABLES / "table17_amended_finetune.csv",
          "Amendment 4 arms: validation-selected lr and checkpoint, scaffold split, "
          "seeds 0-9. AMENDED, not pre-registered")
    write(con, TABLES / "table18_amended_contrasts.csv",
          "paired Wilcoxon over seeds within the amended family, Holm-corrected "
          "within that family only and reported apart from the pre-registered ones")
    pd.set_option("display.width", 220)
    print("\n-- amended curve --")
    print(cur.to_string(index=False))
    print("\n-- amended contrasts --")
    print(con.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
