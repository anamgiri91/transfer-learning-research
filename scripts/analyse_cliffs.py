#!/usr/bin/env python
"""Activity-cliff subset analysis (plan.md §7.2).

The protocol asks for "performance restricted to matched molecular pairs with
>1 log activity difference". A plain cliff/non-cliff split of the test fold
would confound two different kinds of hardness, so we stratify each test
compound by its relationship to the *training* fold -- which is what the model
actually had to work with:

  cliff    at least one training neighbour at Tanimoto >= T whose |dpK_D| > 1
  smooth   at least one training neighbour at Tanimoto >= T, none of them cliffs
  distant  no training neighbour at Tanimoto >= T at all

`distant` is the honest control: without it, "non-cliff" silently mixes
compounds the model could interpolate with compounds it had never seen the
neighbourhood of.

T = 0.7 is the primary threshold, reused rather than newly chosen: it is the
same cut already fixed for the near-neighbour split audit (Table 0, §5.5).
0.6 and 0.8 are reported as sensitivity.

Reads results/predictions/*.npz (written by run_arms.py --save-preds).
Writes results/tables/table6_activity_cliffs.csv
   and results/tables/table7_cliff_pairs.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from evapro.data.io import load_dataset
from evapro.data.splits import load_split
from evapro.evaluation.cliffs import cliff_census, similarity_matrix, stratify
from evapro.evaluation.stats import paired_compare

RDLogger.DisableLog("rdApp.*")

TARGET = "eva71_2a"
SPLIT_DIR = Path("data/processed/splits") / TARGET
PREDS = Path("results/predictions")
OUT_STRATA = Path("results/tables/table6_activity_cliffs.csv")
OUT_PAIRS = Path("results/tables/table7_cliff_pairs.csv")
OUT_TESTS = Path("results/tables/table8_cliff_paired.csv")

DELTA = 1.0                     # log units; plan.md §7.2
THRESHOLDS = [0.6, 0.7, 0.8]
PRIMARY = 0.7
MIN_STRATUM = 5                 # below this an RMSE is too noisy to report per seed
BASELINE = "B1_ecfp_histgb"     # the arm to beat (plan.md §4)
BASELINE0 = "B0_median"         # the variance floor, for skill normalisation


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--n-train", type=int, default=347)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--require-seeds", type=int, default=None,
                    help="Drop any arm without predictions for this many seeds "
                         "(default: all requested). An interim read of a "
                         "part-finished arm has produced a wrong headline on this "
                         "project before -- see docs/decision-log.md.")
    args = ap.parse_args()

    df = load_dataset(TARGET)
    y = df["pactivity"].to_numpy()
    keys = df["inchikey"].to_numpy()
    key_pos = {k: i for i, k in enumerate(keys)}
    sim = similarity_matrix(df["canonical_smiles"].tolist())

    # ---- dataset-level cliff census (independent of any split) --------------
    pairs = pd.DataFrame([cliff_census(sim, y, t, DELTA) for t in THRESHOLDS])

    # ---- per-arm, per-stratum performance -----------------------------------
    need = args.require_seeds if args.require_seeds is not None else len(args.seeds)
    found: dict[str, int] = {}
    for f in PREDS.glob(f"*__{args.split}__seed*__n{args.n_train}.npz"):
        arm, _, seed_tag, _ = f.stem.split("__")
        if int(seed_tag.removeprefix("seed")) in args.seeds:
            found[arm] = found.get(arm, 0) + 1
    arms = sorted(a for a, k in found.items() if k >= need)
    for a, k in sorted(found.items()):
        if k < need:
            print(f"  skipping {a}: {k}/{need} seeds -- incomplete arms are not reported")
    if not arms:
        raise SystemExit(f"no arm has {need} seeds of predictions in {PREDS} for "
                         f"split={args.split} n={args.n_train}; "
                         f"run run_arms.py --save-preds first")

    rows, test_rows = [], []
    for t in THRESHOLDS:
        per_arm: dict[str, dict[str, list[float]]] = {}
        counts: dict[str, list[int]] = {"cliff": [], "smooth": [], "distant": []}
        for seed in args.seeds:
            folds = load_split(SPLIT_DIR / f"{args.split}__seed{seed}.json")
            fold_of = pd.Series(keys).map(folds).to_numpy()
            train_idx = np.flatnonzero(fold_of == "train")
            test_idx = np.flatnonzero(fold_of == "test")
            strata = stratify(test_idx, train_idx, sim, y, t, DELTA)
            for s in counts:
                counts[s].append(int((strata == s).sum()))

            for arm in arms:
                f = PREDS / f"{arm}__{args.split}__seed{seed}__n{args.n_train}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                # Align the stored predictions to our test ordering by InChIKey.
                order = np.array([key_pos[k] for k in z["inchikey"].astype(str)])
                assert np.array_equal(np.sort(order), np.sort(test_idx)), \
                    f"{f.name}: test fold disagrees with the split file"
                pred = np.empty(len(y)); pred[order] = z["y_pred"]
                bucket = per_arm.setdefault(arm, {})
                for s in ("cliff", "smooth", "distant", "all"):
                    m = np.ones(len(strata), bool) if s == "all" else (strata == s)
                    if m.sum() < MIN_STRATUM:
                        continue
                    ii = test_idx[m]
                    bucket.setdefault(s, []).append(rmse(y[ii], pred[ii]))

        # Strata differ in label variance -- the median predictor alone spans
        # 0.72 to 1.32 RMSE across them -- so raw RMSE is not comparable between
        # strata any more than it is between splits (§5.5). Skill against B0 on
        # the *same* compounds is, so it is reported alongside.
        floor = per_arm.get(BASELINE0, {})
        for arm, buckets in sorted(per_arm.items()):
            for s, vals in buckets.items():
                base = floor.get(s)
                skill = ([1.0 - a / b for a, b in zip(vals, base)]
                         if base and len(base) == len(vals) else None)
                rows.append({
                    "tanimoto_threshold": t, "arm": arm, "stratum": s,
                    "n_seeds": len(vals),
                    "median_compounds": (float(np.median(counts[s]))
                                         if s != "all" else float(np.median(
                                             [sum(counts[k][i] for k in counts)
                                              for i in range(len(args.seeds))]))),
                    "rmse_median": round(float(np.median(vals)), 4),
                    "rmse_iqr": round(float(np.subtract(*np.percentile(vals, [75, 25]))), 4),
                    "skill_vs_b0_median": (round(float(np.median(skill)), 4)
                                           if skill is not None else None),
                })

        # Paired across seeds, arm vs B1, within each stratum. Holm is applied
        # over the arm family separately per stratum: the strata are separate
        # questions, not one family.
        for s in ("cliff", "smooth", "distant", "all"):
            scores = {a: b[s] for a, b in per_arm.items()
                      if s in b and a != BASELINE0 and len(b[s]) == len(args.seeds)}
            if BASELINE not in scores or len(scores) < 2:
                continue
            for c in paired_compare(scores, baseline=BASELINE, lower_is_better=True):
                test_rows.append({
                    "tanimoto_threshold": t, "stratum": s, "arm": c.arm,
                    "baseline": c.baseline, "n_seeds": c.n_seeds,
                    "median_delta_rmse": round(c.median_delta, 4),
                    "p_raw": round(c.p_raw, 4), "p_holm": round(c.p_holm, 4),
                    "verdict": ("baseline better" if c.p_holm <= 0.05 and c.median_delta < 0
                                else "arm better" if c.p_holm <= 0.05
                                else "inconclusive"),
                })

    strata_df = pd.DataFrame(rows)
    tests_df = pd.DataFrame(test_rows)
    stamp = (f"# generated by scripts/analyse_cliffs.py on "
             f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
             f"# split={args.split} n_train={args.n_train} seeds={len(args.seeds)} "
             f"delta_p>{DELTA} primary_threshold={PRIMARY}\n")
    for path, frame in ((OUT_PAIRS, pairs), (OUT_STRATA, strata_df), (OUT_TESTS, tests_df)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stamp + frame.to_csv(index=False))
        print(f"wrote {path} ({len(frame)} rows)")

    print(pairs.to_string(index=False))
    print(strata_df[strata_df.tanimoto_threshold == PRIMARY].to_string(index=False))
    print(tests_df[tests_df.tanimoto_threshold == PRIMARY].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
