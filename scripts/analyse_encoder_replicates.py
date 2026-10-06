#!/usr/bin/env python
"""Does the H4 ablation survive re-drawing the encoder? — plan.md Amendment 9.

Every encoder-level contrast in §6.4-6.5 compares two SINGLE pretraining runs,
and the ten downstream seeds resample only the probe's split, so no reported
interval contains any pretraining variance. Amendment 9 adds two replicate
encoders per T4-family condition, varying the initialisation and batch order
only: `--split-seed 0` and `--draw-seed 0` hold the corpus validation split and
the identity of the 61 removed records at their historical values.

The downstream probe is the published one, unchanged: mean-pooled 384-d
embeddings, StandardScaler, then RidgeCV over the same alpha grid, on the
committed scaffold split files at n = 347 across seeds 0-9.

Decision rule, fixed in Amendment 9 before any replicate was trained: the
`T4_clean` - `T4_rand61` contrast is encoder-robust only if its magnitude
exceeds the largest within-condition range of encoder medians.

Writes results/tables/table26_encoder_replicates.csv
       docs/encoder-replicates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from evapro.data.io import load_dataset
from evapro.data.splits import load_split
from evapro.evaluation.metrics import compute_all
from evapro.utils.seeding import set_seed

sys.path.insert(0, "scripts")
from run_arms import indomain_embeddings  # noqa: E402

TARGET = "eva71_2a"
SPLIT_DIR = Path("data/processed/splits") / TARGET
TABLE = Path("results/tables/table26_encoder_replicates.csv")
OUT = Path("docs/encoder-replicates.json")
SEEDS = list(range(10))
N_TRAIN = 347
METRIC_NAMES = ["rmse", "mae", "r2", "spearman", "pearson", "precision_at_10pct"]

# Amendment 9: the T4 family only, R = 3 encoders per condition.
CONDITIONS = {
    "T4": "indomain_T4",
    "T4_clean": "indomain_T4_clean",
    "T4_rand61": "indomain_T4_rand61",
}
REPLICATES = ["", "_init1", "_init2"]
CONTRAST = ("T4_clean", "T4_rand61")


def probe_median_rmse(tag: str, df, y) -> dict:
    """The published frozen probe, one encoder, ten committed scaffold folds."""
    X = indomain_embeddings(tag, df["canonical_smiles"].tolist())
    per_seed = {}
    for seed in SEEDS:
        set_seed(seed)
        folds = load_split(SPLIT_DIR / f"scaffold__seed{seed}.json")
        fold_of = df["inchikey"].map(folds).to_numpy()
        tr = np.flatnonzero(fold_of == "train")
        te = np.flatnonzero(fold_of == "test")
        if len(tr) != N_TRAIN:
            raise ValueError(f"{tag} seed {seed}: train fold is {len(tr)}, not {N_TRAIN}")
        scaler = StandardScaler().fit(X[tr])
        model = RidgeCV(alphas=np.logspace(-2, 4, 25))
        model.fit(scaler.transform(X[tr]), y[tr])
        preds = model.predict(scaler.transform(X[te]))
        per_seed[seed] = compute_all(y[te], preds, METRIC_NAMES)
    rmses = np.array([per_seed[s]["rmse"] for s in SEEDS])
    meta = json.loads(Path(f"models/{tag}.json").read_text())
    return dict(encoder=tag, median_rmse=float(np.median(rmses)),
                mean_rmse=float(rmses.mean()), rmse_sd=float(rmses.std(ddof=1)),
                per_seed_rmse={str(s): per_seed[s]["rmse"] for s in SEEDS},
                init_seed=meta.get("init_seed", meta.get("seed", 0)),
                split_seed=meta.get("split_seed", 0),
                draw_seed=meta.get("draw_seed", 0),
                epochs_run=meta["epochs_run"],
                pretrain_val_rmse=meta["best_val_rmse"],
                n_corpus_used=meta["n_corpus_used"], n_pretrain_val=meta["n_val"])


def build_report() -> dict:
    missing = [f"models/{stem}{r}.pt" for stem in CONDITIONS.values()
               for r in REPLICATES if not Path(f"models/{stem}{r}.pt").exists()]
    if missing:
        raise ValueError(f"incomplete replicate set; {len(missing)} missing: {missing[:3]}")

    df = load_dataset(TARGET)
    y = df["pactivity"].to_numpy()
    rows, by_condition = [], {}
    for condition, stem in CONDITIONS.items():
        encoders = []
        for r in REPLICATES:
            rec = probe_median_rmse(f"{stem}{r}", df, y)
            rec["condition"] = condition
            rec["replicate"] = r or "_init0 (published)"
            encoders.append(rec)
            rows.append(rec)
            print(f"  {rec['encoder']:28s} median RMSE {rec['median_rmse']:.4f} "
                  f"(pretrain val {rec['pretrain_val_rmse']:.4f}, "
                  f"{rec['epochs_run']} epochs)", flush=True)
        meds = np.array([e["median_rmse"] for e in encoders])
        by_condition[condition] = dict(
            n_encoders=len(encoders),
            encoder_medians=[round(float(m), 4) for m in meds],
            condition_mean=float(meds.mean()),
            within_condition_range=float(meds.max() - meds.min()),
            within_condition_sd=float(meds.std(ddof=1)),
            published_encoder_median=float(encoders[0]["median_rmse"]))

    a, b = CONTRAST
    effect = by_condition[a]["condition_mean"] - by_condition[b]["condition_mean"]
    worst_range = max(v["within_condition_range"] for v in by_condition.values())
    published_effect = (by_condition[a]["published_encoder_median"]
                        - by_condition[b]["published_encoder_median"])
    robust = bool(abs(effect) > worst_range)

    pd.DataFrame(rows).drop(columns=["per_seed_rmse"]).to_csv(TABLE, index=False)
    TABLE.write_text(
        f"# generated by scripts/analyse_encoder_replicates.py on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"# plan.md Amendment 9. Frozen probe, scaffold split, n=347, seeds 0-9. "
        f"One row per encoder; init_seed varies within a condition while the "
        f"corpus validation split and the removed-record draw are held fixed\n"
        + TABLE.read_text())

    return dict(
        amendment="plan.md Amendment 9 (2026-10-05)",
        scope="T4 family, R=3 encoders per condition; probe and splits unchanged",
        varied_within_condition="encoder initialisation and batch order only",
        held_fixed="corpus validation split (--split-seed 0) and the identity of "
                   "the 61 removed records (--draw-seed 0)",
        n_downstream_seeds=len(SEEDS), n_train=N_TRAIN,
        conditions=by_condition,
        contrast=dict(
            name=f"{a} minus {b}", condition_mean_difference=round(effect, 4),
            published_single_encoder_difference=round(published_effect, 4),
            largest_within_condition_range=round(worst_range, 4),
            decision_rule="encoder-robust only if |condition difference| exceeds "
                          "the largest within-condition range of encoder medians",
            encoder_robust=robust,
            verdict=("exceeds encoder noise" if robust else "within encoder noise")),
        limits="R=3 supports no encoder-level p-value; this compares the size of "
               "an effect against the size of a nuisance. The removed-record draw "
               "is held fixed, so draw-to-draw uncertainty remains unestimated. "
               "Published per-seed contrasts are unchanged and not recomputed.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        if not OUT.exists():
            print(f"{OUT} is missing")
            return 1
        print(f"{OUT} present (regenerating requires the replicate encoders)")
        return 0
    report = build_report()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    c = report["contrast"]
    print(f"\nwrote {TABLE} and {OUT}")
    for name, v in report["conditions"].items():
        print(f"  {name:10s} medians {v['encoder_medians']}  "
              f"mean {v['condition_mean']:.4f}  range {v['within_condition_range']:.4f}")
    print(f"\n  {c['name']}: condition difference {c['condition_mean_difference']:+.4f}, "
          f"published single-encoder difference {c['published_single_encoder_difference']:+.4f}")
    print(f"  largest within-condition range {c['largest_within_condition_range']:.4f}"
          f"  ->  {c['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
