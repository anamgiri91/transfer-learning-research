#!/usr/bin/env python
"""Run one arm across seeds and training-set sizes; write metrics JSON.

Usage: python scripts/run_benchmark.py --config config/arm_b1_ecfp_lgbm.yaml
Each run writes results/metrics/<arm>__<split>__seed<N>__n<size>.json, which is
committed so a change in any headline number shows up as a reviewable diff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evapro.config import load_config
from evapro.data.splits import load_split
from evapro.evaluation.metrics import compute_all
from evapro.features.fingerprints import REPRESENTATIONS
from evapro.models.baselines import build_model
from evapro.utils.seeding import set_seed


def featurize(df: pd.DataFrame, spec: dict) -> np.ndarray:
    kind = spec["type"]
    if kind not in REPRESENTATIONS:
        raise NotImplementedError(
            f"representation {kind!r} not implemented yet (declared in config); "
            f"available: {sorted(REPRESENTATIONS)}"
        )
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return REPRESENTATIONS[kind](df["canonical_smiles"].tolist(), **kwargs)


def subsample_train(idx: np.ndarray, n: int | None, rng) -> np.ndarray:
    if n is None or n >= len(idx):
        return idx
    return rng.choice(idx, size=n, replace=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    arm, target = cfg["arm"], cfg["data"]["target"]
    strategy = args.split or cfg["split"]["strategy"]
    label = cfg["data"]["label_col"]

    df = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / f"{target}.parquet")
    X_all = featurize(df, arm["representation"])
    y_all = df[label].to_numpy()

    out_dir = Path(cfg["paths"]["metrics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(cfg["data"]["processed_dir"]) / "splits" / target

    for seed in cfg["seed_list"]:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        seed_for_split = 0 if strategy == "temporal" else seed
        folds = load_split(split_dir / f"{strategy}__seed{seed_for_split}.json")
        fold_of = df["inchikey"].map(folds).to_numpy()

        tr_all = np.flatnonzero(fold_of == "train")
        te = np.flatnonzero(fold_of == "test")

        for size in cfg["curve"]["train_sizes"]:
            tr = subsample_train(tr_all, size, rng)
            model = build_model(arm["model"])
            model.fit(X_all[tr], y_all[tr])
            scores = compute_all(y_all[te], model.predict(X_all[te]), cfg["eval"]["metrics"])

            record = {
                "arm": arm["id"], "target": target, "split": strategy, "seed": seed,
                "n_train": int(len(tr)), "n_test": int(len(te)), "metrics": scores,
            }
            tag = f"{arm['id']}__{strategy}__seed{seed}__n{len(tr)}"
            (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=2, sort_keys=True))
            print(f"  {tag}  rmse={scores.get('rmse', float('nan')):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
