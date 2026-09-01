#!/usr/bin/env python
"""Materialise split files and verify they do not leak.

Usage: python scripts/build_splits.py --target eva71_3c
Fails loudly rather than writing a split that would inflate every downstream result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evapro.config import load_config
from evapro.data.io import load_dataset
from evapro.data.splits import STRATEGIES, assert_no_leakage, build_split, save_split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--target", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = args.target or cfg["data"]["target"]
    seeds = args.seeds or cfg["seed_list"]

    df = load_dataset(target)

    out_dir = Path(cfg["data"]["processed_dir"]) / "splits" / target
    for strategy in ("scaffold", "random"):
        # Deterministic strategies need only one file.
        strategy_seeds = [0] if strategy == "temporal" else seeds
        for seed in strategy_seeds:
            kwargs = dict(test_frac=cfg["split"]["test_frac"], val_frac=cfg["split"]["val_frac"])
            if strategy != "temporal":
                kwargs["seed"] = seed
            folds = build_split(df, strategy, **kwargs)
            assert_no_leakage(df, folds, strategy)
            save_split(folds, out_dir / f"{strategy}__seed{seed}.json",
                       meta={"target": target, "strategy": strategy, "seed": seed, "n": len(df)})
            print(f"  ok  {strategy:9s} seed={seed}  n={len(df)}")

    print(f"\nsplits written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
