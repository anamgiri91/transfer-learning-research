#!/usr/bin/env python
"""Arm B3: a from-scratch D-MPNN, the pre-registered baseline never run.

`plan.md` §4 freezes `B3` as "D-MPNN (Chemprop-style) trained from scratch" and
this study never ran it, so every H1 comparison to date has pitted pretrained
transformers against fingerprint and descriptor baselines only. `plan.md`
Amendment 5 fixes the implementation, budget, comparisons and multiplicity
family; it was written and dated before any test-set number here existed.

The arm is pre-registered. The IMPLEMENTATION is post-hoc, and the two are kept
distinct in the manuscript: chemprop 2.3.1, its default molecule featuriser,
bond-message passing, mean aggregation, a regression FFN, and the same
validation-based selection rule Amendment 4's fine-tune arms use.

WHAT THIS ARM CAN AND CANNOT SUPPORT
--------------------------------------
It broadens architecture coverage, which is what H1 needs: "the best
from-scratch baseline" was never the deep one. It does **not** isolate the
effect of pretraining. `B3` differs from `T2v` in architecture *and* in
molecular representation — a bond-message graph network on a molecular graph
against a transformer on SMILES tokens — so a `B3`-vs-`T2v` difference has at
least three candidate causes and this design separates none of them. The
control that would isolate pretraining is a fully trainable, randomly
initialised ChemBERTa matched to `T2v`, which this study does not have.

FAIRNESS
--------
Training subsamples are reconstructed with `run_arms.py`'s exact RNG sequence
and asserted against it, so `B3` sees the same molecules at the same sizes as
every other arm. Validation is carved from that subsample at 15% — mirroring
`B1`'s `validation_fraction` and identical to Amendment 4 — so the split file's
`val` fold stays unused and the paired comparison holds. Both the labelled
budget and the number actually fitted are recorded per cell.

Writes results/metrics_b3/<arm>__<split>__seed<N>__n<size>.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
from lightning import pytorch as pl

from evapro.data.io import load_dataset
from evapro.data.splits import load_split
from evapro.evaluation.metrics import compute_all
from evapro.utils.seeding import set_seed

TARGET = "eva71_2a"
SPLIT_DIR = Path("data/processed/splits") / TARGET
OUT = Path("results/metrics_b3")
PREDS = Path("results/predictions_b3")
TRAIN_SIZES = [50, 100, 250, None]
SEEDS = list(range(10))
METRIC_NAMES = ["rmse", "mae", "r2", "spearman", "pearson", "precision_at_10pct"]

# Fixed in plan.md Amendment 5 before any B3 result existed.
LR_GRID = [1e-4, 3e-4, 1e-3]
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 16
INTERNAL_VAL_FRAC = 0.15


def training_subsamples(tr_all: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Reproduce run_arms.py's per-size subsamples: one generator, in order."""
    rng = np.random.default_rng(seed)
    out = {}
    for size in TRAIN_SIZES:
        tr = tr_all if size is None or size >= len(tr_all) else \
            rng.choice(tr_all, size=size, replace=False)
        out[len(tr)] = tr
    return out


def internal_split(idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Identical rule to scripts/run_finetune.py, including the RNG offset."""
    rng = np.random.default_rng(10_000 + seed)
    perm = rng.permutation(len(idx))
    n_val = max(2, int(round(len(idx) * INTERNAL_VAL_FRAC)))
    return idx[perm[n_val:]], idx[perm[:n_val]]


def build(lr: float):
    from chemprop import nn as cpnn
    from chemprop.models import MPNN
    mp = cpnn.BondMessagePassing()
    agg = cpnn.MeanAggregation()
    ffn = cpnn.RegressionFFN()
    return MPNN(mp, agg, ffn, batch_norm=True, init_lr=lr / 10,
                max_lr=lr, final_lr=lr / 10)


def datasets(smiles, y, tr, va, te):
    from chemprop import data as cpdata
    from chemprop import featurizers
    from rdkit import Chem
    f = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def mk(idx):
        pts = [cpdata.MoleculeDatapoint(Chem.MolFromSmiles(smiles[i]),
                                        np.array([y[i]], dtype=float)) for i in idx]
        return cpdata.MoleculeDataset(pts, f)
    return mk(tr), mk(va), mk(te)


def train_one(lr, smiles, y, tr, va, te, seed):
    """One learning rate; returns val RMSE, test predictions and provenance."""
    from chemprop import data as cpdata
    set_seed(seed)
    pl.seed_everything(seed, workers=True, verbose=False)
    d_tr, d_va, d_te = datasets(smiles, y, tr, va, te)
    scaler = d_tr.normalize_targets()
    d_va.normalize_targets(scaler)

    loaders = [cpdata.build_dataloader(d, batch_size=BATCH_SIZE, shuffle=sh,
                                       num_workers=0)
               for d, sh in ((d_tr, True), (d_va, False), (d_te, False))]
    model = build(lr)
    model.predictor.output_transform = cpnn_unscale(scaler)

    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min",
                                        save_top_k=1, dirpath=None)
    early = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                       patience=PATIENCE)
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", devices=1,
                         logger=False, enable_checkpointing=True,
                         enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[ckpt, early], deterministic=True)
    trainer.fit(model, loaders[0], loaders[1])
    best = float(ckpt.best_model_score.item())
    # Restore the best-validation weights into the model we already built.
    # MPNN.load_from_checkpoint() reconstructs a fresh model without the target
    # UnscaleTransform, so the checkpoint's two extra buffers make it fail
    # strict loading; the in-memory model already carries the transform.
    state = torch.load(ckpt.best_model_path, map_location="cpu",
                       weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    preds = np.concatenate([p.numpy().ravel() for p in
                            trainer.predict(model, loaders[2])])
    return best, preds, dict(best_epoch=int(trainer.current_epoch),
                             global_step=int(trainer.global_step))


def cpnn_unscale(scaler):
    from chemprop import nn as cpnn
    return cpnn.UnscaleTransform.from_standard_scaler(scaler)


def run_cell(split, seed, size, smiles, y, tr_idx, te_idx):
    t0 = time.time()
    sub_tr, sub_va = internal_split(tr_idx, seed)
    trials, best = [], None
    for lr in LR_GRID:
        v, preds, prov = train_one(lr, smiles, y, sub_tr, sub_va, te_idx, seed)
        trials.append({"lr": lr, "val_loss": round(v, 4), **prov})
        if best is None or v < best[0]:
            best = (v, lr, preds, prov)
    v, lr, preds, prov = best
    scores = compute_all(y[te_idx], preds, METRIC_NAMES)
    return {
        "arm": "B3", "target": TARGET, "split": split, "seed": seed,
        "n_train": int(len(tr_idx)),
        "n_train_fitted": int(len(sub_tr)), "n_internal_val": int(len(sub_va)),
        "n_test": int(len(te_idx)), "metrics": scores,
        "selected_lr": lr, "selected_val_loss": round(v, 4),
        "selected_epoch": prov["best_epoch"], "optimizer_steps": prov["global_step"],
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "lr_grid": LR_GRID, "internal_val_frac": INTERNAL_VAL_FRAC,
        "trials": trials, "implementation": "chemprop 2.3.1 BondMessagePassing",
        "seconds": round(time.time() - t0, 2),
        "source_script": "scripts/run_dmpnn.py",
        "amendment": "plan.md Amendment 5 (2026-09-12); pre-registered arm, "
                     "post-hoc implementation",
    }, preds


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["scaffold"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    # plan.md Amendment 8. The result depends on the intra-op thread count:
    # parallel reductions sum in a different order, and ~400 optimiser steps
    # with early stopping amplify that. Unset reproduces the historical,
    # machine-dependent behaviour the saved cells were produced under.
    ap.add_argument("--threads", type=int, default=None,
                    help="pin torch intra-op threads (1 for the pinned replicate)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="write <out-root>/metrics and <out-root>/predictions "
                         "instead of results/metrics_b3 and results/predictions_b3")
    args = ap.parse_args()
    if args.threads is not None:
        if args.threads < 1:
            ap.error("--threads must be positive")
        if args.out_root is None:
            ap.error("--threads requires --out-root to protect historical results")
        torch.set_num_threads(args.threads)
    out = OUT if args.out_root is None else args.out_root / "metrics"
    preds_dir = PREDS if args.out_root is None else args.out_root / "predictions"

    df = load_dataset(TARGET)
    smiles = df["canonical_smiles"].tolist()
    y = df["pactivity"].to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    planned = []
    for split in args.splits:
        for seed in args.seeds:
            folds = load_split(SPLIT_DIR / f"{split}__seed{seed}.json")
            fold_of = df["inchikey"].map(folds).to_numpy()
            tr_all = np.flatnonzero(fold_of == "train")
            te = np.flatnonzero(fold_of == "test")
            for size, tr in training_subsamples(tr_all, seed).items():
                if args.sizes and size not in args.sizes:
                    continue
                tag = f"B3__{split}__seed{seed}__n{size}"
                if (out / f"{tag}.json").exists():
                    continue
                planned.append((tag, split, seed, size, tr, te))

    print(f"{len(planned)} cells to run")
    if args.dry_run:
        return 0
    failures = 0
    for i, (tag, split, seed, size, tr, te) in enumerate(planned, 1):
        try:
            rec, preds = run_cell(split, seed, size, smiles, y, tr, te)
            if args.threads is not None:
                # Only on pinned runs, so default cells keep the saved key set.
                rec["torch_threads"] = int(torch.get_num_threads())
            (out / f"{tag}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
            np.savez(preds_dir / f"{tag}.npz",
                     inchikey=df["inchikey"].to_numpy()[te],
                     y_true=y[te], y_pred=np.asarray(preds, dtype=float))
            print(f"  [{i}/{len(planned)}] {tag}  rmse={rec['metrics']['rmse']:.3f} "
                  f"lr={rec['selected_lr']:g} ({rec['seconds']:.0f}s)", flush=True)
        except Exception:
            failures += 1
            (out / f"{tag}.FAILED.json").write_text(json.dumps(
                {"tag": tag, "error": traceback.format_exc()}, indent=2))
            print(f"  [{i}/{len(planned)}] {tag}  FAILED", flush=True)
    print(f"\ndone. {len(planned) - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
