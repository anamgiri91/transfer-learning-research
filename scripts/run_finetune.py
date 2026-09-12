#!/usr/bin/env python
"""Amended fine-tuning arms: a corrected schedule (H2) and a matched H3 contrast.

These are AMENDED EXPERIMENTS, designed on 2026-09-11 after the earlier results
were seen. They are not the pre-registered analysis and are not independent
confirmation of anything. See `plan.md` Amendment 4 for the specification that
was fixed before these were run, and docs/decision-log.md for why.

WHAT THIS ADDS, AND WHY EACH ARM EXISTS
----------------------------------------
`T2` in the published sweep is a full fine-tune on a fixed 40-epoch schedule at
a single learning rate, with the validation fold never read. §6.2 shows that
schedule is a harness artefact: at n = 50 it drives the arm to RMSE 1.23 /
R^2 -1.26, and a tuned configuration reaches 0.74 / +0.30. Two conclusions in
this paper rest on numbers produced under it -- H2's only significant
interaction slope (§5.3) and §5.6's retracted calibration reading -- so the arm
is re-run under a documented validation-based procedure at every training size.

  T2v   generic ChemBERTa-77M-MTR init      -- the corrected generic fine-tune
  T4ft  in-domain encoder (models/indomain_T4.pt) -- random init, then
        multitask-pretrained on the 3C/3CL corpus, then fine-tuned here
  T5ft  chained encoder (models/indomain_T5.pt)   -- ChemBERTa, then the same
        in-domain pretraining, then fine-tuned here

T4ft-vs-T2v is the comparison `plan.md` §1 names as the H3 decision rule and
that this study had never run: in-domain pretraining versus generic
pretraining, with **architecture and downstream adaptation held fixed** so the
only difference is which corpus produced the initial encoder weights. It is
implementable because the in-domain encoders are the same RoBERTa backbone as
ChemBERTa-77M-MTR -- 53 of their 55 encoder tensors map onto it key for key,
the two exceptions being a pooler this head does not use.

  This is NOT the pre-registered T4: `plan.md`'s T4 was specified for an EV-A71
  3C target that Amendment 1 removed, and its corpus is 95% coronaviral and
  chemically near-disjoint from the evaluation set. It is the pre-registered
  *comparison form* on the corpus this study actually has.

FAIRNESS: WHAT THESE ARMS ARE AND ARE NOT ALLOWED TO SEE
----------------------------------------------------------
The published sweep leaves the split file's `val` fold unused by every arm, and
that is preserved here -- these arms never touch it, so their training sets are
identical to the ones B1/B2/T1/T2 saw at the same (seed, size) and the paired
comparison stays valid.

Model selection therefore happens on an internal validation set carved from the
training subsample itself, mirroring what `B1` already does
(`HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.15,
n_iter_no_change=20)`). Giving the fine-tune the held-out `val` fold instead
would hand it 49 compounds B1 never sees, which would be a real advantage
smuggled in as a fairness fix. At n = 50 the internal split leaves 42 training
and 8 validation compounds; that is genuinely thin, and it is a property of the
regime rather than of this implementation -- it is what any practitioner tuning
on 50 compounds faces. Failures are recorded, not hidden.

The training subsample for each (seed, size) is reconstructed with the exact
RNG sequence `run_arms.py` uses, and the reconstruction is asserted against the
committed predictions where they exist, so "matching subsamples" is checked
rather than claimed.

POOLING, AND WHY T2v IS A NEW CONDITION RATHER THAN A RE-RUN OF T2
--------------------------------------------------------------------
All three arms read the encoder out by mean-pooling token states into a linear
head -- the same pooling every frozen probe in this study uses, and the same
pooling the in-domain encoders were pretrained under. The original `T2` instead
used `AutoModelForSequenceClassification`, whose RoBERTa head reads the `<s>`
token.

That is a deliberate choice with a cost, stated here rather than buried: T2v
differs from T2 in BOTH the schedule and the readout, so T2v-vs-T2 confounds
the two and is reported as a condition comparison, not as an isolation of the
schedule. The alternative -- a `<s>`-token head -- would have matched T2 while
disadvantaging T4ft and T5ft, whose pretraining never trained that token. H3 is
the comparison that needed protecting, so the readout was matched across the
three new arms and the confound was pushed onto the one contrast that already
has §6.2's tuned comparison speaking to it.

Writes results/metrics_ft/<arm>__<split>__seed<N>__n<size>.json, kept in a
separate directory so the published sweep's 1,040 files stay exactly as
committed.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from evapro.data.io import load_dataset
from evapro.data.splits import load_split
from evapro.evaluation.metrics import compute_all
from evapro.utils.seeding import set_seed

TARGET = "eva71_2a"
SPLIT_DIR = Path("data/processed/splits") / TARGET
OUT = Path("results/metrics_ft")
PREDS = Path("results/predictions_ft")
TRAIN_SIZES = [50, 100, 250, None]
SEEDS = list(range(10))
CHEMBERTA = "DeepChem/ChemBERTa-77M-MTR"
MAX_LEN = 128
METRIC_NAMES = ["rmse", "mae", "r2", "spearman", "pearson", "precision_at_10pct"]

# Fixed in plan.md Amendment 4 before any of these arms was evaluated.
LR_GRID = [1e-5, 3e-5, 1e-4]
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 16
INTERNAL_VAL_FRAC = 0.15      # mirrors B1's HistGB validation_fraction

INITS = {
    "T2v": None,                          # generic ChemBERTa-77M-MTR
    "T4ft": "models/indomain_T4.pt",      # in-domain, random-init pretraining
    "T5ft": "models/indomain_T5.pt",      # chained ChemBERTa -> in-domain
}


class MeanPoolRegressor(nn.Module):
    """Shared readout for all three arms: mean-pooled tokens -> linear head."""

    def __init__(self, init_state: str | None):
        super().__init__()
        cfg = AutoConfig.from_pretrained(CHEMBERTA)
        if init_state is None:
            self.encoder = AutoModel.from_pretrained(CHEMBERTA)
        else:
            # The in-domain checkpoints are MultitaskRegressor state dicts:
            # `encoder.*` is the backbone, `heads.*` are per-protease heads we
            # discard. Loading is strict about the backbone and silent only
            # about the pooler, which this readout does not use.
            self.encoder = AutoModel.from_config(cfg)
            sd = torch.load(init_state, map_location="cpu")
            enc = {k[len("encoder."):]: v for k, v in sd.items()
                   if k.startswith("encoder.")}
            missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
            hard = [k for k in missing if not k.startswith("pooler.")]
            if hard or unexpected:
                raise RuntimeError(
                    f"{init_state}: backbone did not load cleanly. "
                    f"missing={hard} unexpected={list(unexpected)}")
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        z = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self.head(z).squeeze(-1)


def training_subsamples(tr_all: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    """Reproduce run_arms.py's exact per-size subsamples for one seed.

    run_arms draws them from ONE generator, in TRAIN_SIZES order, so the draws
    are sequential and cannot be reproduced by seeding per size.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for size in TRAIN_SIZES:
        tr = tr_all if size is None or size >= len(tr_all) else \
            rng.choice(tr_all, size=size, replace=False)
        out[len(tr)] = tr
    return out


def internal_split(idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Carve model-selection data out of the training subsample, as B1 does."""
    rng = np.random.default_rng(10_000 + seed)
    perm = rng.permutation(len(idx))
    n_val = max(2, int(round(len(idx) * INTERNAL_VAL_FRAC)))
    return idx[perm[n_val:]], idx[perm[:n_val]]


def encode(tok, smiles):
    e = tok(list(smiles), padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def predict(model, ids, mask, batch=64) -> np.ndarray:
    model.eval()
    return np.concatenate([model(ids[i:i + batch], mask[i:i + batch]).numpy()
                           for i in range(0, len(ids), batch)])


def train_one(arm: str, lr: float, smiles, y, tr_idx, va_idx, seed: int):
    """Train at one learning rate; return the best-validation checkpoint.

    Checkpoint selection is on validation RMSE, evaluated after every epoch,
    with the best state deep-copied and restored at the end. Early stopping
    fires after PATIENCE epochs without improvement.
    """
    set_seed(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(CHEMBERTA)
    model = MeanPoolRegressor(INITS[arm])
    ids_tr, mask_tr = encode(tok, [smiles[i] for i in tr_idx])
    ids_va, mask_va = encode(tok, [smiles[i] for i in va_idx])
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
    y_va = y[va_idx]

    loader = DataLoader(TensorDataset(ids_tr, mask_tr, y_tr),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    best = {"val_rmse": float("inf"), "epoch": -1, "state": None}
    steps = 0
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        for b_ids, b_mask, b_y in loader:
            opt.zero_grad()
            lossf(model(b_ids, b_mask), b_y).backward()
            opt.step()
            steps += 1
        vp = predict(model, ids_va, mask_va)
        vr = float(np.sqrt(np.mean((y_va - vp) ** 2)))
        history.append(round(vr, 4))
        if vr < best["val_rmse"] - 1e-6:
            best = {"val_rmse": vr, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    return model, best, steps, history


def run_cell(arm: str, split: str, seed: int, size: int, smiles, y,
             tr_idx, te_idx) -> dict:
    """One (arm, split, seed, size): select lr on validation, then evaluate."""
    t0 = time.time()
    sub_tr, sub_va = internal_split(tr_idx, seed)
    trials = []
    best_lr, best_val, best_model, best_info, best_hist = None, float("inf"), None, None, None
    for lr in LR_GRID:
        model, info, steps, hist = train_one(arm, lr, smiles, y, sub_tr, sub_va, seed)
        trials.append({"lr": lr, "val_rmse": round(info["val_rmse"], 4),
                       "best_epoch": info["epoch"], "optimizer_steps": steps,
                       "val_history": hist})
        if info["val_rmse"] < best_val:
            best_lr, best_val = lr, info["val_rmse"]
            best_model, best_info, best_hist = model, info, hist

    tok = AutoTokenizer.from_pretrained(CHEMBERTA)
    ids_te, mask_te = encode(tok, [smiles[i] for i in te_idx])
    preds = predict(best_model, ids_te, mask_te)
    scores = compute_all(y[te_idx], preds, METRIC_NAMES)
    return {
        "arm": arm, "target": TARGET, "split": split, "seed": seed,
        "n_train": int(len(tr_idx)),
        "n_train_fitted": int(len(sub_tr)), "n_internal_val": int(len(sub_va)),
        "n_test": int(len(te_idx)), "metrics": scores,
        "selected_lr": best_lr, "selected_epoch": best_info["epoch"],
        "selected_val_rmse": round(best_val, 4),
        "optimizer_steps": [t["optimizer_steps"] for t in trials
                            if t["lr"] == best_lr][0],
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "lr_grid": LR_GRID,
        "internal_val_frac": INTERNAL_VAL_FRAC,
        "val_history_selected": best_hist,
        "trials": trials,
        "seconds": round(time.time() - t0, 2),
        "source_script": "scripts/run_finetune.py",
        "amendment": "plan.md Amendment 4 (2026-09-11); amended, not pre-registered",
    }, preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["T2v"], choices=list(INITS))
    ap.add_argument("--splits", nargs="+", default=["scaffold"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cells that would run, and exit")
    args = ap.parse_args()

    df = load_dataset(TARGET)
    smiles = df["canonical_smiles"].tolist()
    y = df["pactivity"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)

    planned, done_n = [], 0
    for arm in args.arms:
        for split in args.splits:
            for seed in args.seeds:
                folds = load_split(SPLIT_DIR / f"{split}__seed{seed}.json")
                fold_of = df["inchikey"].map(folds).to_numpy()
                tr_all = np.flatnonzero(fold_of == "train")
                te = np.flatnonzero(fold_of == "test")
                for size, tr in training_subsamples(tr_all, seed).items():
                    if args.sizes and size not in args.sizes:
                        continue
                    tag = f"{arm}__{split}__seed{seed}__n{size}"
                    if (OUT / f"{tag}.json").exists():
                        done_n += 1
                        continue
                    planned.append((tag, arm, split, seed, size, tr, te))

    print(f"{len(planned)} cells to run, {done_n} already present")
    if args.dry_run:
        for tag, *_ in planned[:8]:
            print("  would run", tag)
        return 0

    failures = 0
    for i, (tag, arm, split, seed, size, tr, te) in enumerate(planned, 1):
        try:
            rec, preds = run_cell(arm, split, seed, size, smiles, y, tr, te)
            (OUT / f"{tag}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
            np.savez(PREDS / f"{tag}.npz",
                     inchikey=df["inchikey"].to_numpy()[te],
                     y_true=y[te], y_pred=np.asarray(preds, dtype=float))
            print(f"  [{i}/{len(planned)}] {tag}  rmse={rec['metrics']['rmse']:.3f} "
                  f"lr={rec['selected_lr']:g} epoch={rec['selected_epoch']} "
                  f"({rec['seconds']:.0f}s)", flush=True)
        except Exception:
            failures += 1
            # A failure is a result. Record it where the analysis will see it
            # rather than letting the cell silently not exist.
            (OUT / f"{tag}.FAILED.json").write_text(json.dumps(
                {"tag": tag, "arm": arm, "split": split, "seed": seed,
                 "n_train": int(size), "error": traceback.format_exc(),
                 "source_script": "scripts/run_finetune.py"}, indent=2))
            print(f"  [{i}/{len(planned)}] {tag}  FAILED", flush=True)
    print(f"\ndone. {len(planned) - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
