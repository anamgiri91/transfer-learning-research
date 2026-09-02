#!/usr/bin/env python
"""Hyperparameter search with validation-fold model selection (plan.md §5).

The benchmark in run_arms.py uses fixed defaults and never reads the val fold.
That is a fairness problem for a negative transfer result: an under-tuned
transfer arm loses for the wrong reason. This script gives every arm an
explicit search budget, selects on the validation fold, and reports test
performance for the selected config only.

Writes results/tuned_metrics/<arm>__<split>__seed<N>__n<size>.json including
the winning config and the full trial log, so the selection is auditable.

Usage:
  python scripts/tune_arms.py --arms B1 B2 T1 --trials 32
  python scripts/tune_arms.py --arms T2 --trials 8 --sizes 50 347 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from evapro.data.io import load_dataset
from evapro.data.splits import load_split
from evapro.evaluation.metrics import compute_all, rmse
from evapro.features.fingerprints import ecfp, rdkit_descriptors
from evapro.utils.seeding import set_seed

TARGET = "eva71_2a"
SPLIT_DIR = Path("data/processed/splits") / TARGET
OUT = Path("results/tuned_metrics")
CHEMBERTA = "DeepChem/ChemBERTa-77M-MTR"
METRIC_NAMES = ["rmse", "mae", "r2", "spearman", "pearson", "precision_at_10pct"]

_cache: dict[str, np.ndarray] = {}


def features(kind, smiles):
    if kind not in _cache:
        if kind == "ecfp":
            _cache[kind] = ecfp(smiles, radius=2, n_bits=2048, counts=True)
        elif kind == "descriptors":
            _cache[kind] = rdkit_descriptors(smiles)
        elif kind == "chemberta":
            _cache[kind] = chemberta_embeddings(smiles)
    return _cache[kind]


def chemberta_embeddings(smiles, batch_size=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHEMBERTA)
    model = AutoModel.from_pretrained(CHEMBERTA).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            enc = tok(smiles[i:i + batch_size], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.vstack(outs).astype(np.float32)


def sample_space(arm, rng):
    if arm == "B1":
        return dict(learning_rate=float(rng.choice([0.02, 0.05, 0.1, 0.2])),
                    max_leaf_nodes=int(rng.choice([7, 15, 31, 63])),
                    min_samples_leaf=int(rng.choice([2, 5, 10, 20])),
                    l2_regularization=float(rng.choice([0.0, 0.1, 1.0])),
                    max_iter=int(rng.choice([200, 400, 800])))
    if arm == "B2":
        return dict(n_estimators=int(rng.choice([200, 500, 1000])),
                    max_features=float(rng.choice([0.1, 0.3, 0.5, 1.0])),
                    min_samples_leaf=int(rng.choice([1, 2, 5, 10])),
                    max_depth=rng.choice([None, 10, 20]))
    if arm == "T1":
        return dict(alpha=float(10 ** rng.uniform(-2, 4)))
    if arm == "T2":
        return dict(lr=float(rng.choice([1e-5, 3e-5, 1e-4, 3e-4])),
                    epochs=int(rng.choice([20, 40, 80])),
                    batch_size=int(rng.choice([8, 16, 32])))
    raise ValueError(arm)


def fit_eval_sklearn(arm, cfg, Xtr, ytr, Xva, Xte, seed):
    if arm == "B1":
        m = HistGradientBoostingRegressor(early_stopping=True, n_iter_no_change=20,
                                          validation_fraction=0.15, random_state=seed, **cfg)
    elif arm == "B2":
        c = dict(cfg); c["max_depth"] = None if c["max_depth"] is None else int(c["max_depth"])
        m = RandomForestRegressor(n_jobs=-1, random_state=seed, **c)
    elif arm == "T1":
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        m = Ridge(**cfg)
    m.fit(Xtr, ytr)
    return m.predict(Xva), m.predict(Xte)


def fit_eval_t2(cfg, smi_tr, ytr, smi_va, yva, smi_te, seed):
    """Fine-tune with validation-fold early stopping and best-checkpoint restore.

    run_arms.py trains a fixed number of epochs with no validation signal, which
    at n=50 means 40 unchecked passes over 50 examples. Selecting the epoch on
    the val fold is the correct comparison and is what plan.md §5 specifies.
    """
    import copy

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(CHEMBERTA)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHEMBERTA, num_labels=1, problem_type="regression")

    def enc(s):
        e = tok(list(s), padding="max_length", truncation=True, max_length=128,
                return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    ids, mask = enc(smi_tr)
    y = torch.tensor(np.asarray(ytr), dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(ids, mask, y), batch_size=cfg["batch_size"], shuffle=True)
    va_ids, va_mask = enc(smi_va)
    te_ids, te_mask = enc(smi_te)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    best, best_state, patience, since = np.inf, None, 10, 0
    for _ in range(cfg["epochs"]):
        model.train()
        for b_ids, b_mask, b_y in loader:
            opt.zero_grad()
            model(input_ids=b_ids, attention_mask=b_mask, labels=b_y).loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(input_ids=va_ids, attention_mask=va_mask).logits.squeeze(-1).numpy()
        v = rmse(yva, pv)
        if v < best - 1e-4:
            best, best_state, since = v, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(input_ids=va_ids, attention_mask=va_mask).logits.squeeze(-1).numpy()
        pt = model(input_ids=te_ids, attention_mask=te_mask).logits.squeeze(-1).numpy()
    return pv, pt


ARM_FEAT = {"B1": "ecfp", "B2": "descriptors", "T1": "chemberta"}
ARM_LABEL = {"B1": "B1_ecfp_histgb", "B2": "B2_descriptors_rf",
             "T1": "T1_chemberta_linear_probe", "T2": "T2_chemberta_full_finetune"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["scaffold"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 250, 0])  # 0 = full
    ap.add_argument("--trials", type=int, default=32)
    args = ap.parse_args()

    df = load_dataset(TARGET)
    smiles = df["canonical_smiles"].tolist()
    y_all = df["pactivity"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        X_all = features(ARM_FEAT[arm], smiles) if arm in ARM_FEAT else None
        for split in args.splits:
            for seed in args.seeds:
                set_seed(seed)
                folds = load_split(SPLIT_DIR / f"{split}__seed{seed}.json")
                fold_of = df["inchikey"].map(folds).to_numpy()
                tr_all = np.flatnonzero(fold_of == "train")
                va = np.flatnonzero(fold_of == "val")
                te = np.flatnonzero(fold_of == "test")

                for size in args.sizes:
                    rng = np.random.default_rng(seed)
                    tr = tr_all if size in (0, None) or size >= len(tr_all) else \
                        rng.choice(tr_all, size=size, replace=False)
                    tag = f"{ARM_LABEL[arm]}__{split}__seed{seed}__n{len(tr)}"
                    path = OUT / f"{tag}.json"
                    if path.exists():
                        continue

                    t0, trials, best = time.time(), [], None
                    srng = np.random.default_rng(1000 + seed)
                    seen = set()
                    for _ in range(args.trials):
                        cfg = sample_space(arm, srng)
                        key = json.dumps(cfg, sort_keys=True, default=str)
                        if key in seen:
                            continue
                        seen.add(key)
                        if arm == "T2":
                            pv, pt = fit_eval_t2(cfg, [smiles[i] for i in tr], y_all[tr],
                                                 [smiles[i] for i in va], y_all[va],
                                                 [smiles[i] for i in te], seed)
                        else:
                            pv, pt = fit_eval_sklearn(arm, cfg, X_all[tr], y_all[tr],
                                                      X_all[va], X_all[te], seed)
                        v = rmse(y_all[va], pv)
                        trials.append({"config": {k: str(x) for k, x in cfg.items()},
                                       "val_rmse": round(float(v), 5)})
                        if best is None or v < best["val_rmse"]:
                            best = {"config": cfg, "val_rmse": float(v),
                                    "test": compute_all(y_all[te], pt, METRIC_NAMES)}

                    path.write_text(json.dumps({
                        "arm": ARM_LABEL[arm], "target": TARGET, "split": split, "seed": seed,
                        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                        "n_trials": len(trials), "selected_config": {k: str(v) for k, v in best["config"].items()},
                        "selected_val_rmse": round(best["val_rmse"], 5),
                        "metrics": best["test"], "trials": trials,
                        "seconds": round(time.time() - t0, 1),
                        "source_script": "scripts/tune_arms.py",
                    }, indent=2, sort_keys=True))
                    print(f"  {tag}  val={best['val_rmse']:.3f} test_rmse="
                          f"{best['test']['rmse']:.3f} r={best['test']['spearman']:.3f} "
                          f"({time.time()-t0:.0f}s, {len(trials)} trials)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
