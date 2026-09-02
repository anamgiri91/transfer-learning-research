#!/usr/bin/env python
"""Run every benchmark arm across seeds and training-set sizes.

Writes one JSON per (arm, split, seed, n_train) into results/metrics/.
Every number in the paper traces back to these files via scripts/make_report.py.

Usage:
  python scripts/run_arms.py --arms B0 B1 B2 T1        # fast arms
  python scripts/run_arms.py --arms T2                 # ChemBERTa fine-tune
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from evapro.data.io import load_dataset
from evapro.data.splits import load_split
from evapro.evaluation.metrics import compute_all
from evapro.features.fingerprints import ecfp, rdkit_descriptors
from evapro.utils.seeding import set_seed

TARGET = "eva71_2a"
SPLIT_DIR = Path("data/processed/splits") / TARGET
OUT = Path("results/metrics")
PREDS = Path("results/predictions")
TRAIN_SIZES = [50, 100, 250, None]          # None = all available
SEEDS = list(range(10))
CHEMBERTA = "DeepChem/ChemBERTa-77M-MTR"

METRIC_NAMES = ["rmse", "mae", "r2", "spearman", "pearson", "precision_at_10pct"]


# --------------------------------------------------------------------------
# Representations (cached once per process -- they do not depend on the split)
# --------------------------------------------------------------------------
_cache: dict[str, np.ndarray] = {}


def get_features(kind: str, smiles: list[str]) -> np.ndarray:
    if kind in _cache:
        return _cache[kind]
    if kind == "ecfp":
        x = ecfp(smiles, radius=2, n_bits=2048, counts=True)
    elif kind == "descriptors":
        x = rdkit_descriptors(smiles)
    elif kind == "chemberta":
        x = chemberta_embeddings(smiles)
    elif kind == "random_encoder":
        x = untrained_encoder_embeddings(smiles)
    elif kind.startswith("indomain:"):
        x = indomain_embeddings(kind.split(":", 1)[1], smiles)
    else:
        raise ValueError(kind)
    _cache[kind] = x
    return x


def untrained_encoder_embeddings(smiles: list[str], seed: int = 0) -> np.ndarray:
    """Arm T0r: the SAME architecture, randomly initialised and NEVER trained.

    The control T4 needs. T4 is a randomly-initialised encoder that is then
    multitask-pretrained in-domain; if an untrained one scores the same, T4's
    embeddings are a random projection of SMILES tokens and the in-domain
    pretraining contributed nothing. Random features are a real baseline, so
    this has to be measured rather than assumed away.
    """
    import torch

    from evapro.models.multitask import MultitaskRegressor, embed_smiles

    torch.manual_seed(seed)
    model = MultitaskRegressor(["dummy"], pretrained=False)
    return embed_smiles(model, smiles)


def indomain_embeddings(tag: str, smiles: list[str]) -> np.ndarray:
    """Frozen embeddings from an in-domain multitask-pretrained encoder (T4/T5).

    Mean-pooled exactly as T1 is, and the same 384 dimensions, so the ridge
    probe downstream is given an identical shape of input. Any difference
    against T1 is therefore attributable to what the encoder was pretrained on.
    """
    import torch

    from evapro.models.multitask import MultitaskRegressor, embed_smiles

    meta = json.loads(Path(f"models/{tag}.json").read_text())
    model = MultitaskRegressor(meta["tasks"], pretrained=False)
    model.load_state_dict(torch.load(f"models/{tag}.pt", map_location="cpu"))
    return embed_smiles(model, smiles)


def chemberta_embeddings(smiles: list[str], batch_size: int = 64) -> np.ndarray:
    """Frozen ChemBERTa CLS embeddings (arm T1: linear probe on a frozen encoder)."""
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
            mask = enc["attention_mask"].unsqueeze(-1).float()
            outs.append(((h * mask).sum(1) / mask.sum(1)).numpy())   # mean-pool
    return np.vstack(outs).astype(np.float32)


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------
class MedianRegressor:
    def fit(self, X, y):
        self._v = float(np.median(y)); return self
    def predict(self, X):
        return np.full(len(X), self._v)


def fit_predict_sklearn(arm, Xtr, ytr, Xte, seed):
    if arm == "B0":
        model = MedianRegressor()
    elif arm == "B1":
        # Early stopping on a validation fraction, patience 20 -- as specified in
        # plan.md §5. Without it, 400 unconditional iterations over 2048 ECFP
        # features dominates runtime and overfits n=50 folds.
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, min_samples_leaf=5,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
            random_state=seed)
    elif arm == "B2":
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=1,
                                      n_jobs=-1, random_state=seed)
    elif arm in ("T1", "T4", "T5", "T4c", "T5c", "T0r"):
        # Frozen-encoder linear probe: standardise then ridge with internal CV.
        # Identical for every frozen arm, so the arms differ only in the encoder.
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        model = RidgeCV(alphas=np.logspace(-2, 4, 25))
    else:
        raise ValueError(arm)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def finetune_chemberta(smiles_tr, ytr, smiles_te, seed, epochs=40, lr=3e-5, bs=16):
    """Arm T2: full fine-tune of the pretrained encoder with a regression head."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(CHEMBERTA)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHEMBERTA, num_labels=1, problem_type="regression")

    def encode(s):
        e = tok(list(s), padding="max_length", truncation=True, max_length=128,
                return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    ids, mask = encode(smiles_tr)
    y = torch.tensor(np.asarray(ytr), dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(ids, mask, y), batch_size=bs, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for b_ids, b_mask, b_y in loader:
            opt.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_y)
            out.loss.backward()
            opt.step()

    model.eval()
    ids_te, mask_te = encode(smiles_te)
    with torch.no_grad():
        preds = model(input_ids=ids_te, attention_mask=mask_te).logits.squeeze(-1).numpy()
    return preds


ARM_FEATURES = {
    "B0": "ecfp", "B1": "ecfp", "B2": "descriptors", "T1": "chemberta",
    # In-domain arms: T4 pretrained from random init, T5 chained from ChemBERTa;
    # the "c" variants use the decontaminated corpus (plan.md §7.1).
    "T0r": "random_encoder",
    "T4": "indomain:indomain_T4", "T5": "indomain:indomain_T5",
    "T4c": "indomain:indomain_T4_clean", "T5c": "indomain:indomain_T5_clean",
}
ARM_LABELS = {
    "B0": "B0_median", "B1": "B1_ecfp_histgb", "B2": "B2_descriptors_rf",
    "T1": "T1_chemberta_linear_probe", "T2": "T2_chemberta_full_finetune",
    "T0r": "T0r_untrained_encoder_probe",
    "T4": "T4_indomain_probe", "T5": "T5_chained_probe",
    "T4c": "T4c_indomain_probe_decontaminated",
    "T5c": "T5c_chained_probe_decontaminated",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B0", "B1", "B2", "T1"])
    ap.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="Restrict to these training-set sizes. The subsample RNG is "
                         "still drawn for every size in order, so the folds are "
                         "identical to an unrestricted run.")
    ap.add_argument("--save-preds", action="store_true",
                    help="Also write per-compound predictions to results/predictions/. "
                         "A cell whose metrics already exist is re-run and the recomputed "
                         "metrics asserted equal to the stored ones.")
    args = ap.parse_args()

    df = load_dataset(TARGET)
    smiles = df["canonical_smiles"].tolist()
    y_all = df["pactivity"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        feat = ARM_FEATURES.get(arm)
        X_all = get_features(feat, smiles) if feat else None
        for split in args.splits:
            for seed in args.seeds:
                set_seed(seed)
                rng = np.random.default_rng(seed)
                folds = load_split(SPLIT_DIR / f"{split}__seed{seed}.json")
                fold_of = df["inchikey"].map(folds).to_numpy()
                tr_all = np.flatnonzero(fold_of == "train")
                te = np.flatnonzero(fold_of == "test")

                for size in TRAIN_SIZES:
                    tr = tr_all if size is None or size >= len(tr_all) else \
                        rng.choice(tr_all, size=size, replace=False)
                    if args.sizes is not None and len(tr) not in args.sizes:
                        continue        # rng already advanced above -- folds unchanged
                    tag = f"{ARM_LABELS[arm]}__{split}__seed{seed}__n{len(tr)}"
                    path = OUT / f"{tag}.json"
                    pred_path = PREDS / f"{tag}.npz"
                    done = path.exists()
                    if done and not (args.save_preds and not pred_path.exists()):
                        continue

                    t0 = time.time()
                    if arm == "T2":
                        preds = finetune_chemberta([smiles[i] for i in tr], y_all[tr],
                                                   [smiles[i] for i in te], seed)
                    else:
                        preds = fit_predict_sklearn(arm, X_all[tr], y_all[tr], X_all[te], seed)
                    scores = compute_all(y_all[te], preds, METRIC_NAMES)

                    if args.save_preds:
                        # Re-running a completed cell must reproduce its stored metrics.
                        # This is the determinism claim of the manuscript §8, asserted
                        # rather than asserted-about.
                        if done:
                            stored = json.loads(path.read_text())["metrics"]
                            for k, v in stored.items():
                                if abs(v - scores[k]) > 1e-9:
                                    raise SystemExit(
                                        f"{tag}: {k} changed on re-run: {v} -> {scores[k]}")
                        PREDS.mkdir(parents=True, exist_ok=True)
                        np.savez(pred_path, inchikey=df["inchikey"].to_numpy()[te],
                                 y_true=y_all[te], y_pred=np.asarray(preds, dtype=float))
                    if done:
                        continue

                    path.write_text(json.dumps({
                        "arm": ARM_LABELS[arm], "target": TARGET, "split": split,
                        "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
                        "metrics": scores, "seconds": round(time.time() - t0, 2),
                        "source_script": "scripts/run_arms.py",
                    }, indent=2, sort_keys=True))
                    print(f"  {tag}  rmse={scores['rmse']:.3f} r={scores['spearman']:.3f} "
                          f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
