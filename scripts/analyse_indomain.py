#!/usr/bin/env python
"""In-domain transfer contrasts for H3, plus the untrained-encoder control (§6.4).

`plan.md` decides H3 -- in-domain transfer beats generic self-supervised
pretraining -- by the direct contrast T4 vs T1. Every other arm in this paper
is compared against B1, so those tests do not answer it; this script runs the
contrasts H3 actually needs.

It also runs the control that makes the in-domain result interpretable. T4 is a
randomly-initialised encoder that is then multitask-pretrained in-domain. If a
*never-trained* encoder of the same architecture scores the same, then T4's
embeddings are a random projection of SMILES tokens and the pretraining
contributed nothing. Random features are a genuine baseline, so arm T0r
measures it rather than assuming it away -- and the same control turns out to
be the sharpest available test of what generic pretraining is worth here.

Writes results/tables/table10_indomain_contrasts.csv
       results/tables/table11_random_encoder_draws.csv
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

METRICS = Path("results/metrics")
OUT_C = Path("results/tables/table10_indomain_contrasts.csv")
OUT_D = Path("results/tables/table11_random_encoder_draws.csv")

ARM = {
    "B1": "B1_ecfp_histgb",
    "T4c": "T4c_indomain_probe_decontaminated",
    "T5c": "T5c_chained_probe_decontaminated",
    "T4r": "T4r_indomain_probe_random_ablation",
    "T5r": "T5r_chained_probe_random_ablation",
    "T0r": "T0r_untrained_encoder_probe",
    "T1": "T1_chemberta_linear_probe",
    "T2": "T2_chemberta_full_finetune",
    "T4": "T4_indomain_probe",
    "T5": "T5_chained_probe",
}

# (arm, reference, what the contrast decides)
CONTRASTS = [
    ("T4", "T1", "H3: in-domain vs generic pretraining (the pre-registered test)"),
    ("T5", "T1", "H3: in-domain adaptation on top of generic pretraining"),
    ("T1", "T0r", "control: is generic pretraining better than no pretraining?"),
    ("T2", "T0r", "control: is the generic fine-tune better than no pretraining?"),
    ("T4", "T0r", "control: is in-domain pretraining better than no pretraining?"),
    ("T5", "T0r", "control: is chained pretraining better than no pretraining?"),
    ("T4", "T5", "does starting from ChemBERTa matter once adapted in-domain?"),
    ("T4", "B1", "in-domain vs the fingerprint baseline"),
    ("T5", "B1", "chained vs the fingerprint baseline"),
    # H4 (plan.md §7.1). Decontamination removes the overlapping records AND
    # 2% of the corpus, so on its own it cannot separate leakage from data
    # volume. The size-matched random ablation is what makes it readable, and
    # the third row is the comparison that actually decides H4.
    ("T4c", "T4", "H4: effect of removing the 61 overlapping records"),
    ("T5c", "T5", "H4: effect of removing the 61 overlapping records"),
    ("T4r", "T4", "control: effect of removing 61 RANDOM records"),
    ("T5r", "T5", "control: effect of removing 61 RANDOM records"),
    ("T4c", "T4r", "H4 decided: decontaminated vs size-matched random ablation"),
    ("T5c", "T5r", "H4 decided: decontaminated vs size-matched random ablation"),
]


def load(arm: str, n: int, split: str) -> dict[int, dict]:
    out = {}
    for f in glob.glob(str(METRICS / f"{arm}__{split}__seed*__n{n}.json")):
        d = json.load(open(f))
        out[d["seed"]] = d["metrics"]
    return out


def contrast(a: str, b: str, n: int, split: str, metric: str,
             lower_is_better: bool) -> dict | None:
    A, B = load(ARM[a], n, split), load(ARM[b], n, split)
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 3:
        return None
    x = np.array([A[s][metric] for s in seeds])
    y = np.array([B[s][metric] for s in seeds])
    delta = float(np.median(y - x)) if lower_is_better else float(np.median(x - y))
    wins = int((x < y).sum()) if lower_is_better else int((x > y).sum())
    p = float(stats.wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
    return {"arm": a, "reference": b, "n_train": n, "split": split, "metric": metric,
            "n_seeds": len(seeds), "median_delta": round(delta, 4),
            "arm_better_in_seeds": wins, "p_raw": round(p, 4),
            "verdict": ("arm better" if p <= 0.05 and delta > 0 else
                        "reference better" if p <= 0.05 else "inconclusive")}


def random_encoder_draws(n_draws: int, split: str, n_train: int) -> pd.DataFrame:
    """How much does T0r depend on WHICH random encoder you drew?

    The control is only meaningful if a single random initialisation is
    representative. Re-draws the encoder and re-runs the whole probe for each.
    """
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from evapro.data.io import load_dataset
    from evapro.data.splits import load_split
    from evapro.models.multitask import MultitaskRegressor, embed_smiles

    df = load_dataset("eva71_2a")
    y = df["pactivity"].to_numpy()
    keys = df["inchikey"]
    sd = Path("data/processed/splits/eva71_2a")
    rows = []
    for draw in range(n_draws):
        torch.manual_seed(draw)
        X = embed_smiles(MultitaskRegressor(["d"], pretrained=False), df["canonical_smiles"].tolist())
        rmses = []
        for seed in range(10):
            folds = load_split(sd / f"{split}__seed{seed}.json")
            fo = keys.map(folds).to_numpy()
            tr, te = np.flatnonzero(fo == "train"), np.flatnonzero(fo == "test")
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))
            rmses.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
        rows.append({"encoder_draw": draw, "n_eval_seeds": len(rmses),
                     "rmse_median": round(float(np.median(rmses)), 4)})
        print(f"  encoder draw {draw}: median RMSE {rows[-1]['rmse_median']}")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 347])
    ap.add_argument("--draws", type=int, default=5,
                    help="random-encoder draws for the T0r sensitivity table (0 to skip)")
    args = ap.parse_args()

    rows = []
    for a, b, question in CONTRASTS:
        for n in args.sizes:
            for metric, lower in (("rmse", True), ("spearman", False)):
                r = contrast(a, b, n, args.split, metric, lower)
                if r:
                    rows.append({**r, "question": question})
    df = pd.DataFrame(rows)
    stamp = (f"# generated by scripts/analyse_indomain.py on "
             f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
             f"# paired Wilcoxon across seeds; positive delta favours `arm`\n"
             f"# p is uncorrected: these are pre-specified pairwise contrasts, not a family sweep\n")
    OUT_C.write_text(stamp + df.to_csv(index=False))
    print(f"wrote {OUT_C}")
    if args.draws:
        print("\nT0r sensitivity to the random encoder draw:")
        dd = random_encoder_draws(args.draws, args.split, max(args.sizes))
        OUT_D.write_text(stamp + dd.to_csv(index=False))
        print(f"wrote {OUT_D}  (median {dd.rmse_median.median():.4f}, "
              f"range {dd.rmse_median.min():.4f}-{dd.rmse_median.max():.4f})")

    show = df[(df.metric == "rmse")][
        ["arm", "reference", "n_train", "median_delta", "arm_better_in_seeds",
         "n_seeds", "p_raw", "verdict"]]
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
