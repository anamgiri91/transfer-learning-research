#!/usr/bin/env python
"""Multitask pretraining on the in-domain 3C/3CL corpus (plan.md §4, arms T4/T5).

Usage:
  python scripts/pretrain_indomain.py --arm T4              # random init
  python scripts/pretrain_indomain.py --arm T5              # from ChemBERTa
  python scripts/pretrain_indomain.py --arm T5 --decontaminate

`--decontaminate` drops every corpus record overlapping the EV-A71/CVA16 2A
evaluation set (exact InChIKey, Bemis-Murcko scaffold, or ECFP4 Tanimoto >= 0.7,
flagged by prepare_indomain.py). Running with and without it is the §7.1
ablation that could not be performed for ChemBERTa, whose corpus is not
distributed -- here the corpus is ours, so the comparison is exact.

The corpus train/val split is grouped by compound, so no molecule appears in
both; validation RMSE only selects the pretraining checkpoint and never sees
the downstream evaluation set.

Writes models/indomain_<arm>[_clean].pt and a sidecar JSON of the run.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from evapro.models.multitask import CHEMBERTA, MultitaskRegressor, tokenize
from evapro.utils.seeding import set_seed

CORPUS = Path("data/processed/indomain_3c.csv")
MODELS = Path("models")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["T4", "T5"], required=True)
    ap.add_argument("--decontaminate", action="store_true")
    ap.add_argument("--drop-random", type=int, default=0,
                    help="Drop this many RANDOM measurements instead of the "
                         "contaminated ones. The control for --decontaminate: "
                         "removing overlap also removes training data, so a "
                         "size-matched random ablation separates the two.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 3e-4 for random init (T4), 3e-5 from ChemBERTa (T5)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    # plan.md Amendment 9. One --seed used to control the weight initialisation,
    # the corpus validation split and the random-ablation draw at once, so an
    # ablation condition could not be re-run while holding the other two fixed.
    # Each defaults to --seed, so every historical command is unchanged.
    ap.add_argument("--init-seed", type=int, default=None,
                    help="seed the encoder initialisation and batch order only")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed the corpus train/validation split only")
    ap.add_argument("--draw-seed", type=int, default=None,
                    help="seed the --drop-random draw only")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the output name, for replicate encoders")
    args = ap.parse_args()
    init_seed = args.seed if args.init_seed is None else args.init_seed
    split_seed = args.seed if args.split_seed is None else args.split_seed
    draw_seed = args.seed if args.draw_seed is None else args.draw_seed

    pretrained = args.arm == "T5"
    lr = args.lr if args.lr is not None else (3e-5 if pretrained else 3e-4)
    set_seed(init_seed)

    df = pd.read_csv(CORPUS)
    n_all = len(df)
    if args.decontaminate:
        df = df[~df["contaminated"]].reset_index(drop=True)
    elif args.drop_random:
        rng0 = np.random.default_rng(1000 + draw_seed)
        keep = rng0.permutation(len(df))[args.drop_random:]
        df = df.iloc[sorted(keep)].reset_index(drop=True)
    tasks = sorted(df["target_id"].unique())
    task_idx = {t: i for i, t in enumerate(tasks)}

    # Split by compound: a molecule measured against two targets must not
    # straddle the corpus folds.
    rng = np.random.default_rng(split_seed)
    keys = df["inchikey"].unique()
    val_keys = set(rng.choice(keys, size=max(1, int(len(keys) * args.val_frac)),
                              replace=False).tolist())
    is_val = df["inchikey"].isin(val_keys).to_numpy()

    ids, mask = tokenize(df["canonical_smiles"], CHEMBERTA)
    y = torch.tensor(df["pactivity"].to_numpy(), dtype=torch.float32)
    t = torch.tensor([task_idx[v] for v in df["target_id"]], dtype=torch.long)

    tr, va = ~is_val, is_val
    train_ds = TensorDataset(ids[tr], mask[tr], y[tr], t[tr])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MultitaskRegressor(tasks, pretrained=pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for b_ids, b_mask, b_y, b_t in loader:
            opt.zero_grad()
            lossf(model(b_ids, b_mask, b_t), b_y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(ids[va], mask[va], t[va])
            rmse = float(torch.sqrt(torch.mean((pv - y[va]) ** 2)))
        flag = ""
        if rmse < best - 1e-4:
            best, bad = rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            flag = "  *"
        else:
            bad += 1
        print(f"  epoch {epoch:>3}  val RMSE {rmse:.4f}{flag}", flush=True)
        if bad >= args.patience:
            print(f"  early stop (patience {args.patience})")
            break

    model.load_state_dict(best_state)
    MODELS.mkdir(exist_ok=True)
    tag = (f"indomain_{args.arm}" + ("_clean" if args.decontaminate else "")
           + (f"_rand{args.drop_random}" if args.drop_random else "")
           + args.tag_suffix)
    torch.save(model.state_dict(), MODELS / f"{tag}.pt")
    (MODELS / f"{tag}.json").write_text(json.dumps({
        "arm": args.arm, "decontaminated": args.decontaminate,
        "init_seed": int(init_seed), "split_seed": int(split_seed),
        "draw_seed": int(draw_seed), "tag_suffix": args.tag_suffix,
        "n_dropped_random": int(args.drop_random),
        "initialisation": "ChemBERTa-77M-MTR" if pretrained else "random",
        "tasks": tasks, "lr": lr, "batch_size": args.batch_size,
        "epochs_run": epoch + 1, "best_val_rmse": round(best, 4),
        "n_corpus_all": int(n_all), "n_corpus_used": int(len(df)),
        "n_dropped_contaminated": int(n_all - len(df)) if args.decontaminate else 0,
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "seconds": round(time.time() - t0, 1),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_script": "scripts/pretrain_indomain.py",
    }, indent=2, sort_keys=True))
    print(f"{tag}: best val RMSE {best:.4f} after {epoch + 1} epochs "
          f"({time.time() - t0:.0f}s) -> models/{tag}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
