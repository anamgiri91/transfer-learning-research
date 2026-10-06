#!/usr/bin/env python
"""Design diagnostics the per-arm tables cannot show (audit of 2026-10-05).

Four properties of this study's evaluation design are invisible in every
results table, and each of them bounds how a reported interval should be read:

  1  the ten seeds are OVERLAPPING resamples of 494 compounds, not independent
     replicates, while every paired test treats them as independent
  2  one seed's test fold is composed differently from the other nine, so the
     seeds are not exchangeable by construction; leave-one-seed-out shows
     whether that matters for the headline contrasts
  3  a 49-compound validation fold is reserved by every split file and read by
     no executed arm, so the evaluated dataset is smaller than the curated one
  4  at the same labelled budget, arms do not all fit the same number of
     compounds, because some hold out an internal validation split and some
     do not

None of these is a new experiment: everything here is read from the committed
split files, metric records and predictions.

Writes docs/design-diagnostics.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

SPLIT_DIR = Path("data/processed/splits/eva71_2a")
OUT = Path("docs/design-diagnostics.json")
SPLITS = ["scaffold", "random", "butina"]
SEEDS = list(range(10))
N = 347

# Headline paired contrasts, as §5.4 and §6.6 report them.
CONTRASTS = [
    ("T1", "results/metrics/T1_chemberta_linear_probe", "results/metrics/B1_ecfp_histgb"),
    ("T2", "results/metrics/T2_chemberta_full_finetune", "results/metrics/B1_ecfp_histgb"),
    ("T2v", "results/metrics_ft/T2v", "results/metrics/B1_ecfp_histgb"),
    ("B3", "results/metrics_b3/B3", "results/metrics/B1_ecfp_histgb"),
]


def folds(split: str, seed: int) -> dict:
    return json.loads((SPLIT_DIR / f"{split}__seed{seed}.json").read_text())["folds"]


def seed_dependence() -> dict:
    out = {}
    for split in SPLITS:
        tests = [{k for k, v in folds(split, s).items() if v == "test"} for s in SEEDS]
        overlaps = [len(a & b) / len(a) for a, b in itertools.combinations(tests, 2)]
        appear = collections.Counter(k for t in tests for k in t)
        n_all = len(folds(split, 0))
        out[split] = dict(
            n_curated=n_all,
            mean_pairwise_test_overlap=round(float(np.mean(overlaps)), 4),
            min_pairwise_test_overlap=round(float(np.min(overlaps)), 4),
            max_pairwise_test_overlap=round(float(np.max(overlaps)), 4),
            compounds_ever_in_a_test_fold=len(appear),
            compounds_never_in_any_test_fold=n_all - len(appear),
            max_test_folds_one_compound_appears_in=max(appear.values()),
        )
    return dict(
        per_split=out,
        consequence="Per-seed differences entering every paired test are "
                    "positively correlated through shared compounds. Nominal "
                    "p-values and seed bootstrap intervals are therefore "
                    "anticonservative; no test in this study adjusts for it.",
    )


def fold_composition() -> dict:
    df = pd.read_csv("data/processed/eva71_2a.csv")
    scaffold = {k: MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
                for k, s in zip(df.inchikey, df.canonical_smiles)}
    rows = []
    for seed in SEEDS:
        te = [k for k, v in folds("scaffold", seed).items() if v == "test"]
        counts = collections.Counter(scaffold[k] for k in te)
        largest = counts.most_common(1)[0][1]
        rows.append(dict(seed=seed, n_test=len(te), n_test_scaffolds=len(counts),
                         largest_series_in_test=largest,
                         largest_series_fraction=round(largest / len(te), 4)))
    n_scaf = [r["n_test_scaffolds"] for r in rows]
    odd = [r["seed"] for r in rows
           if r["n_test_scaffolds"] < min(n_scaf) + 0.5 * (max(n_scaf) - min(n_scaf))]
    return dict(
        split="scaffold", per_seed=rows,
        n_test_scaffolds_range=[min(n_scaf), max(n_scaf)],
        least_diverse_seeds=odd,
        consequence="The seeds are not exchangeable: a fold built mostly from "
                    "one congeneric series measures a different quantity from "
                    "a fold spanning sixty scaffolds. `n_test_scaffolds` is "
                    "recorded in table0 but no reported summary conditions on it.",
    )


def rmse(stem: str, seed: int, n: int = N, split: str = "scaffold") -> float | None:
    p = Path(f"{stem}__{split}__seed{seed}__n{n}.json")
    return json.loads(p.read_text())["metrics"]["rmse"] if p.exists() else None


def leave_one_seed_out() -> list[dict]:
    out = []
    for name, arm_stem, ref_stem in CONTRASTS:
        a = np.array([rmse(arm_stem, s) for s in SEEDS], dtype=float)
        b = np.array([rmse(ref_stem, s) for s in SEEDS], dtype=float)
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        full_d = float(np.median(a - b))
        full_p = float(stats.wilcoxon(a, b).pvalue)
        drops = []
        for i, s in enumerate(SEEDS):
            m = np.ones(len(SEEDS), dtype=bool)
            m[i] = False
            d = float(np.median(a[m] - b[m]))
            p = float(stats.wilcoxon(a[m], b[m]).pvalue)
            drops.append(dict(dropped_seed=s, median_paired_delta=round(d, 4),
                              p_raw=round(p, 4)))
        deltas = [x["median_paired_delta"] for x in drops]
        ps = [x["p_raw"] for x in drops]
        out.append(dict(
            contrast=f"{name} vs B1", n_train=N, metric="rmse",
            full_median_paired_delta=round(full_d, 4), full_p_raw=round(full_p, 4),
            leave_one_out=drops,
            median_paired_delta_range=[min(deltas), max(deltas)],
            p_raw_range=[min(ps), max(ps)],
            sign_stable=bool(np.sign(min(deltas)) == np.sign(max(deltas))),
            crosses_0_05=bool(min(ps) <= 0.05 < max(ps))))
    return out


def data_actually_used() -> dict:
    counts = collections.Counter(folds("scaffold", 0).values())
    scripts_reading_val = sorted(
        f for f in glob.glob("scripts/*.py")
        if '"val"' in Path(f).read_text() or "'val'" in Path(f).read_text())
    return dict(
        curated_compounds=sum(counts.values()),
        train=counts["train"], validation_fold=counts["val"], test=counts["test"],
        compounds_reaching_any_executed_arm=counts["train"] + counts["test"],
        validation_fold_read_by=scripts_reading_val,
        consequence="Every split file reserves a validation fold that the "
                    "executed runners never read: they select folds 'train' "
                    "and 'test' only. The largest labelled budget, n = 347, is "
                    "therefore 70% of the curated set, and the reserved "
                    "compounds enter no arm's training or evaluation.",
    )


def fitted_sizes() -> dict:
    rows = []
    for label, stem, held in (
            ("B1_ecfp_histgb", "results/metrics/B1_ecfp_histgb", "internal, not exposed"),
            ("B2_descriptors_rf", "results/metrics/B2_descriptors_rf", "none"),
            ("T1_chemberta_linear_probe", "results/metrics/T1_chemberta_linear_probe", "none"),
            ("T2v", "results/metrics_ft/T2v", "exposed"),
            ("B3", "results/metrics_b3/B3", "exposed")):
        p = Path(f"{stem}__scaffold__seed0__n{N}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append(dict(arm=label, labelled_budget=d["n_train"],
                         n_train_fitted=d.get("n_train_fitted"),
                         n_internal_val=d.get("n_internal_val"),
                         internal_validation=held))
    return dict(
        at_labelled_budget=N, per_arm=rows,
        consequence="At the same labelled budget the arms do not fit the same "
                    "number of compounds. B1 holds out 15% internally without "
                    "exposing the split; the amended fine-tunes and B3 hold out "
                    "52 and report it; B2 and the frozen probes hold out none. "
                    "The headline frozen-probe contrast therefore gives the "
                    "transfer arm more training data than its baseline, which "
                    "is conservative for a negative result but not matched.",
    )


def build_report() -> dict:
    return {
        "audit_date": "2026-10-05",
        "note": "Read from committed split files, metric records and "
                "predictions. No model was fitted and no published artefact "
                "was modified.",
        "seed_dependence": seed_dependence(),
        "test_fold_composition": fold_composition(),
        "leave_one_seed_out": leave_one_seed_out(),
        "data_actually_used": data_actually_used(),
        "fitted_training_sizes": fitted_sizes(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    encoded = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != encoded:
            print(f"{OUT} is missing or stale")
            return 1
        print(f"{OUT} is current")
        return 0
    OUT.write_text(encoded)
    r = json.loads(encoded)
    print(f"wrote {OUT}")
    sd = r["seed_dependence"]["per_split"]["scaffold"]
    print(f"  scaffold: mean pairwise test overlap "
          f"{sd['mean_pairwise_test_overlap']:.3f}, "
          f"{sd['compounds_never_in_any_test_fold']} compounds never tested, "
          f"one compound in up to {sd['max_test_folds_one_compound_appears_in']} folds")
    fc = r["test_fold_composition"]
    print(f"  test scaffolds per seed {fc['n_test_scaffolds_range']}, "
          f"least diverse seed(s) {fc['least_diverse_seeds']}")
    for c in r["leave_one_seed_out"]:
        print(f"  {c['contrast']:12s} delta {c['full_median_paired_delta']:+.4f} "
              f"range {c['median_paired_delta_range']}  p range {c['p_raw_range']}  "
              f"{'sign stable' if c['sign_stable'] else 'SIGN FLIPS'}"
              f"{', CROSSES 0.05' if c['crosses_0_05'] else ''}")
    du = r["data_actually_used"]
    print(f"  curated {du['curated_compounds']}, reaching an arm "
          f"{du['compounds_reaching_any_executed_arm']}, "
          f"reserved and unread {du['validation_fold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
