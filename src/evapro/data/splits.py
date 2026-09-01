"""Split construction. Splits are materialised as JSON so every arm reads the
same folds -- no experiment can quietly reshuffle its own test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

Folds = dict[str, str]  # inchikey -> "train" | "val" | "test"


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)


def _assign_by_group(
    groups: dict[str, list[int]],
    n_total: int,
    test_frac: float,
    val_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pack whole groups into test, then val, then train.

    Group order is *shuffled per seed*. A deterministic order (e.g. strict
    smallest-first) would make every seed produce the identical split, which
    silently removes all split variance from the error bars -- the paired tests
    in plan.md §6 would then be comparing constants.
    """
    keys = sorted(groups)                      # sort first so the shuffle is reproducible
    order = list(rng.permutation(len(keys)))
    n_test, n_val = int(n_total * test_frac), int(n_total * val_frac)

    fold = np.empty(n_total, dtype=object)
    n_assigned = {"test": 0, "val": 0}
    quota = {"test": n_test, "val": n_val}

    for k in order:
        idxs = groups[keys[k]]
        for name in ("test", "val"):
            # Take the group only if it fits, so one large scaffold cannot
            # inflate a held-out fold far past its quota; otherwise fall to train.
            if n_assigned[name] + len(idxs) <= quota[name]:
                fold[idxs] = name
                n_assigned[name] += len(idxs)
                break
        else:
            fold[idxs] = "train"
    return fold


def scaffold_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.1, seed=0) -> Folds:
    """Bemis-Murcko scaffold split -- the primary endpoint (plan.md §3.3)."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(df["canonical_smiles"]):
        groups.setdefault(murcko_scaffold(smi), []).append(i)
    fold = _assign_by_group(groups, len(df), test_frac, val_frac, rng)
    return dict(zip(df["inchikey"], fold))


def butina_split(df, test_frac=0.2, val_frac=0.1, cutoff=0.4, seed=0) -> Folds:
    """Butina cluster split at Tanimoto 0.6 (distance cutoff 0.4). Stricter than
    scaffold: catches analog series that share activity but not a scaffold.
    """
    rng = np.random.default_rng(seed)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df["canonical_smiles"]]

    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    groups = {str(ci): list(c) for ci, c in enumerate(clusters)}
    fold = _assign_by_group(groups, len(df), test_frac, val_frac, rng)
    return dict(zip(df["inchikey"], fold))


def temporal_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.1) -> Folds:
    """Oldest -> train, newest -> test. Deployment realism."""
    order = df.sort_values("doc_year", kind="stable").index.to_numpy()
    n = len(order)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    fold = np.empty(n, dtype=object)
    fold[order[: n - n_test - n_val]] = "train"
    fold[order[n - n_test - n_val : n - n_test]] = "val"
    fold[order[n - n_test :]] = "test"
    return dict(zip(df["inchikey"], fold))


def random_split(df: pd.DataFrame, test_frac=0.2, val_frac=0.1, seed=0) -> Folds:
    """Optimism reference ONLY -- never the headline number (plan.md §3.3)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test, n_val = int(len(df) * test_frac), int(len(df) * val_frac)
    fold = np.empty(len(df), dtype=object)
    fold[idx[:n_test]], fold[idx[n_test : n_test + n_val]] = "test", "val"
    fold[idx[n_test + n_val :]] = "train"
    return dict(zip(df["inchikey"], fold))


STRATEGIES = {
    "scaffold": scaffold_split,
    "butina": butina_split,
    "temporal": temporal_split,
    "random": random_split,
}


def build_split(df: pd.DataFrame, strategy: str, **kwargs) -> Folds:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown split strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](df, **kwargs)


# --------------------------------------------------------------------------
# Integrity checks -- run by `make splits` and by tests/test_splits.py.
# A split that fails these is a bug that would silently inflate every result.
# --------------------------------------------------------------------------

def assert_no_leakage(df: pd.DataFrame, folds: Folds, strategy: str) -> None:
    fold_series = df["inchikey"].map(folds)
    if fold_series.isna().any():
        raise AssertionError("some molecules were not assigned a fold")

    counts = fold_series.value_counts()
    for name in ("train", "test"):
        if counts.get(name, 0) == 0:
            raise AssertionError(f"split '{strategy}' produced an empty {name} fold")

    # No compound in two folds.
    dupes = df.groupby("inchikey")["inchikey"].count()
    if (dupes > 1).any():
        raise AssertionError("duplicate inchikeys survived curation; splits would leak")

    # For grouped splits, no scaffold may straddle train and test.
    if strategy in {"scaffold", "butina"}:
        scaf = df["canonical_smiles"].map(murcko_scaffold)
        table = pd.crosstab(scaf, fold_series)
        if "train" in table and "test" in table:
            straddling = ((table["train"] > 0) & (table["test"] > 0)).sum()
            if strategy == "scaffold" and straddling:
                raise AssertionError(f"{straddling} scaffolds appear in both train and test")


def save_split(folds: Folds, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "folds": folds}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_split(path: str | Path) -> Folds:
    return json.loads(Path(path).read_text())["folds"]
