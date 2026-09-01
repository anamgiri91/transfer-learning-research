"""Molecular representations for the baseline arms."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


def ecfp(smiles_list, radius: int = 2, n_bits: int = 2048, counts: bool = True) -> np.ndarray:
    """ECFP4. Count vectors by default -- the strong-baseline setting (B1)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    get = gen.GetCountFingerprintAsNumPy if counts else gen.GetFingerprintAsNumPy
    return np.vstack([get(Chem.MolFromSmiles(s)) for s in smiles_list]).astype(np.float32)


_DESC = [(n, f) for n, f in Descriptors.descList]


def rdkit_descriptors(smiles_list) -> np.ndarray:
    """Full RDKit descriptor block (B2). NaNs -> 0; callers should scale."""
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        rows.append([f(mol) for _, f in _DESC])
    x = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


REPRESENTATIONS = {"ecfp": ecfp, "rdkit_descriptors": rdkit_descriptors}
