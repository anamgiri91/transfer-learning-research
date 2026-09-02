"""Activity-cliff stratification of a test fold (plan.md §7.2).

The protocol asks for "performance restricted to matched molecular pairs with
>1 log activity difference". Splitting a test fold into cliff / non-cliff would
confound two unrelated kinds of hardness, so each test compound is labelled by
its relationship to the *training* fold -- which is what the model actually had
to work with:

    cliff    >=1 training neighbour at Tanimoto >= T whose |dy| > delta
    smooth   >=1 training neighbour at Tanimoto >= T, none of them a cliff
    distant  no training neighbour at Tanimoto >= T at all

`distant` is the control that makes the comparison mean anything: without it,
"non-cliff" silently pools compounds the model could interpolate with
compounds whose neighbourhood it had never seen.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

CLIFF, SMOOTH, DISTANT = "cliff", "smooth", "distant"
STRATA = (CLIFF, SMOOTH, DISTANT)

# Reused, not newly chosen: the same cut already fixed for the near-neighbour
# split audit (Table 0). Picking a fresh threshold here would be a free
# researcher degree of freedom.
TANIMOTO_THRESHOLD = 0.7
DELTA = 1.0


def similarity_matrix(smiles_list) -> np.ndarray:
    """Pairwise ECFP4 Tanimoto over binary fingerprints.

    Binary, not count: Tanimoto on count vectors is a different and less
    standard quantity, and the split audit this threshold is borrowed from
    uses binary fingerprints.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
    return np.vstack([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])


def stratify(test_idx, train_idx, sim, y, threshold: float = TANIMOTO_THRESHOLD,
             delta: float = DELTA) -> np.ndarray:
    """Label each test compound cliff / smooth / distant against the train fold.

    `sim` is a full pairwise similarity matrix and `y` the full label vector,
    both indexed by the dataset's row order; `test_idx` and `train_idx` index
    into them.
    """
    test_idx, train_idx = np.asarray(test_idx), np.asarray(train_idx)
    y = np.asarray(y)
    out = np.empty(len(test_idx), dtype=object)
    for j, i in enumerate(test_idx):
        near = train_idx[sim[i, train_idx] >= threshold]
        if len(near) == 0:
            out[j] = DISTANT
        elif np.any(np.abs(y[near] - y[i]) > delta):
            out[j] = CLIFF
        else:
            out[j] = SMOOTH
    return out


def cliff_census(sim: np.ndarray, y, threshold: float = TANIMOTO_THRESHOLD,
                 delta: float = DELTA) -> dict:
    """Dataset-level count of cliff pairs, independent of any split."""
    y = np.asarray(y)
    iu = np.triu_indices(len(y), 1)
    d = np.abs(y[:, None] - y[None, :])
    close = sim[iu] >= threshold
    cliff = close & (d[iu] > delta)
    involved = (sim >= threshold) & (d > delta)
    np.fill_diagonal(involved, False)
    return {
        "tanimoto_threshold": threshold,
        "delta_p": delta,
        "similar_pairs": int(close.sum()),
        "cliff_pairs": int(cliff.sum()),
        "cliff_fraction_of_similar": round(float(cliff.sum() / max(close.sum(), 1)), 4),
        "compounds_in_a_cliff": int(involved.any(1).sum()),
        "n_compounds": int(len(y)),
    }
