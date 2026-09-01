"""Pretraining-corpus decontamination (plan.md §7.1).

The load-bearing ablation: measure test-set overlap with the pretraining corpus
at three levels, then re-pretrain without it. The gap between contaminated and
decontaminated performance is a reported result, not a footnote.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from evapro.data.splits import murcko_scaffold

TANIMOTO_THRESHOLD = 0.7


@dataclass
class ContaminationReport:
    n_test: int
    n_pretrain: int
    exact_overlap: int
    scaffold_overlap: int
    near_duplicate_overlap: int

    @property
    def worst_case_frac(self) -> float:
        return self.near_duplicate_overlap / self.n_test if self.n_test else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "worst_case_frac": self.worst_case_frac}


def _fps(smiles_list):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def measure(test_smiles, test_inchikeys, pretrain_smiles, pretrain_inchikeys,
            threshold: float = TANIMOTO_THRESHOLD) -> ContaminationReport:
    """Overlap at exact / scaffold / near-duplicate level."""
    pre_keys = set(pretrain_inchikeys)
    exact = sum(k in pre_keys for k in test_inchikeys)

    pre_scaffolds = {murcko_scaffold(s) for s in pretrain_smiles}
    scaffold = sum(murcko_scaffold(s) in pre_scaffolds for s in test_smiles)

    pre_fps = _fps(pretrain_smiles)
    near = 0
    for fp in _fps(test_smiles):
        sims = DataStructs.BulkTanimotoSimilarity(fp, pre_fps)
        if sims and max(sims) >= threshold:
            near += 1

    return ContaminationReport(len(test_smiles), len(pretrain_smiles), exact, scaffold, near)


def decontaminate(pretrain_df, test_smiles, test_inchikeys, threshold=TANIMOTO_THRESHOLD):
    """Return the pretraining corpus with every test-overlapping record removed.

    Removal is at the *near-duplicate* level -- the strictest of the three -- so
    the decontaminated re-run is a genuine lower bound on transfer benefit.
    """
    keys = set(test_inchikeys)
    keep = ~pretrain_df["inchikey"].isin(keys)
    pretrain_df = pretrain_df[keep]

    test_fps = _fps(test_smiles)
    survivors = []
    for fp in _fps(pretrain_df["canonical_smiles"]):
        sims = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
        survivors.append(not sims or max(sims) < threshold)
    return pretrain_df[np.asarray(survivors)]
