"""Curation: raw assay records -> high-fidelity, model-ready table.

Implements the gate in plan.md §3.2. Every rejection is counted and returned so
`make data` can print an auditable funnel rather than a silent row count.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

VALID_ACTIVITY_TYPES = {"IC50", "Ki", "Kd", "EC50"}
PACTIVITY_RANGE = (3.0, 11.0)
MW_RANGE = (150.0, 900.0)
MAX_REPLICATE_SPREAD_LOG = 1.0
MIN_CONFIDENCE = 8


@dataclass
class CurationReport:
    """Auditable funnel: how many rows died at each gate, and why."""

    n_input: int = 0
    dropped: Counter = field(default_factory=Counter)
    n_output: int = 0
    n_censored_held_out: int = 0

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_censored_held_out": self.n_censored_held_out,
            "dropped": dict(self.dropped),
        }


def standardize_mol(smiles: str) -> Chem.Mol | None:
    """Desalt, neutralise, reject mixtures. Returns None if unusable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if "." in Chem.MolToSmiles(mol):
        return None
    return rdMolStandardize.Uncharger().uncharge(mol)


def to_pactivity(value_nm: float) -> float:
    """nM -> pActivity (9 - log10 nM)."""
    return 9.0 - np.log10(value_nm)


def curate(
    df: pd.DataFrame,
    *,
    censored_out: list | None = None,
) -> tuple[pd.DataFrame, CurationReport]:
    """Apply the fidelity gate.

    Expects columns: smiles, standard_type, standard_relation, standard_value_nm,
    target_id, readout, confidence_score, doc_year, source.
    Censored records ('>' / '<') are *removed from the main table and appended to
    `censored_out`* for the §7.2 sensitivity analysis -- never silently dropped
    and never silently pooled in.
    """
    rep = CurationReport(n_input=len(df))
    df = df.copy()

    mask = df["standard_type"].isin(VALID_ACTIVITY_TYPES)
    rep.dropped["activity_type"] += int((~mask).sum())
    df = df[mask]

    censored = df[df["standard_relation"] != "="]
    rep.n_censored_held_out = len(censored)
    if censored_out is not None:
        censored_out.append(censored)
    df = df[df["standard_relation"] == "="]

    if "confidence_score" in df:
        mask = df["confidence_score"].isna() | (df["confidence_score"] >= MIN_CONFIDENCE)
        rep.dropped["low_confidence"] += int((~mask).sum())
        df = df[mask]

    mask = df["standard_value_nm"].notna() & (df["standard_value_nm"] > 0)
    rep.dropped["missing_value"] += int((~mask).sum())
    df = df[mask]

    df["pactivity"] = to_pactivity(df["standard_value_nm"].astype(float))
    mask = df["pactivity"].between(*PACTIVITY_RANGE)
    rep.dropped["pactivity_out_of_range"] += int((~mask).sum())
    df = df[mask]

    mols = df["smiles"].map(standardize_mol)
    mask = mols.notna()
    rep.dropped["unparseable_structure"] += int((~mask).sum())
    df, mols = df[mask], mols[mask]

    df["canonical_smiles"] = [Chem.MolToSmiles(m) for m in mols]
    df["inchikey"] = [Chem.MolToInchiKey(m) for m in mols]
    df["mw"] = [Descriptors.MolWt(m) for m in mols]

    mask = df["mw"].between(*MW_RANGE)
    rep.dropped["mw_out_of_range"] += int((~mask).sum())
    df = df[mask]

    df, n_disagree = _collapse_replicates(df)
    rep.dropped["replicate_disagreement"] += n_disagree

    rep.n_output = len(df)
    return df.reset_index(drop=True), rep


def _collapse_replicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Median-collapse duplicates; discard groups whose spread exceeds 1 log unit.

    This is the 'high-fidelity' criterion from the title, so it is a hard filter.
    """
    keys = ["inchikey", "target_id", "standard_type", "readout"]
    grouped = df.groupby(keys, dropna=False)["pactivity"]
    spread = grouped.transform(lambda s: s.max() - s.min())

    consistent = df[spread <= MAX_REPLICATE_SPREAD_LOG]
    n_disagree = int(df[keys].drop_duplicates().shape[0] - consistent[keys].drop_duplicates().shape[0])

    agg = {c: "first" for c in consistent.columns if c not in keys + ["pactivity"]}
    agg["pactivity"] = "median"
    collapsed = consistent.groupby(keys, dropna=False, as_index=False).agg(agg)
    return collapsed, n_disagree
