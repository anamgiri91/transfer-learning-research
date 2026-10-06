#!/usr/bin/env python
"""Curate the OpenBind EV-A71 2A crystallographic set into a compound-level table.

The raw master.csv is one row per *crystal complex* (649), not per compound
(499): the same ligand appears in several structures. Modelling on complexes
would leak duplicate compounds across folds, so we collapse to compounds here
and record within-compound label consistency. These rows do not identify
independent assay replicates. Legacy 'replicate_spread' column names are retained
for compatibility with the committed datasets, not as a biological assertion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from evapro.data.openbind import label_audit, make_master, read_release

RDLogger.DisableLog("rdApp.*")

SRC = Path("data/processed/master.csv")
OUT = Path("data/processed/eva71_2a.csv")
REPORT = Path("data/processed/eva71_2a.curation.json")
MAX_SPREAD_LOG = 1.0


def main() -> int:
    raw = read_release()
    df = make_master(raw)
    SRC.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC, index=False)
    OUT.with_suffix(".label_audit.json").write_text(
        json.dumps(label_audit(raw, df), indent=2) + "\n")
    n_complexes = len(df)

    # Structural-quality filter; this does not estimate affinity measurement error.
    quality = ~df["suspected_artefact"] & df["pb_valid_prepared"]
    n_dropped_quality = int((~quality).sum())
    df = df[quality]

    grp = df.groupby("compound_group")
    spread = grp["pKD"].transform(lambda s: s.max() - s.min())
    consistent = df[spread <= MAX_SPREAD_LOG]
    n_dropped_spread = df["compound_group"].nunique() - consistent["compound_group"].nunique()

    rows = []
    for cg, g in consistent.groupby("compound_group"):
        smiles = g["SMILES"].iloc[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({
            "compound_group": cg,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "inchikey": Chem.MolToInchiKey(mol),
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
            "pactivity": float(g["pKD"].median()),
            "n_complexes": int(len(g)),
            "replicate_spread": float(g["pKD"].max() - g["pKD"].min()),
            "covalent": bool(g["covalent"].any()),
            "fragment_screen": bool(g["fragment_screen"].any()),
            "mw": float(Descriptors.MolWt(mol)),
            "sucos_shape": float(g["sucos_shape"].median()),
        })
    out = pd.DataFrame(rows)

    # Guard: an inchikey collision would reintroduce the leakage we just removed.
    dupes = out["inchikey"].duplicated().sum()
    if dupes:
        out = out.groupby("inchikey", as_index=False).agg(
            {**{c: "first" for c in out.columns if c != "pactivity"}, "pactivity": "median"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    report = {
        "n_complexes_in": n_complexes,
        "n_dropped_quality": n_dropped_quality,
        "n_compound_groups_dropped_spread": int(n_dropped_spread),
        "n_inchikey_collisions_merged": int(dupes),
        "n_compounds_out": len(out),
        "max_replicate_spread": float(out["replicate_spread"].max()),
        "pactivity": {k: float(v) for k, v in out["pactivity"].describe().items()},
        "n_scaffolds": int(out["scaffold"].nunique()),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
