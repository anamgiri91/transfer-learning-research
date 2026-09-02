#!/usr/bin/env python
"""Curate the in-domain 3C / 3C-like protease corpus for arm T4 (plan.md §4).

Runs the same fidelity gate as the target dataset (`evapro.data.curate`, the
gate of plan.md §3.2) over the ChEMBL pull from `fetch_indomain.py`, then flags
every corpus compound that overlaps the EV-A71/CVA16 2A evaluation set.

That overlap flag is the point. Contamination could not be measured for
ChemBERTa because its 77M corpus is not distributed (§6.1 falls back to a
PubChem upper bound). Here the pretraining corpus is one we built, so overlap
is *known* rather than bounded, and a decontaminated variant can be pretrained
on. This is what makes H4 answerable for this arm.

Two deviations from §3.2 are recorded rather than hidden:
  - ChEMBL's activity endpoint does not return the assay confidence score, so
    the `confidence >= 8` gate cannot be applied. The assay-description screen
    in fetch_indomain.py is the substitute, and it is weaker.
  - Non-nM units (a handful of ug/mL and 10^n uM records) are dropped rather
    than converted, because ug/mL needs a molecular weight the record does not
    carry and silent conversion is how unit errors enter datasets.

Writes data/processed/indomain_3c.csv and .curation.json
Usage: python scripts/prepare_indomain.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from evapro.data.curate import curate
from evapro.data.io import load_dataset
from evapro.data.splits import murcko_scaffold
from evapro.data.indomain import TARGETS

RDLogger.DisableLog("rdApp.*")

RAW = Path("data/raw")
OUT = Path("data/processed/indomain_3c.csv")
FUNNEL = Path("data/processed/indomain_3c.curation.json")
NEAR_DUPLICATE_T = 0.7          # same cut as Table 0 and §6.3


def load_raw() -> tuple[pd.DataFrame, dict]:
    """Concatenate the per-target pulls into the schema curate() expects."""
    frames, per_target, dropped_units, dropped_no_smiles = [], {}, 0, 0
    for label, (tid, family) in sorted(TARGETS.items()):
        f = RAW / f"indomain_{label}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, low_memory=False)
        n_before = len(d)
        d = d[d["standard_units"] == "nM"]
        dropped_units += n_before - len(d)
        n_units = len(d)
        # A few ChEMBL activity records carry no structure (the compound is
        # withheld or unresolvable). Counted, not silently coerced.
        d = d[d["canonical_smiles"].notna()]
        dropped_no_smiles += n_units - len(d)
        frames.append(pd.DataFrame({
            "smiles": d["canonical_smiles"],
            "standard_type": d["standard_type"],
            "standard_relation": d["standard_relation"],
            "standard_value_nm": pd.to_numeric(d["standard_value"], errors="coerce"),
            "target_id": label,
            "virus_family": family,
            "readout": "enzymatic",
            "confidence_score": np.nan,      # not returned by the activity endpoint
            "doc_year": pd.to_numeric(d.get("document_year"), errors="coerce"),
            "source": f"chembl:{tid}",
        }))
        per_target[label] = n_before
    return pd.concat(frames, ignore_index=True), {
        "raw_rows_per_target": per_target,
        "dropped_non_nm_units": int(dropped_units),
        "dropped_no_structure": int(dropped_no_smiles),
    }


def flag_overlap(corpus: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Mark corpus compounds overlapping the 2A evaluation set, three ways."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def fps(smis):
        return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]

    eval_keys = set(target_df["inchikey"])
    eval_scaffolds = set(target_df["scaffold"])
    eval_fps = fps(target_df["canonical_smiles"])
    corpus_fps = fps(corpus["canonical_smiles"])

    corpus = corpus.copy()
    corpus["scaffold"] = [murcko_scaffold(s) for s in corpus["canonical_smiles"]]
    corpus["overlap_exact"] = corpus["inchikey"].isin(eval_keys)
    corpus["overlap_scaffold"] = corpus["scaffold"].isin(eval_scaffolds)
    corpus["max_tanimoto_to_eval"] = [
        max(DataStructs.BulkTanimotoSimilarity(f, eval_fps), default=0.0)
        for f in corpus_fps
    ]
    corpus["overlap_near_duplicate"] = corpus["max_tanimoto_to_eval"] >= NEAR_DUPLICATE_T
    corpus["contaminated"] = (corpus["overlap_exact"] | corpus["overlap_scaffold"]
                              | corpus["overlap_near_duplicate"])
    return corpus


def main() -> int:
    argparse.ArgumentParser().parse_args()
    raw, raw_stats = load_raw()
    censored: list = []
    cur, rep = curate(raw, censored_out=censored)

    target_df = load_dataset("eva71_2a")
    cur = flag_overlap(cur, target_df)

    keep = ["inchikey", "canonical_smiles", "scaffold", "pactivity", "target_id",
            "virus_family", "standard_type", "mw", "max_tanimoto_to_eval",
            "overlap_exact", "overlap_scaffold", "overlap_near_duplicate",
            "contaminated"]
    cur = cur[keep].sort_values(["target_id", "inchikey"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cur.to_csv(OUT, index=False)

    funnel = {
        **raw_stats,
        **rep.to_dict(),
        "n_unique_compounds": int(cur["inchikey"].nunique()),
        "n_measurements": int(len(cur)),
        "per_target_measurements": cur["target_id"].value_counts().to_dict(),
        "per_family_measurements": cur["virus_family"].value_counts().to_dict(),
        "n_overlap_exact": int(cur["overlap_exact"].sum()),
        "n_overlap_scaffold": int(cur["overlap_scaffold"].sum()),
        "n_overlap_near_duplicate": int(cur["overlap_near_duplicate"].sum()),
        "n_contaminated": int(cur["contaminated"].sum()),
        "n_clean": int((~cur["contaminated"]).sum()),
        "near_duplicate_threshold": NEAR_DUPLICATE_T,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_script": "scripts/prepare_indomain.py",
    }
    FUNNEL.write_text(json.dumps(funnel, indent=2, sort_keys=True, default=str))

    print(f"in  {rep.n_input} rows ({raw_stats['dropped_non_nm_units']} non-nM units and "
          f"{raw_stats['dropped_no_structure']} structureless dropped first)")
    for k, v in rep.dropped.items():
        print(f"  -{v:>5}  {k}")
    print(f"  held out censored: {rep.n_censored_held_out}")
    print(f"out {len(cur)} measurements on {cur['inchikey'].nunique()} compounds")
    print(cur.groupby(["virus_family", "target_id"]).size().to_string())
    print(f"\ncontaminated vs 2A eval set: {funnel['n_contaminated']} "
          f"(exact {funnel['n_overlap_exact']}, scaffold {funnel['n_overlap_scaffold']}, "
          f"near-dup {funnel['n_overlap_near_duplicate']}) -> {funnel['n_clean']} clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
