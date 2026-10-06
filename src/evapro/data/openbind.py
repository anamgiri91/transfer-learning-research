"""Versioned OpenBind input and row-level label provenance.

Repeated crystal complexes are not evidence of independent affinity assays.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

RECORD = "20026661"
ARCHIVE = Path("data/raw/OpenBind_EV-A71_2A.zip")
ARCHIVE_SHA256 = "551d247e1e2d6cd541b97ce00e00eeae596ef6b37647c7d2b63b2a9c41a24056"
METADATA_MEMBER = "OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv"
DOWNLOAD_URL = f"https://zenodo.org/records/{RECORD}/files/OpenBind_EV-A71_2A.zip?download=1"


def verify_archive(path: Path) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"OpenBind archive checksum mismatch: {actual}")


def read_release(path: Path = ARCHIVE) -> pd.DataFrame:
    verify_archive(path)
    with ZipFile(path) as archive, archive.open(METADATA_MEMBER) as handle:
        return pd.read_csv(handle)


def make_master(raw: pd.DataFrame) -> pd.DataFrame:
    """The original 925 -> 649 row transformation, without changing labels."""
    return raw.rename(columns={"complex_name": "Complex", "smiles": "SMILES",
                               "experimental_pKD": "pKD"}).dropna(subset=["pKD"]).copy()


def label_audit(raw: pd.DataFrame, master: pd.DataFrame) -> dict:
    eligible = master[~master["suspected_artefact"] & master["pb_valid_prepared"]]
    groups = eligible.groupby("compound_group")["pKD"].agg(["size", "nunique", "min", "max"])
    multiple = groups["size"] > 1
    return {
        "source_record": RECORD,
        "source_sha256": ARCHIVE_SHA256,
        "raw_columns": list(raw.columns),
        "raw_complexes": len(raw),
        "labelled_complexes": len(master),
        "quality_filtered_complexes": len(eligible),
        "compound_groups": len(groups),
        "single_complex_groups": int((~multiple).sum()),
        "multiple_complex_groups": int(multiple.sum()),
        "multiple_complex_identical_labels": int((multiple & (groups["nunique"] == 1)).sum()),
        "multiple_complex_distinct_labels": int((multiple & (groups["nunique"] > 1)).sum()),
        "max_within_compound_label_spread": float((groups["max"] - groups["min"]).max()),
        "independent_assay_replicates_identifiable": False,
        "interpretation": "Rows identify crystal complexes. No measurement identifiers or replicate counts are supplied; repeated labels cannot establish independent assay replication.",
    }
