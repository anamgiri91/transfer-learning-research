#!/usr/bin/env python
"""Fetch raw assay records from public APIs into data/raw/, with manifests.

Usage: python scripts/fetch_data.py --target-chembl-id CHEMBL5062
Writes <source>.csv plus <source>.manifest.json recording the exact query,
retrieval date and row count, so a rebuild is checkable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
RAW = Path("data/raw")
ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]


def fetch_chembl_activities(target_chembl_id: str, page_size: int = 1000) -> pd.DataFrame:
    """Page through ChEMBL activities for one target."""
    rows, url = [], f"{CHEMBL_BASE}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(ACTIVITY_TYPES),
        "limit": page_size,
        "offset": 0,
    }
    while True:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("activities", [])
        rows.extend(batch)
        print(f"  fetched {len(rows)} records...")
        nxt = payload.get("page_meta", {}).get("next")
        if not nxt or not batch:
            break
        params["offset"] += page_size
    return pd.DataFrame(rows)


def normalise_chembl(df: pd.DataFrame, target_id: str) -> pd.DataFrame:
    """Map ChEMBL fields onto the schema curate.py expects."""
    if df.empty:
        return df
    out = pd.DataFrame({
        "smiles": df.get("canonical_smiles"),
        "standard_type": df.get("standard_type"),
        "standard_relation": df.get("standard_relation"),
        "standard_value_nm": pd.to_numeric(df.get("standard_value"), errors="coerce"),
        "standard_units": df.get("standard_units"),
        "target_id": target_id,
        "readout": "enzymatic",
        "confidence_score": pd.to_numeric(df.get("confidence_score"), errors="coerce"),
        "doc_year": pd.to_numeric(df.get("document_year"), errors="coerce"),
        "source": "chembl",
        "assay_chembl_id": df.get("assay_chembl_id"),
    })
    # Only nM rows are usable without further conversion; the rest are dropped
    # here and counted in the manifest so the loss is visible.
    return out[out["standard_units"] == "nM"]


def write_with_manifest(df: pd.DataFrame, name: str, query: dict, n_raw: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    csv_path = RAW / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (RAW / f"{name}.manifest.json").write_text(json.dumps({
        "name": name,
        "query": query,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "n_records_returned": n_raw,
        "n_records_kept": len(df),
        "sha256": digest,
    }, indent=2))
    (RAW / f"{name}.sha256").write_text(f"{digest}  {csv_path.name}\n")
    print(f"  wrote {csv_path} ({len(df)} rows of {n_raw} returned)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chembl-id", required=True,
                    help="e.g. the ChEMBL id for EV-A71 3C protease")
    ap.add_argument("--name", default=None, help="output basename, e.g. eva71_3c_chembl")
    args = ap.parse_args()

    name = args.name or f"{args.target_chembl_id.lower()}_chembl"
    print(f"Fetching ChEMBL activities for {args.target_chembl_id}")
    raw = fetch_chembl_activities(args.target_chembl_id)
    if raw.empty:
        print("  no records returned -- check the target id")
        return 1
    df = normalise_chembl(raw, args.target_chembl_id)
    write_with_manifest(df, name, {"target_chembl_id": args.target_chembl_id,
                                   "standard_types": ACTIVITY_TYPES}, len(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
