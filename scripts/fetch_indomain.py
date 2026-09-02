#!/usr/bin/env python
"""Fetch the in-domain 3C / 3C-like protease transfer corpus (plan.md §3.1, arm T4).

`plan.md` names HRV 3C, CVB3 3C and SARS-CoV-2 3CL-pro as the related-protease
corpus for the in-domain arm. Two things about that list have to be checked
rather than assumed, and both are checked here:

1. ChEMBL indexes picornaviral proteases under the **whole genome polyprotein**,
   whose synonyms cover 2A, 3C, capsid and RNA polymerase alike. A target id is
   therefore not evidence that an activity measures 3C. Every record is screened
   on its assay description, and anything not clearly a 3C / 3C-like protease
   assay is dropped and counted.

2. Two of the obvious candidates are the wrong enzyme entirely -- poliovirus
   CHEMBL5127 is RNA-polymerase data and HCoV-NL63 CHEMBL3232683 is the
   papain-like protease PLP2, a different fold. They are excluded by name here,
   with the reason recorded, so that the exclusion is visible rather than silent.

Writes data/raw/indomain_<label>.csv plus a manifest per target.
Usage: python scripts/fetch_indomain.py
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from evapro.data.indomain import EXCLUDED, TARGETS, screen

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
RAW = Path("data/raw")
ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]

def get(url: str, params: dict | None = None, tries: int = 5) -> dict:
    """ChEMBL intermittently 500s and times out; retry with backoff."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(4 * (i + 1))
    raise RuntimeError("unreachable")


def fetch_target(target_id: str, page_size: int = 1000) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        j = get(f"{CHEMBL_BASE}/activity.json", {
            "target_chembl_id": target_id,
            "standard_type__in": ",".join(ACTIVITY_TYPES),
            "limit": page_size, "offset": offset,
        })
        batch = j.get("activities", [])
        rows.extend(batch)
        total = j["page_meta"]["total_count"]
        print(f"    {len(rows)}/{total}", flush=True)
        offset += page_size
        if not batch or offset >= total:
            break
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=sorted(TARGETS))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    for label in args.targets:
        tid, family = TARGETS[label]
        out = RAW / f"indomain_{label}.csv"
        if out.exists():
            print(f"{label}: cached"); continue
        print(f"{label} ({tid}, {family}):", flush=True)
        raw = fetch_target(tid)
        kept, funnel = screen(raw)
        kept.to_csv(out, index=False)
        (RAW / f"indomain_{label}.manifest.json").write_text(json.dumps({
            "label": label, "target_chembl_id": tid, "virus_family": family,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "activity_types": ACTIVITY_TYPES,
            "n_raw": int(len(raw)), **funnel,
            "excluded_targets_and_why": EXCLUDED,
            "source_script": "scripts/fetch_indomain.py",
        }, indent=2, sort_keys=True))
        print(f"  -> {funnel['kept']} kept of {len(raw)} "
              f"(dropped {funnel['dropped_not_3c']} not-3C, "
              f"{funnel['dropped_wrong_enzyme']} wrong-enzyme)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
