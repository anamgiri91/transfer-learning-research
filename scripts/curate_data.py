#!/usr/bin/env python
"""Apply the fidelity gate to everything in data/raw/ and write data/processed/.

Usage: python scripts/curate_data.py --target eva71_3c
Prints the full rejection funnel -- an unexplained drop is a bug, not a detail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evapro.data.curate import curate

RAW, INTERIM, PROCESSED = Path("data/raw"), Path("data/interim"), Path("data/processed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="eva71_3c")
    ap.add_argument("--pattern", default=None, help="glob within data/raw (default: <target>*.csv)")
    args = ap.parse_args()

    pattern = args.pattern or f"{args.target}*.csv"
    sources = sorted(RAW.glob(pattern))
    if not sources:
        raise SystemExit(f"no raw files matching {pattern!r} -- run `make data` / fetch_data.py first.")

    df = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True)
    print(f"Loaded {len(df)} raw records from {len(sources)} file(s)")

    censored: list[pd.DataFrame] = []
    out, report = curate(df, censored_out=censored)

    print("\nCuration funnel:")
    print(f"  input                    {report.n_input}")
    for reason, n in sorted(report.dropped.items(), key=lambda kv: -kv[1]):
        print(f"  dropped: {reason:<22} {n}")
    print(f"  censored (held out)      {report.n_censored_held_out}")
    print(f"  OUTPUT                   {report.n_output}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED / f"{args.target}.parquet", index=False)
    if censored:
        pd.concat(censored, ignore_index=True).to_parquet(
            INTERIM / f"{args.target}_censored.parquet", index=False)
    (PROCESSED / f"{args.target}.curation.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True))

    # plan.md §3.2 decision rule, enforced rather than remembered.
    if report.n_output < 300:
        print(f"\n!! N = {report.n_output} < 300. Per plan.md §3.2 the deep arms are")
        print("   reported as a negative result and the paper re-centres on baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
