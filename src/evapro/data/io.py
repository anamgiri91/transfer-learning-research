"""Dataset loading. CSV or parquet, whichever is present."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")


def load_dataset(target: str = "eva71_2a") -> pd.DataFrame:
    for ext, reader in ((".csv", pd.read_csv), (".parquet", pd.read_parquet)):
        path = PROCESSED / f"{target}{ext}"
        if path.exists():
            return reader(path)
    raise FileNotFoundError(
        f"no {target}.csv/.parquet in {PROCESSED} -- run scripts/prepare_openbind.py first."
    )
