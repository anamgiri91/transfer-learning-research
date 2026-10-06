#!/usr/bin/env python
"""Download and verify the exact OpenBind release used by this study."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests

from evapro.data.openbind import ARCHIVE, DOWNLOAD_URL, verify_archive


def main() -> int:
    if ARCHIVE.exists():
        verify_archive(ARCHIVE)
        print(f"Verified cached release: {ARCHIVE}")
        return 0
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=ARCHIVE.parent, delete=False) as handle:
            name = Path(handle.name)
            with requests.get(DOWNLOAD_URL, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        verify_archive(name)
        os.replace(name, ARCHIVE)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)
    print(f"Downloaded and verified: {ARCHIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
