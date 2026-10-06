#!/usr/bin/env python
"""Build a local review archive with source, results and checksummed inputs.

This does not publish or upload anything. It includes only known public
project paths and the inputs explicitly listed in docs/input-checksums.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = {"src", "scripts", "tests", "paper", "docs", "config", "notebooks"}
PUBLIC_FILES = {"README.md", "LICENSE", "Makefile", "pyproject.toml", "plan.md",
                "requirements-repro.txt", ".python-version", ".gitignore", ".gitattributes"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=Path("release/evapro-0.1.0-review-20261004.tar.gz"))
    args = ap.parse_args()
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    paths = set()
    for name in listed.decode().split("\0"):
        if not name:
            continue
        path = Path(name)
        if (name in PUBLIC_FILES or path.parts[0] in PUBLIC_ROOTS
                or name.startswith(("data/processed/", "results/"))
                or name == "data/raw/README.md" or name.endswith(".manifest.json")):
            if (ROOT / path).is_file():
                paths.add(name)
    inputs = json.loads((ROOT / "docs/input-checksums.json").read_text())["files"]
    for name, expected in inputs.items():
        if not (name.startswith("data/raw/") or name.startswith("models/")):
            raise ValueError(f"Unexpected input-manifest path: {name}")
        if ".." in Path(name).parts or sha256(ROOT / name) != expected["sha256"]:
            raise ValueError(f"Input does not match committed checksum: {name}")
        paths.add(name)
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "results/figures").glob("*.png"))
    manifest = {
        "name": "evapro-0.1.0-review-20261004",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "published": False,
        "note": "Review snapshot, including uncommitted corrections. Per-file hashes identify its contents. See docs/data-licences.md for separate source/data terms.",
        "files": {name: {"sha256": sha256(ROOT / name), "bytes": (ROOT / name).stat().st_size}
                  for name in sorted(paths)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as output:
            temporary = Path(output.name)
            with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as archive:
                    for name in sorted(paths):
                        info = archive.gettarinfo(str(ROOT / name), arcname=name)
                        info.mtime = info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        # Store regular files, never links into a user's machine.
                        if not info.isfile():
                            raise ValueError(f"Archive input is not a regular file: {name}")
                        with (ROOT / name).open("rb") as handle:
                            archive.addfile(info, handle)
                    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
                    info = tarfile.TarInfo("release-manifest.json")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n")
    print(f"Built {args.output}: {len(paths)} files, {args.output.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
