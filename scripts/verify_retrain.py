#!/usr/bin/env python
"""Retrain a fixed subset of the amended arms in isolation and diff the result.

`verify_reproducibility.py` reports three kinds of evidence, and they are not
interchangeable:

  reconstructed   an artefact a stage rewrote during the run, then diffed
  compared only   a committed artefact no executed stage rewrote
  supplied input  a file verified by checksum and never rebuilt

The 120 Amendment 4 artefacts and the 80 Amendment 5 ones sit in *compared
only*, because retraining all of them costs ~7 h. That leaves the training
path itself unverified: the committed bytes are self-consistent, but nothing
had shown they can be produced again from the documented inputs.

This closes the gap for a **predetermined** subset — fixed in the source below
rather than chosen after seeing which cells happen to reproduce:

    T2v   seed 0, n = 50      (cheapest fine-tune cell)
    T2v   seed 0, n = 347     (most expensive)
    T4ft  seed 0, n = 347     (in-domain encoder path)
    T5ft  seed 0, n = 347     (chained encoder path)
    B3    seed 0, n = 347     (chemprop path)

Everything runs inside a throwaway `git worktree` at an immutable ref, with the
supplied inputs symlinked in read-only. **The committed outputs are never
written to** — they are the reference, read from the worktree's own pristine
checkout before the rerun clears and regenerates them.

Run: python scripts/verify_retrain.py [--ref REF] [--keep]
Exit 0 only if every retrained cell matches its committed counterpart within
the declared torch tolerance, with identical test-row identities.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
TOL_TORCH = 1e-6          # same tolerance verify_reproducibility declares

SUBSET = [
    ("T2v", 0, 50), ("T2v", 0, 347),
    ("T4ft", 0, 347), ("T5ft", 0, 347),
    ("B3", 0, 347),
]

SUPPLIED = ["data/raw/*.csv", "data/raw/OpenBind_EV-A71_2A.zip",
            "models/*.pt", "models/*.json"]


def paths(arm: str, seed: int, n: int):
    d = "results/metrics_b3" if arm == "B3" else "results/metrics_ft"
    pd_ = "results/predictions_b3" if arm == "B3" else "results/predictions_ft"
    tag = f"{arm}__scaffold__seed{seed}__n{n}"
    return f"{d}/{tag}.json", f"{pd_}/{tag}.npz", tag


def rel(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(1.0, abs(a), abs(b))


def make_worktree(ref: str, dest: Path) -> None:
    subprocess.run(["git", "worktree", "add", "--detach", str(dest), ref],
                   check=True, capture_output=True, text=True)
    for pat in SUPPLIED:
        for f in ROOT.glob(pat):
            tgt = dest / f.relative_to(ROOT)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            if not tgt.exists():
                tgt.symlink_to(f)
    src = ROOT / "data/raw/OpenBind_EV-A71_2A"
    if src.is_dir() and not (dest / "data/raw/OpenBind_EV-A71_2A").exists():
        (dest / "data/raw/OpenBind_EV-A71_2A").symlink_to(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="retrain-"))
    wt = tmp / "tree"
    print(f"retraining {len(SUBSET)} predetermined cells at {args.ref} in {wt}")
    print("the committed outputs are the reference and are never written to\n")
    try:
        make_worktree(args.ref, wt)

        # Reference copies, taken from the worktree's pristine checkout.
        ref_copy = tmp / "reference"
        targets = []
        for arm, seed, n in SUBSET:
            mp, pp, tag = paths(arm, seed, n)
            for rp in (mp, pp):
                src = wt / rp
                if not src.exists():
                    print(f"  [FAIL] {tag}: {rp} not in the baseline")
                    return 1
                dst = ref_copy / rp
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            targets.append((arm, seed, n, mp, pp, tag))

        # Clear only the targeted cells, so the scripts actually refit them.
        for _a, _s, _n, mp, pp, _t in targets:
            (wt / mp).unlink()
            (wt / pp).unlink()

        rows, bad = [], []
        for arm, seed, n, mp, pp, tag in targets:
            script = "scripts/run_dmpnn.py" if arm == "B3" else "scripts/run_finetune.py"
            cmd = [sys.executable, script, "--seeds", str(seed), "--sizes", str(n)]
            if arm != "B3":
                cmd += ["--arms", arm]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=wt)
            dt = time.time() - t0
            if r.returncode != 0 or not (wt / mp).exists():
                bad.append(f"{tag}: rerun failed\n{(r.stderr or r.stdout)[-700:]}")
                print(f"  [FAIL] {tag}")
                continue
            a = json.loads((ref_copy / mp).read_text())
            b = json.loads((wt / mp).read_text())
            worst = max(rel(float(a["metrics"][k]), float(b["metrics"][k]))
                        for k in a["metrics"] if a["metrics"][k] is not None)
            za = np.load(ref_copy / pp, allow_pickle=True)
            zb = np.load(wt / pp, allow_pickle=True)
            ident = bool(np.array_equal(za["inchikey"], zb["inchikey"]))
            labels = bool(np.array_equal(za["y_true"], zb["y_true"]))
            pw = float(max(rel(x, y) for x, y in zip(za["y_pred"], zb["y_pred"])))
            # Selection provenance must land in the same place too, or the
            # numbers agreeing would be a coincidence rather than a rerun.
            sel = all(a.get(k) == b.get(k) for k in
                      ("selected_lr", "n_train", "n_train_fitted", "n_internal_val"))
            ok = worst <= TOL_TORCH and pw <= TOL_TORCH and ident and labels and sel
            rows.append(dict(tag=tag, seconds=round(dt, 1), metric_drift=worst,
                             pred_drift=pw, rows_identical=ident,
                             labels_identical=labels, selection_identical=sel,
                             passed=ok))
            if not ok:
                bad.append(f"{tag}: metrics {worst:.3g}, preds {pw:.3g}, "
                           f"rows {ident}, labels {labels}, selection {sel}")
            print(f"  [{'ok  ' if ok else 'FAIL'}] {tag:26s} {dt:6.0f}s  "
                  f"metrics {worst:.2e}  preds {pw:.2e}  "
                  f"rows {'=' if ident else 'X'} sel {'=' if sel else 'X'}")

        out = Path("docs/retrain-verification.json")
        out.write_text(json.dumps(
            {"baseline_ref": subprocess.run(
                ["git", "rev-parse", "--short", args.ref], capture_output=True,
                text=True).stdout.strip(),
             "tolerance_relative": TOL_TORCH,
             "cells": rows,
             "note": "Predetermined subset retrained from supplied inputs in an "
                     "isolated worktree. Committed outputs were the reference and "
                     "were not modified."}, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}")
        if bad:
            print(f"\n{len(bad)} FAILURE(S):")
            for x in bad:
                print("  -", x)
            return 1
        print(f"All {len(rows)} retrained cells match their committed counterparts "
              f"(tolerance {TOL_TORCH:g} relative; row identities, labels and "
              f"selection provenance compared exactly).")
        return 0
    finally:
        if not args.keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           capture_output=True, text=True)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
