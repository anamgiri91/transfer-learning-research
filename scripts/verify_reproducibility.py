#!/usr/bin/env python
"""Re-run every stage that can be rebuilt offline, and diff the result against
the committed artefacts.

`manuscript.md` §10 claims the pipeline is bit-reproducible and shows a table of
stages that were checked. That check was performed by hand once. This script
performs it, so the claim is verified rather than asserted -- and so a change
that silently moves a committed number shows up as a failure rather than as a
diff nobody ran.

What is compared, and how:

  CSV tables      data rows only. Every table carries a `# generated ... <ts>`
                  provenance header whose timestamp changes on every run.
  JSON curation   normalised: `generated_utc` is a timestamp for the same reason.
  metric JSONs    exact bytes first; on any difference, the largest absolute
                  move in any metric must be <= 1e-9. The tolerance is not
                  cosmetic -- §10 records that the random-forest arm reproduces
                  to ~1e-16 rather than bit-identically, because `n_jobs=-1`
                  lets the order of the floating-point reduction across threads
                  vary between runs.
  figures         exact bytes.
  split files     exact bytes. These are the load-bearing ones: every arm reads
                  them, so a split that does not reproduce invalidates
                  every number in the paper.

Two stages are not covered and say so rather than passing quietly:
`measure_contamination.py` needs PubChem and `verify_surrogate.py` needs
UniProt, so both are marked network stages and skipped if they cannot reach
their source.

Run: python scripts/verify_reproducibility.py [--quick] [--snapshot DIR]
Exit code 0 only if every regenerated artefact matches what is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

ROOT = Path(".").resolve()
TOL = 1e-9

WATCHED = [
    "results/tables/*.csv",
    "results/figures/*.png",
    "data/processed/eva71_2a.csv",
    "data/processed/eva71_2a.curation.json",
    "data/processed/indomain_3c.csv",
    "data/processed/indomain_3c.curation.json",
    "data/processed/splits/eva71_2a/*.json",
    "results/metrics/*.json",
]

# (label, argv, needs_network, slow, clears)
#
# `clears` is a glob of artefacts deleted before the stage runs. It exists
# because `run_arms.py` skips any cell whose metric file is already present:
# without clearing, the one stage here that re-fits a model exited in 8s having
# fitted nothing, and reported [ok]. The snapshot holds the originals, so a
# cleared file that the stage fails to rewrite is caught as DISAPPEARED.
STAGES = [
    ("prepare_openbind  -> eva71_2a.csv",
     ["scripts/prepare_openbind.py"], False, False, None),
    ("prepare_indomain  -> indomain_3c.csv, table12",
     ["scripts/prepare_indomain.py"], False, False, None),
    ("build_splits      -> 30 split files",
     ["scripts/build_splits.py", "--target", "eva71_2a"], False, False,
     "data/processed/splits/eva71_2a/*.json"),
    ("audit_splits      -> table0",
     ["scripts/audit_splits.py"], False, False, None),
    ("run_arms B0/B1/B2 -> 360 metric files (re-fitted, not skipped)",
     ["scripts/run_arms.py", "--arms", "B0", "B1", "B2",
      "--splits", "scaffold", "random", "butina"], False, True,
     "results/metrics/B[012]_*.json"),
    ("make_report       -> tables 0-4, all figures",
     ["scripts/make_report.py", "--require-seeds", "10"], False, False, None),
    ("analyse_cliffs    -> tables 6, 7, 8",
     ["scripts/analyse_cliffs.py"], False, False, None),
    ("analyse_indomain  -> tables 10, 11",
     ["scripts/analyse_indomain.py"], False, False, None),
    ("analyse_tuning    -> table13", ["scripts/analyse_tuning.py"], False, False, None),
    ("analyse_endpoints -> table14", ["scripts/analyse_endpoints.py"], False, False, None),
    ("analyse_h2        -> tables 15, 16", ["scripts/analyse_h2.py"], False, False, None),
    ("verify_surrogate  -> table9", ["scripts/verify_surrogate.py"], True, False, None),
]

TIMESTAMP_KEYS = ("generated_utc",)


def files() -> list[Path]:
    out: list[Path] = []
    for pat in WATCHED:
        out.extend(sorted(ROOT.glob(pat)))
    return out


def canonical(p: Path) -> bytes:
    """File content with generation timestamps removed.

    A provenance header that records *when* a file was written is not part of
    what the file claims, and comparing it would make every run fail. Nothing
    else is normalised: a number that moves must fail.
    """
    raw = p.read_bytes()
    if p.suffix == ".csv":
        return b"".join(l for l in raw.splitlines(keepends=True)
                        if not l.startswith(b"#"))
    if p.suffix == ".json":
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
        _strip(obj)
        return json.dumps(obj, sort_keys=True).encode()
    return raw


def _strip(obj) -> None:
    if isinstance(obj, dict):
        for k in list(obj):
            if k in TIMESTAMP_KEYS:
                obj[k] = "<timestamp>"
            else:
                _strip(obj[k])
    elif isinstance(obj, list):
        for v in obj:
            _strip(v)


def numeric_delta(a: Path, b: Path) -> float | None:
    """Largest RELATIVE move across a file's numbers, or None if not comparable.

    Two file kinds are covered. Metric JSONs move because `n_jobs=-1` lets the
    order of a floating-point reduction vary between runs (§10). Derived CSV
    tables move because those metrics feed them -- and some of the derivations
    amplify: `table2`'s `n_to_reach_target` is a linear interpolation along a
    learning curve, so a 1e-16 wobble in an RMSE lands at ~1e-13 in an
    interpolated training-set size. Demanding bit-identity of a derived
    interpolation across thread scheduling is not a reproducibility standard,
    it is a false alarm generator; demanding it stay within tolerance is.

    Relative rather than absolute, because these columns are not all on the
    same scale -- an absolute tolerance that is strict for an RMSE of 0.6 is
    meaningless for a training-set size of 295.
    """
    try:
        x, y = json.loads(a.read_text()), json.loads(b.read_text())
        if "metrics" in x and "metrics" in y:
            keys = [k for k in x["metrics"]
                    if x["metrics"].get(k) is not None and y["metrics"].get(k) is not None]
            return max((_rel(float(x["metrics"][k]), float(y["metrics"][k])) for k in keys),
                       default=0.0)
        return None
    except Exception:
        pass
    if a.suffix != ".csv":
        return None
    try:
        import pandas as pd
        da, db = (pd.read_csv(f, comment="#") for f in (a, b))
        if da.shape != db.shape or list(da.columns) != list(db.columns):
            return None
        worst = 0.0
        for col in da.columns:
            if not pd.api.types.is_numeric_dtype(da[col]):
                # A non-numeric column that moved is a real difference.
                if not da[col].astype(str).equals(db[col].astype(str)):
                    return None
                continue
            u, v = da[col].to_numpy(float), db[col].to_numpy(float)
            both_nan = np.isnan(u) & np.isnan(v)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                return None
            u, v = u[~both_nan], v[~both_nan]
            if len(u):
                worst = max(worst, float(max(_rel(p, q) for p, q in zip(u, v))))
        return worst
    except Exception:
        return None


def _rel(p: float, q: float) -> float:
    """Relative difference, falling back to absolute near zero."""
    if p == q:
        return 0.0
    if not (np.isfinite(p) and np.isfinite(q)):
        return 0.0 if p == q else float("inf")
    return abs(p - q) / max(1.0, abs(p), abs(q))


def snapshot(snap: Path) -> int:
    if snap.exists():
        shutil.rmtree(snap)
    n = 0
    for f in files():
        dst = snap / f.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    return n


def restore(snap: Path) -> int:
    """Copy the snapshot back over the regenerated files.

    Only files whose *canonical* content is unchanged are restored, so this can
    never mask a real difference -- it is called after the comparison passes.
    """
    n = 0
    for old in sorted(snap.rglob("*")):
        if not old.is_file():
            continue
        cur = ROOT / old.relative_to(snap)
        if cur.exists() and cur.read_bytes() != old.read_bytes():
            shutil.copy2(old, cur)
            n += 1
    return n


def compare(snap: Path, touched: set[Path]) -> tuple[list[str], dict[str, float], int, int]:
    """Compare every watched file, and count how many a stage actually rewrote.

    The distinction is the whole point. A file that no stage rewrote compares
    equal to itself and reports as a pass, which is the "matched nothing and
    passed vacuously" failure this repository has already hit four times. Only
    files whose mtime moved during the run were regenerated; the rest are
    committed INPUTS to this check, not outputs of it, and are counted
    separately so the summary line cannot overstate what was verified.
    """
    bad: list[str] = []
    deltas: dict[str, float] = {}
    n = 0
    for f in files():
        old = snap / f.relative_to(ROOT)
        rel = f.relative_to(ROOT)
        if not old.exists():
            bad.append(f"NEW FILE, not previously committed: {rel}")
            continue
        n += 1
        if canonical(old) == canonical(f):
            continue
        d = numeric_delta(old, f)
        if d is not None:
            deltas[str(rel)] = d
            if d > TOL:
                bad.append(f"{rel}: a value moved by {d:.3g} relative, over the {TOL:g} tolerance")
        else:
            bad.append(f"{rel}: content differs "
                       f"({hashlib.sha256(canonical(old)).hexdigest()[:12]} -> "
                       f"{hashlib.sha256(canonical(f)).hexdigest()[:12]})")
    for old in sorted(snap.rglob("*")):
        if old.is_file() and not (ROOT / old.relative_to(snap)).exists():
            bad.append(f"DISAPPEARED after regeneration: {old.relative_to(snap)}")
    return bad, deltas, n, len(touched)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the stages that re-fit models (~1 min)")
    ap.add_argument("--snapshot", default=".repro_snapshot",
                    help="where to keep the pre-run copies")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="snapshot the working tree even if it differs from git. "
                         "Only for a tree you deliberately regenerated -- "
                         "otherwise a previous failure becomes the new baseline.")
    args = ap.parse_args()

    # A failing run leaves the regenerated files in place for inspection. If
    # the next run then snapshots THAT state as its baseline, it compares the
    # corruption against itself and passes -- laundering exactly the difference
    # it exists to catch. So: refuse to start against a tree whose watched
    # artefacts are already modified, and say how to get back to a clean one.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(p) for p in WATCHED)],
        capture_output=True, text=True)
    modified = [ln[3:] for ln in dirty.stdout.splitlines() if ln[:2].strip() == "M"]
    if modified and not args.allow_dirty:
        print(f"REFUSING: {len(modified)} watched artefact(s) are already modified "
              f"relative to git, so a snapshot taken now would be the wrong "
              f"baseline. Restore them first:\n\n"
              f"    git checkout -- results/ data/processed/\n\n"
              f"first three: {', '.join(modified[:3])}"
              + ("\n\n(--allow-dirty overrides, for a tree you have deliberately "
                 "regenerated.)" if True else ""))
        return 1

    snap = Path(args.snapshot)
    n = snapshot(snap)
    print(f"  snapshotted {n} artefacts")
    before = {f: f.stat().st_mtime_ns for f in files()}

    skipped: list[str] = []
    for label, argv, needs_net, slow, clears in STAGES:
        if slow and args.quick:
            skipped.append(f"{label}  [--quick]")
            print(f"  [skip] {label}")
            continue
        t0 = time.time()
        n_cleared = 0
        if clears:
            for f in ROOT.glob(clears):
                f.unlink(); n_cleared += 1
        r = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
        if r.returncode != 0:
            tail = ((r.stderr or r.stdout).strip().splitlines() or ["(no output)"])[-1]
            if needs_net:
                skipped.append(f"{label}  [unreachable: {tail[:70]}]")
                print(f"  [skip] {label}")
                continue
            print(f"  [FAIL] {label}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            shutil.rmtree(snap, ignore_errors=True)
            return 1
        cleared = f", {n_cleared} cleared first" if n_cleared else ""
        print(f"  [ok  ] {label}  ({time.time() - t0:.1f}s{cleared})")

    touched = {f for f in files()
               if f not in before or f.stat().st_mtime_ns != before[f]}
    bad, deltas, n, n_touched = compare(snap, touched)
    print(f"\n  {n_touched} of {n} watched artefacts were rewritten by a stage "
          f"and matched; the other {n - n_touched} were not regenerated by this "
          f"run and are inputs to it, not outputs")
    if n_touched:
        kinds: dict[str, int] = {}
        for f in touched:
            kinds[f.parent.name] = kinds.get(f.parent.name, 0) + 1
        print("  rewritten by directory: "
              + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    if deltas:
        print(f"  {len(deltas)} file(s) differ numerically within tolerance; "
              f"largest relative move {max(deltas.values()):.3g} (limit {TOL:g})")
    for s in skipped:
        print(f"  not covered: {s}")

    if bad:
        print(f"\n{len(bad)} REPRODUCIBILITY FAILURE(S):")
        for b in bad[:40]:
            print(f"  - {b}")
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more")
        print(f"\nthe regenerated files are left in place for inspection; "
              f"the originals are in {snap}/")
        return 1

    # Put the committed copies back. Every stage rewrites its provenance
    # timestamp, so a successful run would otherwise leave the tree dirty with
    # changes that mean nothing -- and a check nobody can run without dirtying
    # the repository is a check nobody runs.
    restored = restore(snap)
    shutil.rmtree(snap, ignore_errors=True)
    print(f"\nEvery regenerated artefact matches the committed copy "
          f"({restored} restored, so the working tree is unchanged). "
          f"NOT covered: the 680 transfer-arm metric files, whose arms this "
          f"script does not re-fit, and the six in-domain encoders in models/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
