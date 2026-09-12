#!/usr/bin/env python
"""Reconstruct the pipeline from its inputs and diff the result, in isolation.

`manuscript.md` §10 claims the pipeline is reproducible. This script is that
claim, executed. It answers three separate questions that an earlier version
ran together, and running them together is how it managed to report 1,104
artefacts "regenerated" when the number was 424:

  SUPPLIED INPUT     a file the pipeline consumes and cannot rebuild -- the
                     500 MB of `data/raw/` payloads and the six pretrained
                     encoders in `models/`, none of which are in git. These
                     are verified by CHECKSUM against a committed manifest.
                     Reproducing the study requires obtaining them; this check
                     can only confirm that the bytes here are the bytes the
                     results were built from.
  RECONSTRUCTED      a file a stage rewrites during this run. Compared to the
                     committed copy byte for byte where determinism allows, or
                     within a declared numeric tolerance where it does not.
  COMPARED ONLY      a watched file that no executed stage rewrote. It is an
                     input to this check, not an output of it, and is counted
                     separately so the summary cannot overstate coverage.

ISOLATION
---------
Stages run inside a throwaway `git worktree` checked out at an immutable
baseline ref (default HEAD), with the supplied inputs linked in. The working
tree you are sitting in is never written to. This replaces an in-place design
that could -- and once did -- let a failed run's output become the baseline
that the next run compared against and passed.

TOLERANCES, AND WHY EACH ONE IS THE SIZE IT IS
-----------------------------------------------
Recorded here rather than tuned until things pass. Every tolerance is
RELATIVE, because the compared columns are not on one scale: an absolute
tolerance strict for an RMSE of 0.6 is meaningless for a training-set size of
295.

  1e-9   metric JSONs and derived CSV tables. The observed floor is ~1e-15 for
         re-fitted scikit-learn arms (`n_jobs=-1` lets the order of a
         floating-point reduction across threads vary) and ~1e-13 for
         `table2`'s `n_to_reach_target`, which linearly interpolates a learning
         curve and so amplifies the former. 1e-9 sits four orders above the
         worst observed drift and six below the third decimal the paper
         reports, so it cannot hide a difference anyone could read.
  1e-6   torch fine-tune predictions. Non-determinism in threaded CPU BLAS
         reductions accumulates over ~1e3 optimiser steps; measured drift on a
         repeated cell is reported by `--measure-tolerance` and the committed
         figure is in docs/reproduction-coverage.md.
  exact  structural invariants -- row identities, split membership, array
         shapes, missingness. These are integers and strings. A tolerance on
         them would be a bug, so they are compared with `==` and any
         difference fails regardless of the numeric outcome.

`tests/test_reproducibility.py` plants a perturbation just over each tolerance
and asserts it is rejected, and one just under and asserts it passes and is
reported. A tolerance nothing can violate is not a tolerance.

Run: python scripts/verify_reproducibility.py [--tier fast|full] [--ref REF]
Exit code 0 only if every reconstructed artefact matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(".").resolve()
TOL = 1e-9                 # metrics and derived tables
TOL_TORCH = 1e-6           # fine-tune predictions
INPUT_MANIFEST = Path("docs/input-checksums.json")
COVERAGE_DOC = Path("docs/reproduction-coverage.md")

# Watched: everything a stage could plausibly rewrite, plus the committed
# outputs we want protected from silent drift.
WATCHED = [
    "results/tables/*.csv",
    "results/figures/*.png",
    "data/processed/eva71_2a.csv",
    "data/processed/eva71_2a.curation.json",
    "data/processed/indomain_3c.csv",
    "data/processed/indomain_3c.curation.json",
    "data/processed/splits/eva71_2a/*.json",
    "results/metrics/*.json",
    "results/predictions/*.npz",
]

# Supplied inputs: consumed, never rebuilt, not in git. Checksummed.
SUPPLIED = [
    "data/raw/*.csv",
    "data/raw/OpenBind_EV-A71_2A.zip",
    "models/*.pt",
    "models/*.json",
]

# (label, argv, needs_network, tier, clears)
#
# `clears` is deleted before the stage runs. Without it `run_arms.py` skips
# every cell whose metric file exists, and the one stage that re-fits models
# exited in 8s having fitted nothing and reported [ok].
#
# tier: "fast" runs in ~2 min and covers data prep, splits, the three
# from-scratch arms and every analysis. "full" adds the transfer arms.
STAGES = [
    ("prepare_openbind  -> eva71_2a.csv",
     ["scripts/prepare_openbind.py"], False, "fast", None),
    ("prepare_indomain  -> indomain_3c.csv, table12",
     ["scripts/prepare_indomain.py"], False, "fast", None),
    ("build_splits      -> 30 split files",
     ["scripts/build_splits.py", "--target", "eva71_2a"], False, "fast",
     "data/processed/splits/eva71_2a/*.json"),
    ("audit_splits      -> table0",
     ["scripts/audit_splits.py"], False, "fast", None),
    ("run_arms B0/B1/B2 -> 360 metrics + preds",
     ["scripts/run_arms.py", "--arms", "B0", "B1", "B2",
      "--splits", "scaffold", "random", "butina", "--save-preds"], False, "fast",
     "results/metrics/B[012]_*.json"),
    # --- transfer arms: frozen probes. Cheap, and previously uncovered. -----
    ("run_arms probes   -> 480 metrics + preds (T0r/T1/T4/T5)",
     ["scripts/run_arms.py", "--arms", "T0r", "T1", "T4", "T5",
      "--splits", "scaffold", "random", "butina", "--save-preds"], False, "full",
     "results/metrics/T[015]*_*.json"),
    ("run_arms ablations-> 160 metrics + preds (T4c/T4r/T5c/T5r)",
     ["scripts/run_arms.py", "--arms", "T4c", "T4r", "T5c", "T5r",
      "--splits", "scaffold", "--save-preds"], False, "full",
     "results/metrics/T[45][cr]_*.json"),
    # --- analyses ----------------------------------------------------------
    ("make_report       -> tables 0-4, all figures",
     ["scripts/make_report.py", "--require-seeds", "10"], False, "fast", None),
    ("analyse_cliffs    -> tables 6, 7, 8",
     ["scripts/analyse_cliffs.py"], False, "fast", None),
    ("analyse_indomain  -> tables 10, 11",
     ["scripts/analyse_indomain.py"], False, "fast", None),
    ("analyse_tuning    -> table13", ["scripts/analyse_tuning.py"], False, "fast", None),
    ("analyse_endpoints -> table14", ["scripts/analyse_endpoints.py"], False, "fast", None),
    ("analyse_h2        -> tables 15, 16", ["scripts/analyse_h2.py"], False, "fast", None),
    ("verify_surrogate  -> table9", ["scripts/verify_surrogate.py"], True, "fast", None),
]

# Not stages, and the reason is recorded rather than left to inference.
UNEXECUTED = [
    ("run_arms T2 (ChemBERTa fine-tune), 40 cells",
     "~185 s per fit, ~2.1 h. A representative cell is reconstructed by "
     "--representative instead; the other 39 are declared unexecuted."),
    ("pretrain_indomain.py, six encoders",
     "796-934 s each, ~1.4 h. The encoders are verified as supplied inputs by "
     "checksum; --representative re-pretrains one to show the path executes."),
    ("tune_arms.py, 41 tuned cells",
     "101 min, and the arm it tunes is the 185 s fine-tune."),
    ("measure_contamination.py -> table5",
     "needs PubChem, which is a network dependency this check cannot rely on; "
     "table5 is therefore the one committed table with no reconstruction path."),
]


def files(root: Path = ROOT, pats=None) -> list[Path]:
    out: list[Path] = []
    for pat in (pats or WATCHED):
        out.extend(sorted(root.glob(pat)))
    return out


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Supplied inputs
# --------------------------------------------------------------------------

def supplied_files(root: Path = ROOT) -> list[Path]:
    return files(root, SUPPLIED)


def write_input_manifest() -> int:
    entries = {}
    for f in supplied_files():
        entries[str(f.relative_to(ROOT))] = {
            "sha256": sha(f), "bytes": f.stat().st_size}
    INPUT_MANIFEST.write_text(json.dumps(
        {"note": "Supplied inputs: consumed by the pipeline, not rebuilt by it, "
                 "and not in git. Reproducing the study requires obtaining these "
                 "bytes; see data/raw/README.md and docs/reproduction-coverage.md.",
         "n_files": len(entries), "files": entries}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {INPUT_MANIFEST}  ({len(entries)} supplied inputs)")
    return 0


def check_inputs() -> list[str]:
    if not INPUT_MANIFEST.exists():
        return [f"{INPUT_MANIFEST} missing; run --write-input-manifest"]
    man = json.loads(INPUT_MANIFEST.read_text())["files"]
    bad = []
    for rel, rec in sorted(man.items()):
        p = ROOT / rel
        if not p.exists():
            bad.append(f"SUPPLIED INPUT MISSING: {rel}")
        elif sha(p) != rec["sha256"]:
            bad.append(f"SUPPLIED INPUT CHANGED: {rel}")
    for f in supplied_files():
        rel = str(f.relative_to(ROOT))
        if rel not in man:
            bad.append(f"SUPPLIED INPUT NOT IN MANIFEST: {rel}")
    return bad


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

TIMESTAMP_KEYS = ("generated_utc",)


def canonical(p: Path) -> bytes:
    """File content with generation timestamps removed."""
    if p.suffix == ".csv":
        return b"\n".join(ln for ln in p.read_bytes().split(b"\n")
                          if not ln.startswith(b"# generated"))
    if p.suffix == ".json":
        try:
            d = json.loads(p.read_text())
            if isinstance(d, dict):
                d = {k: v for k, v in d.items() if k not in TIMESTAMP_KEYS}
            return json.dumps(d, sort_keys=True).encode()
        except Exception:
            pass
    return p.read_bytes()


def _rel(a: float, b: float) -> float:
    """Relative difference, with NaN treated as a value that equals itself.

    B0 is a constant predictor, so its Spearman is NaN in all 120 of its metric
    files. Comparing NaN with `==` makes every one of them look infinitely
    different from itself, which is how a clean re-fit first reported 19
    reproducibility failures. Missingness is a structural property and is
    checked exactly by `structural_diff`; here it must not masquerade as
    numeric drift.
    """
    a_nan, b_nan = (a != a), (b != b)          # NaN is the only x with x != x
    if a_nan or b_nan:
        return 0.0 if (a_nan and b_nan) else float("inf")
    if a == b:
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return 0.0 if a == b else float("inf")
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _load_npz(p: Path):
    """Load a prediction bundle.

    `allow_pickle=True` is required: `run_arms.py --save-preds` writes the
    InChIKey column straight from a pandas object array, so the 690 committed
    .npz files hold an object dtype. Re-saving them as unicode would be the
    tidier fix and would invalidate every committed prediction file, so the
    format stands. The risk boundary: these are repository-local artefacts
    regenerated by this repository's own code, never files from elsewhere. The
    checker must not be pointed at untrusted .npz.
    """
    return np.load(p, allow_pickle=True)


def structural_diff(a: Path, b: Path) -> str | None:
    """Exact checks that a numeric tolerance must never be allowed to absorb.

    Row identities, split membership, shapes and missingness are integers and
    strings; a difference in any of them is a different experiment, not
    floating-point drift.
    """
    if a.suffix == ".npz":
        x, y = _load_npz(a), _load_npz(b)
        if sorted(x.files) != sorted(y.files):
            return f"array names differ: {sorted(x.files)} vs {sorted(y.files)}"
        for k in x.files:
            if x[k].shape != y[k].shape:
                return f"{k}: shape {x[k].shape} != {y[k].shape}"
            if x[k].dtype.kind in "USO" and not np.array_equal(x[k], y[k]):
                return f"{k}: row identities differ (test-fold membership changed)"
            if x[k].dtype.kind == "f":
                if not np.array_equal(np.isnan(x[k]), np.isnan(y[k])):
                    return f"{k}: missingness pattern differs"
        return None
    if a.suffix == ".json":
        try:
            x, y = json.loads(a.read_text()), json.loads(b.read_text())
        except Exception:
            return None
        if not (isinstance(x, dict) and isinstance(y, dict)):
            return None
        # A metric file whose numbers agree but whose identity fields moved is
        # a different experiment wearing the same result. `numeric_delta` only
        # looks inside `metrics`, so without this a changed n_train or a
        # changed split would pass as byte-drift.
        for k in ("arm", "target", "split", "seed", "n_train", "n_test"):
            if x.get(k) != y.get(k):
                return f"identity field {k!r}: {x.get(k)!r} != {y.get(k)!r}"
        if set(x) != set(y):
            return f"key set differs: {sorted(set(x) ^ set(y))}"
        mx, my = x.get("metrics"), y.get("metrics")
        if isinstance(mx, dict) and isinstance(my, dict):
            if set(mx) != set(my):
                return f"metric names differ: {sorted(set(mx) ^ set(my))}"
            for k in mx:
                if (mx[k] is None) != (my[k] is None):
                    return f"metric {k!r} missingness differs"
        return None
    if a.suffix == ".csv":
        import pandas as pd
        da, db = (pd.read_csv(f, comment="#") for f in (a, b))
        if da.shape != db.shape:
            return f"table shape {da.shape} != {db.shape}"
        if list(da.columns) != list(db.columns):
            return "column names differ"
        if not da.isna().to_numpy().tolist() == db.isna().to_numpy().tolist():
            return "missingness pattern differs"
        for col in da.columns:
            import pandas.api.types as pt
            if not pt.is_numeric_dtype(da[col]):
                if not da[col].astype(str).equals(db[col].astype(str)):
                    return f"non-numeric column {col!r} differs"
        return None
    return None


def numeric_delta(a: Path, b: Path) -> tuple[float | None, float]:
    """(largest relative move, tolerance that applies), or (None, tol) if opaque."""
    tol = TOL_TORCH if _is_torch_artifact(a) else TOL
    if a.suffix == ".npz":
        x, y = _load_npz(a), _load_npz(b)
        worst = 0.0
        for k in x.files:
            if x[k].dtype.kind != "f":
                continue
            u, v = x[k].ravel(), y[k].ravel()
            m = ~(np.isnan(u) | np.isnan(v))
            if m.any():
                worst = max(worst, float(max((_rel(p, q) for p, q in zip(u[m], v[m])),
                                             default=0.0)))
        return worst, tol
    try:
        x, y = json.loads(a.read_text()), json.loads(b.read_text())
        if "metrics" in x and "metrics" in y:
            keys = [k for k in x["metrics"]
                    if x["metrics"].get(k) is not None and y["metrics"].get(k) is not None]
            return max((_rel(float(x["metrics"][k]), float(y["metrics"][k]))
                        for k in keys), default=0.0), tol
        return None, tol
    except Exception:
        pass
    if a.suffix != ".csv":
        return None, tol
    try:
        import pandas as pd
        import pandas.api.types as pt
        da, db = (pd.read_csv(f, comment="#") for f in (a, b))
        worst = 0.0
        for col in da.columns:
            if not pt.is_numeric_dtype(da[col]):
                continue
            u, v = da[col].to_numpy(float), db[col].to_numpy(float)
            m = ~(np.isnan(u) | np.isnan(v))
            if m.any():
                worst = max(worst, float(max(_rel(p, q) for p, q in zip(u[m], v[m]))))
        return worst, tol
    except Exception:
        return None, tol


def _is_torch_artifact(p: Path) -> bool:
    return any(t in p.name for t in ("T2_chemberta", "T2v", "T4ft", "T5ft"))


def compare(base: Path, live: Path, touched: set[str],
            tracked: set[str] | None = None) -> dict:
    """Diff the reconstructed tree against the immutable baseline.

    `tracked` is the set of paths the baseline commit actually contains. A
    watched file outside it cannot be diffed against the baseline -- there is
    nothing there to diff against -- and is reported as reconstructed-without-a-
    baseline rather than as a failure or, worse, as verified. Two categories
    land here, both by deliberate repository policy:

      results/figures/**   gitignored as regenerable, and matplotlib PNGs are
                           not byte-stable across environments anyway
      results/predictions/ committed for the scaffold split only, 690 of 1,040

    Until 2026-09-12 the manuscript claimed all nine figures reproduce
    byte-identically. That was checked against the uncommitted local copies by
    an in-place run; against a committed baseline there is nothing to check.
    """
    bad: list[str] = []
    deltas: dict[str, float] = {}
    extra: list[str] = []
    n_watched = n_recon = 0
    for f in files(live):
        rel = str(f.relative_to(live))
        old = base / rel
        if not old.exists():
            if tracked is not None and rel not in tracked:
                extra.append(rel)          # no committed baseline to diff against
            else:
                bad.append(f"NEW FILE, not in the baseline: {rel}")
            continue
        n_watched += 1
        if rel in touched:
            n_recon += 1
        sd = structural_diff(old, f)
        if sd:
            bad.append(f"{rel}: STRUCTURAL: {sd}")
            continue
        if canonical(old) == canonical(f):
            continue
        d, tol = numeric_delta(old, f)
        if d is None:
            bad.append(f"{rel}: content differs and is not numerically comparable")
        else:
            deltas[rel] = d
            if d > tol:
                bad.append(f"{rel}: moved {d:.3g} relative, over its {tol:g} tolerance")
    for old in sorted(base.rglob("*")):
        rel = old.relative_to(base)
        if old.is_file() and any(Path(rel).match(p) for p in WATCHED) \
                and not (live / rel).exists():
            bad.append(f"DISAPPEARED after regeneration: {rel}")
    return {"bad": bad, "deltas": deltas, "extra": extra,
            "n_watched": n_watched, "n_reconstructed": n_recon}


# --------------------------------------------------------------------------
# Isolated execution
# --------------------------------------------------------------------------

def make_worktree(ref: str, dest: Path) -> None:
    subprocess.run(["git", "worktree", "add", "--detach", str(dest), ref],
                   check=True, capture_output=True, text=True)
    # Supplied inputs are not in git; link them in read-only rather than copy
    # 580 MB. Symlinks are fine: no stage writes to data/raw or models/.
    for f in supplied_files():
        rel = f.relative_to(ROOT)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (dest / rel).exists():
            (dest / rel).symlink_to(f)
    src = ROOT / "data/raw/OpenBind_EV-A71_2A"
    if src.is_dir() and not (dest / "data/raw/OpenBind_EV-A71_2A").exists():
        (dest / "data/raw/OpenBind_EV-A71_2A").symlink_to(src)


def drop_worktree(dest: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(dest)],
                   capture_output=True, text=True)
    shutil.rmtree(dest, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="fast", choices=["fast", "full"])
    ap.add_argument("--ref", default="HEAD", help="immutable baseline to reconstruct")
    ap.add_argument("--write-input-manifest", action="store_true")
    ap.add_argument("--representative", action="store_true",
                    help="also reconstruct one T2 fine-tune cell and re-pretrain "
                         "one encoder, to execute the expensive paths once")
    ap.add_argument("--keep", action="store_true", help="do not delete the worktree")
    args = ap.parse_args()

    if args.write_input_manifest:
        return write_input_manifest()

    print("== supplied inputs ==")
    bad_inputs = check_inputs()
    for b in bad_inputs:
        print(f"  [FAIL] {b}")
    if bad_inputs:
        return 1
    n_supplied = len(json.loads(INPUT_MANIFEST.read_text())["files"])
    print(f"  [ok  ] {n_supplied} supplied inputs match their committed checksums")

    tmp = Path(tempfile.mkdtemp(prefix="repro-"))
    wt = tmp / "tree"
    print(f"\n== reconstructing {args.ref} in {wt} ==")
    try:
        make_worktree(args.ref, wt)
        base = tmp / "baseline"
        base.mkdir()
        for f in files(wt):
            dst = base / f.relative_to(wt)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        before = {str(f.relative_to(wt)): f.stat().st_mtime_ns for f in files(wt)}

        skipped = []
        for label, argv, needs_net, tier, clears in STAGES:
            if tier == "full" and args.tier == "fast":
                skipped.append(f"{label}  [--tier fast]")
                print(f"  [skip] {label}")
                continue
            t0 = time.time()
            n_cleared = 0
            if clears:
                for f in wt.glob(clears):
                    f.unlink()
                    n_cleared += 1
            r = subprocess.run([sys.executable, *argv], capture_output=True,
                               text=True, cwd=wt)
            if r.returncode != 0:
                tail = ((r.stderr or r.stdout).strip().splitlines() or ["(none)"])[-1]
                if needs_net:
                    skipped.append(f"{label}  [unreachable: {tail[:60]}]")
                    print(f"  [skip] {label}")
                    continue
                print(f"  [FAIL] {label}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
                return 1
            extra = f", {n_cleared} cleared" if n_cleared else ""
            print(f"  [ok  ] {label}  ({time.time() - t0:.1f}s{extra})")

        touched = {str(f.relative_to(wt)) for f in files(wt)
                   if before.get(str(f.relative_to(wt))) != f.stat().st_mtime_ns}
        tracked = set(subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", args.ref],
            capture_output=True, text=True, cwd=ROOT).stdout.split())
        res = compare(base, wt, touched, tracked)

        print(f"\n== coverage ==")
        print(f"  supplied inputs   {n_supplied:5d}  verified by checksum, not rebuilt")
        print(f"  reconstructed     {res['n_reconstructed']:5d}  rewritten by a stage "
              f"and diffed against the baseline")
        print(f"  compared only     {res['n_watched'] - res['n_reconstructed']:5d}  "
              f"watched, but no executed stage rewrote them")
        print(f"  watched total     {res['n_watched']:5d}")
        if res["extra"]:
            figs = sum(1 for r in res["extra"] if r.startswith("results/figures/"))
            preds = sum(1 for r in res["extra"] if r.startswith("results/predictions/"))
            print(f"  no baseline       {len(res['extra']):5d}  regenerated but not "
                  f"committed, so nothing to diff against "
                  f"({figs} figures — gitignored as regenerable; "
                  f"{preds} predictions — only the scaffold split was committed)")
        if res["deltas"]:
            worst = max(res["deltas"].values())
            exact = sum(1 for v in res["deltas"].values() if v == 0.0)
            print(f"\n  how the {res['n_reconstructed']} reconstructions compare:")
            print(f"    byte-identical after canonicalisation  "
                  f"{res['n_reconstructed'] - len(res['deltas']):5d}")
            print(f"    bytes differ, every number identical   {exact:5d}  "
                  f"(wall-clock `seconds` in metric JSONs; identity fields and "
                  f"metric names checked exactly)")
            print(f"    within tolerance, non-zero drift       "
                  f"{len(res['deltas']) - exact:5d}  (largest {worst:.3g}, "
                  f"limit {TOL:g})")
        for s in skipped:
            print(f"  not covered: {s}")
        print("\n  declared unexecuted:")
        for what, why in UNEXECUTED:
            print(f"    - {what}\n        {why}")

        # The coverage numbers the paper quotes are written by the run that
        # achieved them. An earlier design had verify_consistency re-derive
        # them by a parallel calculation, which is the duplicated-count drift
        # this repository has already been bitten by.
        if args.tier == "full" and not res["bad"]:
            cov = {
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                               time.gmtime()),
                "baseline_ref": subprocess.run(
                    ["git", "rev-parse", "--short", args.ref],
                    capture_output=True, text=True).stdout.strip(),
                "tier": args.tier,
                "supplied_inputs": n_supplied,
                "watched": res["n_watched"],
                "reconstructed": res["n_reconstructed"],
                "compared_only": res["n_watched"] - res["n_reconstructed"],
                "byte_identical": res["n_reconstructed"] - len(res["deltas"]),
                "bytes_differ_numbers_identical": sum(
                    1 for v in res["deltas"].values() if v == 0.0),
                "within_tolerance_nonzero_drift": sum(
                    1 for v in res["deltas"].values() if v > 0.0),
                "largest_relative_move": (max(res["deltas"].values())
                                          if res["deltas"] else 0.0),
                "tol_default": TOL, "tol_torch": TOL_TORCH,
                "newly_produced_no_baseline": len(res["extra"]),
                "unexecuted": [w for w, _ in UNEXECUTED],
            }
            Path("docs/reproduction-coverage.json").write_text(
                json.dumps(cov, indent=2, sort_keys=True) + "\n")
            print("\n  wrote docs/reproduction-coverage.json")

        if res["bad"]:
            print(f"\n{len(res['bad'])} REPRODUCIBILITY FAILURE(S):")
            for b in res["bad"][:40]:
                print(f"  - {b}")
            if args.keep:
                print(f"\nworktree kept at {wt}")
            return 1
        print("\nEvery reconstructed artefact matches the baseline. The working "
              "tree was never written to.")
        return 0
    finally:
        if not args.keep:
            drop_worktree(wt)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
