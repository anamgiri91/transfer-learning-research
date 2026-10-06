#!/usr/bin/env python
"""Independently refit Amendment 8's fixed cells without changing any baseline.

Run with the desired environment's interpreter. Sources, processed inputs and
the existing pinned baseline are copied before any training; subprocesses run
in that temporary snapshot. Historical results are never used as new baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B3_TAGS = [f"B3__scaffold__seed{s}__n{n}"
           for s in range(10) for n in (50, 100, 250, 347)]
T2V_TAG = "T2v__scaffold__seed0__n50"


def hashes(root, paths):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def compare(reference: Path, rerun: Path, tag: str) -> dict:
    a = json.loads((reference / "metrics" / f"{tag}.json").read_text())
    b = json.loads((rerun / "metrics" / f"{tag}.json").read_text())
    a.pop("seconds", None)
    b.pop("seconds", None)
    with np.load(reference / "predictions" / f"{tag}.npz", allow_pickle=True) as za, \
            np.load(rerun / "predictions" / f"{tag}.npz", allow_pickle=True) as zb:
        same_keys = set(za.files) == set(zb.files) == {"inchikey", "y_true", "y_pred"}
        arrays = {k: bool(k in za and k in zb and za[k].dtype == zb[k].dtype
                         and np.array_equal(za[k], zb[k]))
                  for k in ("inchikey", "y_true", "y_pred")}
        drift = (float(np.max(np.abs(za["y_pred"] - zb["y_pred"])))
                 if za["y_pred"].shape == zb["y_pred"].shape else None)
    return dict(tag=tag, record_identical_excluding_seconds=a == b,
                arrays_identical=arrays, max_prediction_absolute_drift=drift,
                passed=bool(a == b and same_keys and all(arrays.values())))


def check_report(path: Path) -> bool:
    """Reject missing, stale or partial evidence without retraining."""
    report = json.loads(path.read_text())
    expected = set(B3_TAGS + [T2V_TAG])
    rows = report["cells"]
    if (not report["passed"] or not report["protected_artifacts_unchanged"]
            or len(rows) != len(expected) or {r["tag"] for r in rows} != expected
            or not all(r["passed"] and r["record_identical_excluding_seconds"]
                       and set(r["arrays_identical"]) == {"inchikey", "y_true", "y_pred"}
                       and all(r["arrays_identical"].values())
                       and r["max_prediction_absolute_drift"] == 0 for r in rows)):
        raise ValueError("repeatability evidence is incomplete or failed")
    protected = report["protected_artifact_sha256"]
    if hashes(ROOT, [ROOT / p for p in protected]) != protected:
        raise ValueError("historical or pinned artifacts changed since repeatability check")
    # Analysis and documentation can evolve without invalidating a fit; its
    # training scripts, seeding, checkpoint identity and processed inputs cannot.
    relevant = {p: h for p, h in report["snapshot_sha256"].items()
                if p.startswith("data/processed/") or p in (
                    "scripts/run_dmpnn.py", "scripts/run_finetune.py",
                    "src/evapro/utils/seeding.py", "src/evapro/models/pretrained.py",
                    "src/evapro/data/io.py", "src/evapro/data/splits.py")}
    if hashes(ROOT, [ROOT / p for p in relevant]) != relevant:
        raise ValueError("training source or inputs changed since repeatability check")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "docs/numerical-repeatability.json")
    ap.add_argument("--check", action="store_true", help="validate existing evidence only")
    args = ap.parse_args()
    if args.check:
        check_report(args.output)
        print(f"Verified {args.output} against protected artifacts and training inputs")
        return 0
    pinned = ROOT / "results/sensitivity/b3_threads1"
    protected = [p for directory in ("results/metrics", "results/metrics_ft",
                 "results/metrics_b3", "results/predictions", "results/predictions_ft",
                 "results/predictions_b3", "results/sensitivity/b3_threads1")
                 for p in (ROOT / directory).rglob("*") if p.is_file()]
    before = hashes(ROOT, protected)
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="40 existing pinned B3 cells; two independent T2v seed0 n50 fits",
                  historical_full_reproduction_passed=False,
                  comparison="exact arrays and full records except seconds",
                  protected_artifact_sha256=before, cells=[], commands=[], passed=False)
    with tempfile.TemporaryDirectory(prefix="evapro-repeatability-") as temp:
        work = Path(temp)
        for directory in ("src", "scripts", "data/processed"):
            shutil.copytree(ROOT / directory, work / directory,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(pinned, work / "reference")
        sources = [p for directory in ("src", "scripts", "data/processed")
                   for p in (work / directory).rglob("*") if p.is_file()]
        report["snapshot_sha256"] = hashes(work, sources)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(work / "src"), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1")
        info = subprocess.check_output([sys.executable, "-c",
            "import json,platform,sys,torch,importlib.metadata as m; "
            "print(json.dumps(dict(python=sys.version,platform=platform.platform(),"
            "packages={k:m.version(k) for k in ['numpy','scipy','torch','chemprop',"
            "'lightning','transformers','rdkit']},torch_config=torch.__config__.show())))"],
            cwd=work, env=env, text=True)
        report["environment"] = json.loads(info)
        report["environment"]["interpreter"] = sys.executable
        report["environment"]["thread_environment"] = {
            k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
        try:
            jobs = [("scripts/run_dmpnn.py", "b3", []),
                    ("scripts/run_finetune.py", "t2v_first", ["--arms", "T2v",
                     "--seeds", "0", "--sizes", "50"]),
                    ("scripts/run_finetune.py", "t2v_second", ["--arms", "T2v",
                     "--seeds", "0", "--sizes", "50"])]
            for script, dest, extra in jobs:
                cmd = [sys.executable, script, "--threads", "1", "--out-root", dest, *extra]
                report["commands"].append(cmd)
                print(f"Running {script} -> {dest}", flush=True)
                with (work / f"{dest}.log").open("w") as log:
                    result = subprocess.run(cmd, cwd=work, env=env, stdout=log,
                                            stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError((work / f"{dest}.log").read_text()[-4000:])
                if dest == "b3":
                    report["cells"].extend(compare(work / "reference", work / dest, t)
                                           for t in B3_TAGS)
                    print(f"B3: {sum(c['passed'] for c in report['cells'])}/40 exact", flush=True)
            report["cells"].append(compare(work / "t2v_first", work / "t2v_second", T2V_TAG))
            report["t2v_pinned_metrics"] = json.loads(
                (work / "t2v_first/metrics" / f"{T2V_TAG}.json").read_text())["metrics"]
            report["protected_artifacts_unchanged"] = before == hashes(ROOT, protected)
            report["passed"] = (report["protected_artifacts_unchanged"]
                                and len(report["cells"]) == 41
                                and all(c["passed"] for c in report["cells"]))
        except Exception as exc:
            report["error"] = str(exc)
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
