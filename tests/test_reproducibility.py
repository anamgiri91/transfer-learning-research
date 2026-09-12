"""The reproducibility checker, fault-injected.

Passing on clean input is not evidence a checker works -- this repository has
now been bitten four times by a check that matched nothing and passed. Every
test here plants a difference the comparison exists to catch and asserts it is
caught, or plants a difference it must tolerate and asserts it is tolerated
*and reported* rather than silently absorbed.

The tolerance tests are the load-bearing ones. A tolerance nothing can violate
is not a tolerance, so each is probed from both sides: just over, and just
under.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "scripts")
_spec = importlib.util.spec_from_file_location(
    "verify_reproducibility", "scripts/verify_reproducibility.py")
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)

METRIC = {"arm": "B1_ecfp_histgb", "split": "scaffold", "seed": 0, "n_train": 347,
          "metrics": {"rmse": 0.6031, "r2": 0.5068}, "seconds": 2.2}


def _trees(tmp_path, files: dict[str, str | bytes]):
    """A baseline tree and a live tree, identical to start with."""
    base, live = tmp_path / "base", tmp_path / "live"
    for d in (base, live):
        d.mkdir()
        for name, content in files.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(content)
    return base, live


def _watch(monkeypatch, pats):
    monkeypatch.setattr(vr, "WATCHED", pats)


def _npz(path: Path, keys, y_true, y_pred):
    np.savez(path, inchikey=np.array(keys), y_true=np.array(y_true, float),
             y_pred=np.array(y_pred, float))


# --------------------------------------------------------------------------
# Differences that must be caught
# --------------------------------------------------------------------------

def test_a_metric_moving_beyond_tolerance_is_caught(tmp_path, monkeypatch):
    moved = json.loads(json.dumps(METRIC))
    moved["metrics"]["rmse"] += 1e-6
    base, live = _trees(tmp_path, {"m.json": json.dumps(METRIC)})
    (live / "m.json").write_text(json.dumps(moved))
    _watch(monkeypatch, ["*.json"])
    r = vr.compare(base, live, {"m.json"})
    assert any("over its" in b for b in r["bad"]), r


def test_a_table_value_moving_beyond_tolerance_is_caught(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {"t.csv": "# generated\na,b\n1.0,295.05781384\n"})
    (live / "t.csv").write_text("# generated\na,b\n1.0,295.05781684\n")
    _watch(monkeypatch, ["*.csv"])
    r = vr.compare(base, live, {"t.csv"})
    assert any("over its" in b for b in r["bad"]), r


def test_a_vanished_artefact_is_caught(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {"t.csv": "# generated\na\n1\n"})
    (live / "t.csv").unlink()
    _watch(monkeypatch, ["*.csv"])
    assert any("DISAPPEARED" in b for b in vr.compare(base, live, set())["bad"])


def test_an_uncommitted_new_artefact_is_caught(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {"t.csv": "# generated\na\n1\n"})
    (live / "extra.csv").write_text("# generated\na\n9\n")
    _watch(monkeypatch, ["*.csv"])
    assert any("NEW FILE" in b for b in vr.compare(base, live, set())["bad"])


def test_a_verdict_string_flipping_is_never_waved_through(tmp_path, monkeypatch):
    base, live = _trees(
        tmp_path, {"t.csv": "# generated\narm,verdict\nT1,inconclusive\n"})
    (live / "t.csv").write_text("# generated\narm,verdict\nT1,beats B1\n")
    _watch(monkeypatch, ["*.csv"])
    r = vr.compare(base, live, {"t.csv"})
    assert any("non-numeric" in b or "STRUCTURAL" in b for b in r["bad"]), r


# --------------------------------------------------------------------------
# Structural invariants: exact, and never absorbed by a numeric tolerance
# --------------------------------------------------------------------------

def test_changed_row_identities_fail_even_when_the_numbers_are_identical(tmp_path, monkeypatch):
    """The test fold changing membership is a different experiment.

    The predictions are bit-identical here; only the inchikeys move. A checker
    that compared floats alone would pass this, and it must not.
    """
    base, live = _trees(tmp_path, {})
    _npz(base / "p.npz", ["AAA", "BBB", "CCC"], [1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
    _npz(live / "p.npz", ["AAA", "BBB", "ZZZ"], [1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
    _watch(monkeypatch, ["*.npz"])
    r = vr.compare(base, live, {"p.npz"})
    assert any("row identities differ" in b for b in r["bad"]), r


def test_a_changed_shape_fails(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {})
    _npz(base / "p.npz", ["A", "B", "C"], [1.0, 2.0, 3.0], [1.1, 2.1, 3.1])
    _npz(live / "p.npz", ["A", "B"], [1.0, 2.0], [1.1, 2.1])
    _watch(monkeypatch, ["*.npz"])
    assert any("shape" in b for b in vr.compare(base, live, {"p.npz"})["bad"])


def test_a_changed_missingness_pattern_fails(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {})
    _npz(base / "p.npz", ["A", "B"], [1.0, 2.0], [1.1, 2.1])
    _npz(live / "p.npz", ["A", "B"], [1.0, 2.0], [1.1, np.nan])
    _watch(monkeypatch, ["*.npz"])
    assert any("missingness" in b for b in vr.compare(base, live, {"p.npz"})["bad"])


def test_a_changed_table_shape_fails(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {"t.csv": "# generated\na,b\n1,2\n3,4\n"})
    (live / "t.csv").write_text("# generated\na,b\n1,2\n")
    _watch(monkeypatch, ["*.csv"])
    assert any("shape" in b for b in vr.compare(base, live, {"t.csv"})["bad"])


# --------------------------------------------------------------------------
# Differences that must be tolerated -- and reported, not hidden
# --------------------------------------------------------------------------

def test_a_provenance_timestamp_is_not_a_difference(tmp_path, monkeypatch):
    base, live = _trees(tmp_path, {"t.csv": "# generated on 2026-01-01\na\n1\n"})
    (live / "t.csv").write_text("# generated on 2026-09-11\na\n1\n")
    _watch(monkeypatch, ["*.csv"])
    r = vr.compare(base, live, {"t.csv"})
    assert not r["bad"] and not r["deltas"]


def test_sub_tolerance_float_drift_passes_and_is_reported(tmp_path, monkeypatch):
    """The measured n_jobs=-1 / interpolation drift. Must pass, must be visible."""
    base, live = _trees(
        tmp_path, {"t.csv": "# generated\na,b\n1.0,295.057813849438\n"})
    (live / "t.csv").write_text("# generated\na,b\n1.0,295.05781384943833\n")
    _watch(monkeypatch, ["*.csv"])
    r = vr.compare(base, live, {"t.csv"})
    assert not r["bad"]
    assert r["deltas"] and max(r["deltas"].values()) < vr.TOL


def test_the_torch_tolerance_is_looser_and_applies_only_to_torch_artifacts(tmp_path, monkeypatch):
    """A 1e-7 move passes for a fine-tune arm and fails for a sklearn arm.

    Same magnitude, different verdict, because the two arms have different
    declared tolerances and the file name selects between them.
    """
    sk = json.loads(json.dumps(METRIC))
    tt = json.loads(json.dumps(METRIC)); tt["arm"] = "T2_chemberta_full_finetune"
    base, live = _trees(tmp_path, {
        "B1_ecfp_histgb__s0.json": json.dumps(sk),
        "T2_chemberta_full_finetune__s0.json": json.dumps(tt)})
    for name, rec in (("B1_ecfp_histgb__s0.json", sk),
                      ("T2_chemberta_full_finetune__s0.json", tt)):
        moved = json.loads(json.dumps(rec))
        moved["metrics"]["rmse"] += 1e-7
        (live / name).write_text(json.dumps(moved))
    _watch(monkeypatch, ["*.json"])
    r = vr.compare(base, live, {"B1_ecfp_histgb__s0.json",
                                "T2_chemberta_full_finetune__s0.json"})
    assert any("B1_ecfp_histgb" in b for b in r["bad"]), r["bad"]
    assert not any("T2_chemberta" in b for b in r["bad"]), r["bad"]


def test_the_torch_tolerance_still_binds(tmp_path, monkeypatch):
    """Looser is not unlimited: 1e-4 must fail even for a fine-tune arm."""
    tt = json.loads(json.dumps(METRIC)); tt["arm"] = "T2_chemberta_full_finetune"
    base, live = _trees(tmp_path, {"T2_chemberta_full_finetune__s0.json": json.dumps(tt)})
    moved = json.loads(json.dumps(tt)); moved["metrics"]["rmse"] += 1e-4
    (live / "T2_chemberta_full_finetune__s0.json").write_text(json.dumps(moved))
    _watch(monkeypatch, ["*.json"])
    assert any("over its" in b for b in vr.compare(
        base, live, {"T2_chemberta_full_finetune__s0.json"})["bad"])


# --------------------------------------------------------------------------
# Coverage accounting
# --------------------------------------------------------------------------

def test_untouched_files_are_not_counted_as_reconstructed(tmp_path, monkeypatch):
    """The defect this accounting was added for.

    run_arms.py skips cells whose metric file exists, so the one stage that
    re-fits a model once exited in 8s having fitted nothing and reported ok.
    """
    base, live = _trees(tmp_path, {"a.csv": "# generated\na\n1\n",
                                   "b.csv": "# generated\na\n2\n"})
    _watch(monkeypatch, ["*.csv"])
    r = vr.compare(base, live, {"a.csv"})
    assert r["n_watched"] == 2 and r["n_reconstructed"] == 1


def test_every_stage_names_a_script_that_exists():
    for label, argv, _net, _tier, _clears in vr.STAGES:
        assert Path(argv[0]).exists(), f"{label}: {argv[0]} missing"


def test_every_clears_glob_matches_something():
    """A `clears` pattern matching nothing means the stage is skipping again."""
    for label, _argv, _net, _tier, clears in vr.STAGES:
        if clears:
            assert list(vr.ROOT.glob(clears)), f"{label}: {clears!r} matches nothing"


def test_the_unexecuted_list_is_not_empty_and_carries_reasons():
    """Coverage claims must name what is NOT covered, with a reason each."""
    assert vr.UNEXECUTED
    for what, why in vr.UNEXECUTED:
        assert what and why and len(why) > 20


def test_supplied_inputs_are_all_in_the_manifest():
    """Including the six encoders: 12 model files plus the raw payloads."""
    bad = vr.check_inputs()
    assert not bad, bad
    man = json.loads(vr.INPUT_MANIFEST.read_text())["files"]
    assert sum(1 for k in man if k.startswith("models/") and k.endswith(".pt")) == 6
    assert all(len(v["sha256"]) == 64 for v in man.values())


def test_a_tampered_supplied_input_is_detected(tmp_path, monkeypatch):
    """Checksums must reject a changed input, not just note its absence."""
    real = json.loads(vr.INPUT_MANIFEST.read_text())
    victim = next(k for k in real["files"] if k.endswith(".json"))
    fake = tmp_path / "m.json"
    tampered = json.loads(json.dumps(real))
    tampered["files"][victim]["sha256"] = "0" * 64
    fake.write_text(json.dumps(tampered))
    monkeypatch.setattr(vr, "INPUT_MANIFEST", fake)
    assert any("CHANGED" in b for b in vr.check_inputs())


def test_NaN_compared_with_NaN_is_not_an_infinite_difference(tmp_path, monkeypatch):
    """B0's Spearman is NaN in all 120 of its metric files.

    Comparing NaN with `==` made every one look infinitely different from
    itself; a clean re-fit reported 19 false failures before this was fixed.
    """
    rec = {"arm": "B0_median", "metrics": {"rmse": 0.85, "spearman": None}}
    base, live = _trees(tmp_path, {"m.json": json.dumps(rec)})
    (live / "m.json").write_text(json.dumps(rec))
    _watch(monkeypatch, ["*.json"])
    assert not vr.compare(base, live, {"m.json"})["bad"]
    assert vr._rel(float("nan"), float("nan")) == 0.0
    assert vr._rel(float("nan"), 1.0) == float("inf")   # still a real difference


def test_an_untracked_artefact_has_no_baseline_and_is_not_a_failure(tmp_path, monkeypatch):
    """Figures are gitignored and predictions were committed for one split only.

    Neither can be diffed against a commit that does not contain them. They are
    reported as reconstructed-without-a-baseline, which is weaker than verified
    and honest about being weaker.
    """
    base, live = _trees(tmp_path, {})
    for d in (base, live):
        (d / "results/predictions").mkdir(parents=True)
    _npz(live / "results/predictions/new.npz", ["A"], [1.0], [1.1])
    _watch(monkeypatch, ["results/predictions/*.npz"])
    r = vr.compare(base, live, set(), tracked=set())     # nothing tracked
    assert not r["bad"] and r["extra"] == ["results/predictions/new.npz"]


def test_a_TRACKED_file_missing_from_the_baseline_is_still_a_failure(tmp_path, monkeypatch):
    """The exemption keys on tracked-ness, and must not become a blanket one."""
    base, live = _trees(tmp_path, {})
    for d in (base, live):
        (d / "results/tables").mkdir(parents=True)
    (live / "results/tables/rogue.csv").write_text("# generated\na\n1\n")
    _watch(monkeypatch, ["results/tables/*.csv"])
    r = vr.compare(base, live, set(), tracked={"results/tables/rogue.csv"})
    assert any("NEW FILE" in b for b in r["bad"]), r


def test_a_metric_file_with_matching_numbers_but_a_changed_n_train_fails(tmp_path, monkeypatch):
    """The same result attached to a different experiment must not pass.

    numeric_delta only inspects the `metrics` dict, so identity fields need
    their own exact check or a changed n_train reads as harmless byte drift.
    """
    a = json.loads(json.dumps(METRIC))
    b = json.loads(json.dumps(METRIC)); b["n_train"] = 250
    base, live = _trees(tmp_path, {"m.json": json.dumps(a)})
    (live / "m.json").write_text(json.dumps(b))
    _watch(monkeypatch, ["*.json"])
    r = vr.compare(base, live, {"m.json"}, tracked={"m.json"})
    assert any("identity field 'n_train'" in x for x in r["bad"]), r


def test_a_changed_split_label_fails(tmp_path, monkeypatch):
    a = json.loads(json.dumps(METRIC))
    b = json.loads(json.dumps(METRIC)); b["split"] = "random"
    base, live = _trees(tmp_path, {"m.json": json.dumps(a)})
    (live / "m.json").write_text(json.dumps(b))
    _watch(monkeypatch, ["*.json"])
    assert any("identity field 'split'" in x
               for x in vr.compare(base, live, {"m.json"}, tracked={"m.json"})["bad"])


def test_a_wall_clock_change_alone_is_not_a_failure(tmp_path, monkeypatch):
    """`seconds` moves on every run and is not part of what the file claims."""
    a = json.loads(json.dumps(METRIC))
    b = json.loads(json.dumps(METRIC)); b["seconds"] = 99.9
    base, live = _trees(tmp_path, {"m.json": json.dumps(a)})
    (live / "m.json").write_text(json.dumps(b))
    _watch(monkeypatch, ["*.json"])
    r = vr.compare(base, live, {"m.json"}, tracked={"m.json"})
    assert not r["bad"] and r["deltas"]["m.json"] == 0.0


def test_a_dropped_metric_is_caught(tmp_path, monkeypatch):
    a = json.loads(json.dumps(METRIC))
    b = json.loads(json.dumps(METRIC)); del b["metrics"]["r2"]
    base, live = _trees(tmp_path, {"m.json": json.dumps(a)})
    (live / "m.json").write_text(json.dumps(b))
    _watch(monkeypatch, ["*.json"])
    assert any("metric names differ" in x
               for x in vr.compare(base, live, {"m.json"}, tracked={"m.json"})["bad"])
