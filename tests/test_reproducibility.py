"""Tests for the reproducibility checker.

Passing on clean input is not evidence that a checker works -- the 2026-09-02
fault-injection round found one of eight checkers that exited 0 on a corrupted
split file. So each test here plants the specific difference the checker exists
to catch, and one asserts it does *not* fire on the provenance timestamps that
change on every single run.
"""
import json

import pytest

from scripts import verify_reproducibility as vr


def _snapshot(tmp_path, files: dict[str, str]):
    """Build a snapshot dir + a live tree, and point the checker at them."""
    snap, live = tmp_path / "snap", tmp_path / "live"
    for name, body in files.items():
        for base in (snap, live):
            p = base / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
    return snap, live


def _point_at(monkeypatch, live, patterns):
    monkeypatch.setattr(vr, "ROOT", live)
    monkeypatch.setattr(vr, "WATCHED", patterns)


METRIC = {"arm": "B1_ecfp_histgb", "split": "scaffold", "seed": 0,
          "metrics": {"rmse": 0.651358230421361, "spearman": 0.6790546141352592}}


def test_a_csv_provenance_timestamp_is_not_a_difference(tmp_path, monkeypatch):
    """Every table records when it was generated. Comparing that would make the
    check fail on every run, which is how a check gets switched off."""
    snap, live = _snapshot(tmp_path, {"t.csv": "# generated on 2026-09-02\na,b\n1,2\n"})
    (live / "t.csv").write_text("# generated on 2026-09-06\na,b\n1,2\n")
    _point_at(monkeypatch, live, ["*.csv"])
    bad, _, n, _ = vr.compare(snap, set())
    assert bad == [] and n == 1


def test_a_json_generated_utc_is_not_a_difference(tmp_path, monkeypatch):
    """indomain_3c.curation.json carries the same timestamp inside the JSON."""
    a = {"n_compounds": 2743, "meta": {"generated_utc": "2026-09-02T17:20:51+00:00"}}
    b = {"n_compounds": 2743, "meta": {"generated_utc": "2026-09-06T06:06:10+00:00"}}
    snap, live = _snapshot(tmp_path, {"c.json": json.dumps(a)})
    (live / "c.json").write_text(json.dumps(b))
    _point_at(monkeypatch, live, ["*.json"])
    bad, _, _, _ = vr.compare(snap, set())
    assert bad == [], bad


def test_a_changed_data_row_is_caught(tmp_path, monkeypatch):
    """A moved number in a derived table must fail, however it is reported.

    Derived CSVs are compared within a relative tolerance (a 1e-16 wobble in an
    RMSE lands at ~1e-13 in table2's interpolated training-set size), so this
    is caught as an over-tolerance numeric move rather than as raw byte
    inequality. Either message is a failure; silence would not be.
    """
    snap, live = _snapshot(tmp_path, {"t.csv": "# generated\na,b\n1,2\n"})
    (live / "t.csv").write_text("# generated\na,b\n1,3\n")
    _point_at(monkeypatch, live, ["*.csv"])
    bad, _, _, _ = vr.compare(snap, set())
    assert any("content differs" in b or "moved by" in b for b in bad), bad


def test_a_csv_moving_beyond_tolerance_is_caught(tmp_path, monkeypatch):
    """The tolerance must bind: a relative move over 1e-9 in a table fails."""
    snap, live = _snapshot(tmp_path, {"t.csv": "# generated\na,b\n1.0,295.05781384\n"})
    (live / "t.csv").write_text("# generated\na,b\n1.0,295.05781684\n")
    _point_at(monkeypatch, live, ["*.csv"])
    bad, _, _, _ = vr.compare(snap, set())
    assert any("moved by" in b for b in bad), bad


def test_a_csv_moving_within_tolerance_passes_and_is_reported(tmp_path, monkeypatch):
    """...and must not bind so tight that thread scheduling trips it.

    295.057813849438 -> ...438 33 is the exact move observed when the random
    forest is re-fitted with n_jobs=-1 and the interpolation in table2 amplifies
    it. That has to pass, and has to be visible in `deltas` rather than silent.
    """
    snap, live = _snapshot(
        tmp_path, {"t.csv": "# generated\na,b\n1.0,295.057813849438\n"})
    (live / "t.csv").write_text("# generated\na,b\n1.0,295.05781384943833\n")
    _point_at(monkeypatch, live, ["*.csv"])
    bad, deltas, _, _ = vr.compare(snap, set())
    assert not bad, bad
    assert deltas and max(deltas.values()) < vr.TOL


def test_a_non_numeric_csv_change_is_never_waved_through(tmp_path, monkeypatch):
    """A verdict string flipping is not a floating-point wobble."""
    snap, live = _snapshot(
        tmp_path, {"t.csv": "# generated\narm,verdict\nT1,inconclusive\n"})
    (live / "t.csv").write_text("# generated\narm,verdict\nT1,beats B1\n")
    _point_at(monkeypatch, live, ["*.csv"])
    bad, _, _, _ = vr.compare(snap, set())
    assert any("content differs" in b for b in bad), bad


def test_a_stage_that_regenerates_nothing_is_visible(tmp_path, monkeypatch):
    """The defect this accounting was added for.

    `run_arms.py` skips any cell whose metric file exists, so the one stage
    that re-fits a model used to exit in 8s having fitted nothing and report
    [ok]. Untouched files must not be counted as regenerated.
    """
    snap, live = _snapshot(tmp_path, {"a.csv": "# generated\na\n1\n",
                                      "b.csv": "# generated\na\n2\n"})
    _point_at(monkeypatch, live, ["*.csv"])
    _, _, n, n_touched = vr.compare(snap, {live / "a.csv"})
    assert n == 2 and n_touched == 1


def test_a_metric_moving_beyond_tolerance_is_caught(tmp_path, monkeypatch):
    """The failure that matters: a committed number silently moved."""
    moved = json.loads(json.dumps(METRIC))
    moved["metrics"]["rmse"] += 1e-6
    snap, live = _snapshot(tmp_path, {"m.json": json.dumps(METRIC)})
    (live / "m.json").write_text(json.dumps(moved))
    _point_at(monkeypatch, live, ["*.json"])
    bad, deltas, _, _ = vr.compare(snap, set())
    assert any("over the 1e-09 tolerance" in b for b in bad), bad
    assert deltas["m.json"] == pytest.approx(1e-6)


def test_a_metric_moving_within_tolerance_is_reported_but_passes(tmp_path, monkeypatch):
    """The random-forest arm reproduces to ~1e-16, not bit-identically, because
    n_jobs=-1 varies the order of the float reduction (manuscript §10). That is
    below any reported precision, so it is surfaced and not failed."""
    jitter = json.loads(json.dumps(METRIC))
    jitter["metrics"]["rmse"] += 1e-15
    snap, live = _snapshot(tmp_path, {"m.json": json.dumps(METRIC)})
    (live / "m.json").write_text(json.dumps(jitter))
    _point_at(monkeypatch, live, ["*.json"])
    bad, deltas, _, _ = vr.compare(snap, set())
    assert bad == [], bad
    assert 0 < deltas["m.json"] <= vr.TOL


def test_a_disappearing_artefact_is_caught(tmp_path, monkeypatch):
    snap, live = _snapshot(tmp_path, {"t.csv": "# generated\na\n1\n"})
    (live / "t.csv").unlink()
    _point_at(monkeypatch, live, ["*.csv"])
    bad, _, _, _ = vr.compare(snap, set())
    assert any("DISAPPEARED" in b for b in bad), bad


def test_a_new_uncommitted_artefact_is_caught(tmp_path, monkeypatch):
    """A stage that starts writing a file nobody committed is a provenance hole:
    the paper could cite a number that is not in the repository."""
    snap, live = _snapshot(tmp_path, {"t.csv": "# generated\na\n1\n"})
    (live / "extra.csv").write_text("# generated\na\n9\n")
    _point_at(monkeypatch, live, ["*.csv"])
    bad, _, _, _ = vr.compare(snap, set())
    assert any("NEW FILE" in b for b in bad), bad


def test_restore_only_touches_files_it_changed(tmp_path, monkeypatch):
    """restore() runs after the comparison passes, so it can never mask a real
    difference -- but it must also not rewrite files that already match."""
    snap, live = _snapshot(tmp_path, {"a.csv": "# gen 1\nx\n1\n", "b.csv": "# gen 1\nx\n2\n"})
    (live / "a.csv").write_text("# gen 2\nx\n1\n")
    monkeypatch.setattr(vr, "ROOT", live)
    assert vr.restore(snap) == 1
    assert (live / "a.csv").read_text() == "# gen 1\nx\n1\n"


def test_every_stage_names_a_script_that_exists():
    """A stage pointing at a renamed script would skip silently on a network
    error and loudly otherwise; either way the artefact goes unchecked."""
    from pathlib import Path
    for label, argv, _net, _slow, _clears in vr.STAGES:
        assert Path(argv[0]).exists(), f"{label}: {argv[0]} does not exist"
