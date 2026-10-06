"""Adversarial checks for Amendment 8 evidence, without fitting any models."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_numerical_sensitivity as sensitivity
from verify_numerical_repeatability import compare
import verify_numerical_repeatability as repeatability
from evapro.evaluation.metrics import precision_at_k_frac_bounds


def test_bounds_equal_exhaustive_cutoff_memberships():
    # k=3: one certain hit; choose two of four tied rows (two hits, two misses).
    y = np.array([10, 9, 8, 1, 0, 2], dtype=float)
    pred = np.array([5, 3, 3, 3, 3, 0], dtype=float)
    scores = [(1 + len(set(pick) & {1, 2})) / 3
              for pick in itertools.combinations([1, 2, 3, 4], 2)]
    assert precision_at_k_frac_bounds(y, pred, frac=0.5) == (min(scores), max(scores))


@pytest.mark.parametrize("y,pred,kwargs", [
    ([], [], {}), ([1], [1, 2], {}), ([1, 2], [1, np.nan], {}),
    ([1, np.inf], [1, 2], {}), ([[1]], [[1]], {}),
    ([1], [1], {"eps": -1}), ([1], [1], {"eps": np.nan}),
    ([1], [1], {"frac": 0}), ([1], [1], {"frac": 1.1}),
])
def test_bounds_reject_invalid_evidence(y, pred, kwargs):
    with pytest.raises(ValueError):
        precision_at_k_frac_bounds(y, pred, **kwargs)


def test_p_value_extrema_include_mixed_seed_assignments(monkeypatch):
    monkeypatch.setattr(sensitivity, "SEEDS", list(range(6)))
    arm = dict.fromkeys(range(6), (0.4, 0.4, 0.6))
    ref = dict.fromkeys(range(6), (0.5, 0.5, 0.5))
    arms = [("example", arm, ref)]
    corners = sensitivity.scenario_tests(arms)
    result = sensitivity.exhaustive_bounds(arms, dict.fromkeys(range(6), 10))[0]
    assert result["configurations"] == 3 ** 6
    assert result["p_raw_max"] == 1
    assert result["p_raw_max"] > max(r["p_raw"] for r in corners)
    assert not result["all_configurations_robust"]


def test_missing_pinned_cells_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(sensitivity, "PINNED", tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        sensitivity.analysis_a()


def test_sensitivity_check_rejects_stale_report_without_replacing_it(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text('{"old": true}\n')
    monkeypatch.setattr(sensitivity, "OUT", report)
    monkeypatch.setattr(sensitivity, "build_report", lambda: {"new": True})
    monkeypatch.setattr(sys, "argv", ["analysis", "--check"])
    assert sensitivity.main() == 1
    assert report.read_text() == '{"old": true}\n'


def test_repeatability_check_rejects_partial_and_stale_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(repeatability, "ROOT", tmp_path)
    protected = tmp_path / "baseline.json"
    protected.write_text("immutable baseline")
    rows = [dict(tag=t, passed=True, record_identical_excluding_seconds=True,
                 arrays_identical=dict(inchikey=True, y_true=True, y_pred=True),
                 max_prediction_absolute_drift=0.)
            for t in repeatability.B3_TAGS + [repeatability.T2V_TAG]]
    data = dict(passed=True, protected_artifacts_unchanged=True, cells=rows,
                protected_artifact_sha256=repeatability.hashes(tmp_path, [protected]),
                snapshot_sha256={})
    report = tmp_path / "report.json"
    report.write_text(json.dumps(data))
    assert repeatability.check_report(report)
    data["cells"] = rows[:-1]
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        repeatability.check_report(report)
    data["cells"] = rows
    report.write_text(json.dumps(data))
    protected.write_text("silently replaced")
    with pytest.raises(ValueError, match="artifacts changed"):
        repeatability.check_report(report)


def test_pinned_prediction_identity_is_checked(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(sensitivity.PINNED, tmp_path / "pinned")
    monkeypatch.setattr(sensitivity, "PINNED", tmp_path / "pinned")
    path = sensitivity.PINNED / "predictions/B3__scaffold__seed0__n50.npz"
    with np.load(path, allow_pickle=True) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["inchikey"] = arrays["inchikey"][::-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="identity"):
        sensitivity.analysis_a()


def test_repeatability_rejects_shape_and_selection_changes(tmp_path):
    for directory in ("a", "b"):
        root = tmp_path / directory
        (root / "metrics").mkdir(parents=True)
        (root / "predictions").mkdir()
        (root / "metrics/cell.json").write_text(json.dumps(
            {"metrics": {"rmse": 0.1}, "selected_lr": 0.001, "seconds": len(directory)}))
        np.savez(root / "predictions/cell.npz", inchikey=np.array(["a", "b"]),
                 y_true=np.array([1., 2.]), y_pred=np.array([1., 2.]))
    assert compare(tmp_path / "a", tmp_path / "b", "cell")["passed"]
    np.savez(tmp_path / "b/predictions/cell.npz", inchikey=np.array(["a", "b"]),
             y_true=np.array([1., 2.]), y_pred=np.array([1.]))
    assert not compare(tmp_path / "a", tmp_path / "b", "cell")["passed"]
    (tmp_path / "b/metrics/cell.json").write_text(json.dumps(
        {"metrics": {"rmse": 0.1}, "selected_lr": 0.01}))
    assert not compare(tmp_path / "a", tmp_path / "b", "cell")["record_identical_excluding_seconds"]
