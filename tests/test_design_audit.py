"""The design diagnostics must report the properties the audit found, or fail."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_design  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return audit_design.build_report()


def test_reserved_validation_fold_reaches_no_executed_arm(report):
    used = report["data_actually_used"]
    assert used["curated_compounds"] == 494
    assert used["validation_fold"] == 49
    assert used["compounds_reaching_any_executed_arm"] == 445
    # The runners that produced every published metric read train and test only.
    for runner in ("scripts/run_arms.py", "scripts/run_finetune.py",
                   "scripts/run_dmpnn.py"):
        assert runner not in used["validation_fold_read_by"]


def test_seeds_are_reported_as_overlapping_rather_than_independent(report):
    for split, row in report["seed_dependence"]["per_split"].items():
        assert row["mean_pairwise_test_overlap"] > 0, split
        assert row["max_test_folds_one_compound_appears_in"] > 1, split
        assert row["compounds_never_in_any_test_fold"] > 0, split


def test_the_least_diverse_test_fold_is_named(report):
    fc = report["test_fold_composition"]
    lo, hi = fc["n_test_scaffolds_range"]
    assert lo < hi, "a single fold composition would make the seeds exchangeable"
    assert fc["least_diverse_seeds"], "the outlying fold must be named"


def test_leave_one_seed_out_covers_the_headline_contrasts(report):
    names = {c["contrast"] for c in report["leave_one_seed_out"]}
    assert {"T1 vs B1", "T2v vs B1", "B3 vs B1"} <= names
    for c in report["leave_one_seed_out"]:
        lo, hi = c["median_paired_delta_range"]
        assert lo <= c["full_median_paired_delta"] <= hi, c["contrast"]
        assert len(c["leave_one_out"]) == len(audit_design.SEEDS)


def test_check_mode_rejects_a_stale_report(tmp_path, monkeypatch):
    stale = tmp_path / "design.json"
    stale.write_text('{"stale": true}\n')
    monkeypatch.setattr(audit_design, "OUT", stale)
    monkeypatch.setattr(sys, "argv", ["audit", "--check"])
    assert audit_design.main() == 1
    assert json.loads(stale.read_text()) == {"stale": True}
