"""Keep repeated structures separate from evidence of repeat assays."""
import pandas as pd
import pytest

from evapro.data.openbind import label_audit, make_master, verify_archive


def test_repeated_complex_labels_are_not_assay_replication():
    raw = pd.DataFrame({
        "complex_name": ["a", "b", "c", "d"],
        "compound_group": ["same", "same", "single", "missing"],
        "smiles": ["CC", "CC", "CCC", "C"],
        "experimental_pKD": [5.0, 5.0, 6.0, float("nan")],
        "suspected_artefact": [False] * 4,
        "pb_valid_prepared": [True] * 4,
    })
    master = make_master(raw)
    audit = label_audit(raw, master)
    assert master["Complex"].tolist() == ["a", "b", "c"]
    assert audit["multiple_complex_identical_labels"] == 1
    assert audit["single_complex_groups"] == 1
    assert audit["max_within_compound_label_spread"] == 0
    assert audit["independent_assay_replicates_identifiable"] is False


def test_changed_source_bytes_are_rejected(tmp_path):
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"not the recorded release")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(archive)
