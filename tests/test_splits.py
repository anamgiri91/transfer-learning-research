"""Split integrity. A leaking split silently inflates every result in the paper,
so these are the most important tests in the repo.
"""
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit", reason="split construction needs RDKit")

from evapro.data.splits import assert_no_leakage, build_split, murcko_scaffold  # noqa: E402

SMILES = [
    "c1ccccc1C(=O)NC", "c1ccccc1C(=O)NCC", "c1ccccc1C(=O)NCCC",
    "C1CCNCC1C(=O)O", "C1CCNCC1C(=O)OC", "C1CCNCC1C(=O)OCC",
    "c1ccncc1CN", "c1ccncc1CNC", "c1ccncc1CNCC", "CCCCCCO",
]


@pytest.fixture
def df():
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    return pd.DataFrame({
        "canonical_smiles": [Chem.MolToSmiles(m) for m in mols],
        "inchikey": [Chem.MolToInchiKey(m) for m in mols],
        "pactivity": [6.0 + i * 0.1 for i in range(len(SMILES))],
        "doc_year": [2010 + i for i in range(len(SMILES))],
    })


@pytest.mark.parametrize("strategy", ["scaffold", "random", "temporal"])
def test_split_assigns_every_molecule_exactly_once(df, strategy):
    folds = build_split(df, strategy, test_frac=0.2, val_frac=0.1,
                        **({} if strategy == "temporal" else {"seed": 0}))
    assert set(folds) == set(df["inchikey"])
    assert set(folds.values()) <= {"train", "val", "test"}


def test_scaffold_split_puts_no_scaffold_in_both_train_and_test(df):
    folds = build_split(df, "scaffold", test_frac=0.2, val_frac=0.1, seed=0)
    assert_no_leakage(df, folds, "scaffold")   # raises on straddling scaffolds


def test_temporal_split_puts_newest_compounds_in_test(df):
    folds = build_split(df, "temporal", test_frac=0.2, val_frac=0.1)
    years = df.set_index("inchikey")["doc_year"]
    test_years = [years[k] for k, v in folds.items() if v == "test"]
    train_years = [years[k] for k, v in folds.items() if v == "train"]
    assert min(test_years) >= max(train_years)


def test_assert_no_leakage_catches_duplicate_compounds(df):
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    folds = build_split(df, "random", seed=0)
    folds[dup["inchikey"].iloc[0]] = "test"
    with pytest.raises(AssertionError, match="duplicate"):
        assert_no_leakage(dup, folds, "random")


def test_unknown_strategy_is_rejected(df):
    with pytest.raises(ValueError, match="unknown split strategy"):
        build_split(df, "shuffle_everything")


# ---------------------------------------------------------------------------
# The COMMITTED split files, not the split-building functions.
#
# Fault injection on 2026-09-02 planted a corruption in
# data/processed/splits/eva71_2a/scaffold__seed0.json and this suite passed:
# every test above builds splits from a synthetic frame and never reads the
# files on disk that every arm actually loads. audit_splits.py described the
# corruption in table 0 and exited 0. These tests close that gap.
# ---------------------------------------------------------------------------

def test_committed_split_files_satisfy_their_invariants():
    import pandas as pd

    from scripts.audit_splits import check_invariants

    aud = pd.read_csv("results/tables/table0_split_audit.csv", comment="#")
    n = len(pd.read_csv("data/processed/eva71_2a.csv"))
    assert check_invariants(aud.to_dict("records"), n) == []


def test_the_invariant_checker_rejects_compound_leakage():
    from scripts.audit_splits import check_invariants
    row = {"split": "scaffold", "seed": 0, "n_train": 347, "n_val": 49, "n_test": 98,
           "compounds_in_train_and_test": 3, "scaffolds_in_train_and_test": 0}
    assert any("both train and test" in b for b in check_invariants([row], 494))


def test_the_invariant_checker_rejects_scaffold_leakage_in_a_scaffold_split():
    from scripts.audit_splits import check_invariants
    row = {"split": "scaffold", "seed": 0, "n_train": 347, "n_val": 49, "n_test": 98,
           "compounds_in_train_and_test": 0, "scaffolds_in_train_and_test": 4}
    assert any("scaffolds in" in b for b in check_invariants([row], 494))


def test_the_invariant_checker_rejects_folds_that_do_not_cover_the_dataset():
    """The exact shape of the planted fault: a compound moved between folds."""
    from scripts.audit_splits import check_invariants
    row = {"split": "scaffold", "seed": 0, "n_train": 346, "n_val": 49, "n_test": 98,
           "compounds_in_train_and_test": 0, "scaffolds_in_train_and_test": 0}
    assert any("folds cover" in b for b in check_invariants([row], 494))


def test_the_invariant_checker_rejects_inconsistent_fold_sizes():
    from scripts.audit_splits import check_invariants
    rows = [{"split": "scaffold", "seed": 0, "n_train": 347, "n_val": 49, "n_test": 98,
             "compounds_in_train_and_test": 0, "scaffolds_in_train_and_test": 0},
            {"split": "random", "seed": 1, "n_train": 340, "n_val": 56, "n_test": 98,
             "compounds_in_train_and_test": 0, "scaffolds_in_train_and_test": 0}]
    assert any("not identical" in b for b in check_invariants(rows, 494))
