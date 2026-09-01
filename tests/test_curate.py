"""Curation gate (plan.md §3.2) -- the 'high-fidelity' claim in the title."""
import pandas as pd
import pytest

pytest.importorskip("rdkit", reason="curation needs RDKit")

from evapro.data.curate import curate, to_pactivity  # noqa: E402


def _row(**kw):
    # N-benzylbenzamide, 211 Da -- clears the 150 Da fidelity floor.
    base = dict(smiles="O=C(NCc1ccccc1)c1ccccc1",
                standard_type="IC50", standard_relation="=",
                standard_value_nm=100.0, target_id="EVA71_3C", readout="enzymatic",
                confidence_score=9, doc_year=2020, source="chembl")
    return {**base, **kw}


def test_pactivity_conversion():
    assert to_pactivity(1000.0) == pytest.approx(6.0)
    assert to_pactivity(1.0) == pytest.approx(9.0)


def test_censored_values_are_held_out_not_dropped_silently():
    df = pd.DataFrame([_row(), _row(standard_relation=">", smiles="O=C(NCc1ccccc1)c1ccccc1O")])
    held = []
    out, rep = curate(df, censored_out=held)
    assert rep.n_censored_held_out == 1
    assert len(held[0]) == 1
    assert len(out) == 1


def test_replicate_disagreement_beyond_one_log_is_discarded():
    # Same compound, IC50 100 nM vs 100000 nM -> 3 log units apart.
    df = pd.DataFrame([_row(standard_value_nm=100.0), _row(standard_value_nm=100_000.0)])
    out, rep = curate(df)
    assert len(out) == 0
    assert rep.dropped["replicate_disagreement"] == 1


def test_consistent_replicates_collapse_to_median():
    df = pd.DataFrame([_row(standard_value_nm=100.0), _row(standard_value_nm=200.0)])
    out, _ = curate(df)
    assert len(out) == 1
    assert out["pactivity"].iloc[0] == pytest.approx(
        (to_pactivity(100.0) + to_pactivity(200.0)) / 2)


def test_molecules_below_the_mw_floor_are_rejected():
    # Guards the fidelity gate itself: N-methylbenzamide (135 Da) must not pass.
    df = pd.DataFrame([_row(smiles="c1ccccc1C(=O)NC")])
    out, rep = curate(df)
    assert len(out) == 0 and rep.dropped["mw_out_of_range"] == 1


def test_unparseable_structures_are_rejected():
    df = pd.DataFrame([_row(smiles="not_a_molecule")])
    out, rep = curate(df)
    assert len(out) == 0 and rep.dropped["unparseable_structure"] == 1


def test_enzymatic_and_cellular_readouts_are_never_pooled():
    df = pd.DataFrame([_row(readout="enzymatic", standard_value_nm=100.0),
                       _row(readout="cellular", standard_value_nm=100_000.0)])
    out, _ = curate(df)
    assert len(out) == 2, "different readouts must stay separate rows"
