"""Tests for the in-domain corpus screen (arms T4/T5).

The assay-description screen decides what counts as a 3C / 3C-like protease
measurement, and it dropped 36% of the fetched records. ChEMBL files
picornaviral proteases under the whole genome polyprotein, so the target id
alone cannot make that call -- these tests pin the cases that matter, using
descriptions taken verbatim from the pulled data.
"""
import pandas as pd
import pytest

from evapro.data.indomain import EXCLUDED, TARGETS, screen


def _screen_one(description: str) -> bool:
    df = pd.DataFrame({"assay_description": [description]})
    kept, _ = screen(df)
    return len(kept) == 1


@pytest.mark.parametrize("desc", [
    "Inhibition of EV71 3C protease expressed in Escherichia coli BL21(DE3)",
    "Inhibition of recombinant Coxsackievirus B3 3C protease expressed in E. coli",
    "SARS-CoV-2 3CL-Pro protease inhibition IC50 determined by FRET",
    "Inhibition of HRV-16 protease 3C expressed in Escherichia coli by FRET assay",
    "MERS_3CL Pro protease inhibition IC50 by FRET",
    "In vitro inhibitory concentration against SARS coronavirus main protease",
])
def test_genuine_3c_assays_are_kept(desc):
    assert _screen_one(desc)


@pytest.mark.parametrize("desc", [
    # The failure this screen exists to prevent: same ChEMBL target, wrong enzyme.
    "Inhibition of Enterovirus 71 Shenzhen/120F1/09 capsid infected in human RD cells",
    "Inhibitory activity against polio virus RNA polymerase",
    "Inhibition of human coronavirus NL63 PLP2 papain-like protease",
    "Inhibition of SARS-CoV-2 PLpro",
    "Inhibition of enterovirus 2A protease",
])
def test_wrong_enzyme_assays_are_dropped(desc):
    assert not _screen_one(desc)


def test_a_description_naming_both_prefers_the_exclusion():
    """'3CL' plus 'papain-like' in one description is ambiguous; drop it.

    Being conservative here costs a few records and protects the corpus from
    silently absorbing the wrong fold."""
    assert not _screen_one("Inhibition of 3CL protease and papain-like protease")


def test_unrelated_description_is_dropped_even_without_a_stop_word():
    assert not _screen_one("Antiviral activity against influenza A virus")


def test_screen_counts_reconcile_with_the_input():
    df = pd.DataFrame({"assay_description": [
        "Inhibition of CVB3 3C protease",          # kept
        "Inhibition of SARS-CoV-2 PLpro",          # wrong enzyme
        "Cytotoxicity against Vero cells",         # wrong enzyme (cytotox)
        "Antiviral activity against influenza",    # not 3C
    ]})
    kept, funnel = screen(df)
    assert len(kept) == funnel["kept"] == 1
    assert (funnel["kept"] + funnel["dropped_not_3c"]
            + funnel["dropped_wrong_enzyme"]) == len(df)


def test_empty_input_does_not_crash():
    kept, funnel = screen(pd.DataFrame())
    assert len(kept) == 0 and funnel["kept"] == 0


def test_excluded_targets_are_not_silently_absent_from_the_target_list():
    """Every exclusion carries a stated reason and is not in TARGETS."""
    assert EXCLUDED, "exclusions must be recorded, not simply omitted"
    ids = {tid for tid, _ in TARGETS.values()}
    for tid, reason in EXCLUDED.items():
        assert tid not in ids
        assert len(reason) > 20, "an exclusion needs a reason a reader can check"
