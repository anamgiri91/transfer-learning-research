"""Regression tests for the cross-document consistency checker.

Each case is a stale summary statement the 2026-09-02 audit actually found.
They were all in prose that *restates* a conclusion rather than computing one,
which is why neither verify_manuscript.py nor verify_citations.py could see
them -- eleven of them needed a human read of the paper. These pin the ones a
machine can catch.
"""
import pytest

from scripts.verify_consistency import check


def _doc(tmp_path, text, name="doc.md"):
    f = tmp_path / name
    f.write_text(text)
    return [str(f)]


def test_a_document_with_no_countable_claims_passes(tmp_path):
    fails, n = check(_doc(tmp_path, "# Paper\n\nNothing countable here.\n"))
    assert fails == [] and n == 0


def test_wrong_evaluated_run_count_is_caught(tmp_path):
    """The abstract said 640 after the count had moved to 720."""
    fails, _ = check(_doc(tmp_path, "compared with 3 evaluated runs.\n"))
    assert any("evaluated runs" in f for f in fails), fails


def test_correct_evaluated_run_count_passes(tmp_path):
    import glob
    n = len(glob.glob("results/metrics/*.json"))
    fails, _ = check(_doc(tmp_path, f"compared with {n} evaluated runs.\n"))
    assert not any("evaluated runs" in f for f in fails), fails


def test_wrong_arm_word_count_is_caught(tmp_path):
    """'Five arms are compared' survived until ten arms had run."""
    fails, _ = check(_doc(tmp_path, "Five arms are compared on identical splits.\n"))
    assert any("arms compared" in f for f in fails), fails


def test_wrong_claim_count_is_caught(tmp_path):
    """README carried a hand-typed 93 against an actual 222."""
    fails, _ = check(_doc(tmp_path, "make verify # 93 machine-checked claims\n"))
    assert any("machine-checked claims" in f for f in fails), fails


@pytest.mark.parametrize("stale", [
    "the decontamination ablation was not performed, so results are an upper bound",
    "Affinities are measured on CVA16 2A protease as a five-residue surrogate for EV-A71.",
])
def test_sentinel_catches_work_that_has_since_been_done(tmp_path, stale):
    """If the artefact exists, no summary may still say the work was not done."""
    fails, _ = check(_doc(tmp_path, stale + "\n"))
    assert any("stale claim" in f for f in fails), fails


def test_a_retrospective_mention_of_a_superseded_count_is_not_flagged(tmp_path):
    """§10 legitimately narrates that the count 'was left reading 85'. The
    patterns are current-tense on purpose so history does not trip them."""
    text = ("It has already caught two errors: this sentence itself, left "
            "reading 85 after the ablations of §6 added checks.\n")
    fails, _ = check(_doc(tmp_path, text))
    assert fails == [], fails


def test_a_quoted_past_claim_is_not_flagged(tmp_path):
    """§10 records the errors this checker prevents, so it reproduces their
    wording verbatim. A quoted superseded claim is documentation, not a live
    assertion -- and the un-quoted form must still be caught."""
    quoted = ('It previously said decontamination "remains the one part of '
              '\u00a77.1 not performed", which was wrong.\n')
    fails, _ = check(_doc(tmp_path, quoted))
    assert fails == [], fails

    bare = "Decontamination remains the one part of \u00a77.1 not performed.\n"
    fails, _ = check(_doc(tmp_path, bare))
    assert any("stale claim" in f for f in fails), fails


def test_the_shipped_documents_are_consistent():
    """The real files, not fixtures."""
    fails, checked = check()
    assert fails == [], fails
    assert checked > 0, "checker matched nothing at all -- patterns may have rotted"


# ---------------------------------------------------------------------------
# The blind spot the 2026-09-06 audit found: several patterns matched nothing
# in the shipped documents and so passed vacuously, while the fixtures above
# kept passing because they were written to match. A fixture-driven regression
# suite cannot see a rotted pattern; only the real corpus can.
# ---------------------------------------------------------------------------

def test_a_run_count_with_a_thousands_separator_is_checked(tmp_path):
    """The live failure: the abstract writes '1,040 evaluated runs' and the
    pattern required \\d+, so the check silently matched nothing for the whole
    life of the project."""
    fails, n = check(_doc(tmp_path, "compared with 1,040 evaluated runs.\n"))
    assert n > 0, "a separator-bearing count was not matched at all"
    import glob
    actual = len(glob.glob("results/metrics/*.json"))
    if actual != 1040:
        assert any("evaluated runs" in f for f in fails), fails


def test_an_arm_count_above_ten_is_checked(tmp_path):
    """Same shape: the vocabulary stopped at 'ten' while the abstract had moved
    to 'Twelve arms are compared'."""
    fails, n = check(_doc(tmp_path, "Twelve arms are compared on identical splits.\n"))
    assert n > 0, "'twelve' was not matched at all"


def test_a_required_pattern_that_matches_nothing_is_reported(monkeypatch):
    """The guard itself. A check that matches nothing is not a passing check."""
    import re
    from scripts import verify_consistency as vc

    dead = ("fabricated", re.compile(r"this phrase appears in no document (\d+)"),
            lambda: 1, True)
    monkeypatch.setattr(vc, "COUNTS", vc.COUNTS + [dead])
    fails, _ = vc.check()
    assert any("dead pattern" in f for f in fails), fails


def test_a_pattern_not_marked_required_may_match_nothing(monkeypatch):
    """Defensive patterns are allowed to lie dormant; only `required` ones must
    fire. Otherwise the guard would force prose to exist to satisfy a checker."""
    import re
    from scripts import verify_consistency as vc

    dormant = ("fabricated", re.compile(r"this phrase appears in no document (\d+)"),
               lambda: 1, False)
    monkeypatch.setattr(vc, "COUNTS", vc.COUNTS + [dormant])
    fails, _ = vc.check()
    assert not any("dead pattern" in f for f in fails), fails


@pytest.mark.parametrize("stale", [
    # 8.4 described the in-domain arm as still to be run, in a paper reporting it
    "Two experiments, both specified in the pre-registration and neither run:",
    "This is the highest-value remaining experiment, because it tests H3.",
    # README asserted the source's five-residue count as our own finding
    "Affinities are measured on CVA16 2A protease; the two differ at five residues, none near the active site.",
])
def test_sentinel_catches_the_2026_09_06_stale_summaries(tmp_path, stale):
    fails, _ = check(_doc(tmp_path, stale + "\n"))
    assert any("stale claim" in f for f in fails), fails


def test_a_verdict_the_tables_do_not_support_is_caught(tmp_path):
    """H3 clears Benjamini-Hochberg and not Holm, so no summary may call it
    supported. Unlike a sentinel this is keyed on a p-value, not a file: if a
    larger seed budget ever pushed it under Holm, the claim becomes allowed."""
    fails, _ = check(_doc(tmp_path, "H3 is supported for the chained arm T5.\n"))
    assert any("h3-not-holm-significant" in f for f in fails), fails


def test_h4_may_not_be_called_answered(tmp_path):
    """plan.md's checklist said 'H4 is answered negatively there' while the same
    document's closing paragraph and manuscript 6.5 both say it is bounded."""
    fails, _ = check(_doc(tmp_path, "H4 is answered negatively for the in-domain arms.\n"))
    assert any("h4-bounded-not-answered" in f for f in fails), fails


def test_the_derived_verdicts_reflect_the_table_not_a_hardcoded_answer():
    """The predicate must actually read table 10, so that it stops forbidding
    the claim when the evidence changes."""
    from scripts import verify_consistency as vc

    assert not vc.holm_significant("T5", "T1"), "H3's contrast is not Holm-significant"
    assert not vc.holm_significant("T4c", "T4r"), "H4's contrast is not Holm-significant"
    # A contrast that IS Holm-significant, as a control on the reader function.
    assert vc.holm_significant("T5", "B1"), "T5 vs B1 is Holm-significant (p = 0.030)"
