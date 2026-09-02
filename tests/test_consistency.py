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
