"""Regression tests for the citation checker.

Every case here is a defect the 2026-09-02 citation audit actually found in
this manuscript. They are pinned as tests because all of them lived in prose,
which was the one surface `make verify` did not cover -- the audit found seven
errors by hand and the point of the checker is that the eighth is found by CI.
"""
import pytest

from scripts.verify_citations import check as _check


def check(text, lit, **kw):
    """Structural checks only; the register has its own tests below."""
    return _check(text, lit, register=None, **kw)

LIT = """# Annotated literature review

- **[full text]** -- the complete paper was retrieved and read.
- **[abstract]** -- only the abstract was retrieved.
- **[secondary]** -- known through a search summary. **Not citable for a
  specific number.**

### 1.1 Guo, Hernandez-Hernandez, Ballester (2024) - [abstract]
*Scaffold splits overestimate virtual screening performance.*
<https://arxiv.org/abs/2406.00873>

### 1.2 Sultan et al. (2025) - [secondary]
*Domain adaptation.* <https://arxiv.org/abs/2503.03360>
"""

BODY = """# Paper

Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873).

## References

### Evaluation

1. Guo, Q., Hernandez-Hernandez, S., Ballester, P. J. (2024). *Scaffold splits
   overestimate virtual screening performance.* <https://arxiv.org/abs/2406.00873>
   -- **[abstract]**.
"""


def test_a_clean_manuscript_passes():
    fails, _ = check(BODY, LIT)
    assert fails == [], fails


def test_duplicate_reference_numbers_are_caught():
    """The bug introduced while fixing the audit: two entries numbered 10."""
    bad = BODY.replace(
        "1. Guo, Q.",
        "1. Placeholder. <https://example.org/a> -- **[abstract]**.\n\n1. Guo, Q.")
    fails, _ = check(bad, LIT)
    assert any("numbering" in f for f in fails), fails


def test_non_contiguous_numbering_is_caught():
    bad = BODY.replace("1. Guo, Q.", "3. Guo, Q.")
    fails, _ = check(bad, LIT)
    assert any("numbering" in f for f in fails), fails


def test_missing_reading_depth_label_is_caught():
    bad = BODY.replace("   -- **[abstract]**.\n", "\n")
    fails, _ = check(bad, LIT)
    assert any("reading-depth label" in f for f in fails), fails


def test_orphan_reference_is_caught():
    """Reference 9 in the real manuscript: listed, never cited."""
    bad = BODY.replace(
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873).",
        "Splitting is discussed.")
    fails, _ = check(bad, LIT)
    assert any("never cited" in f for f in fails), fails


def test_body_citation_missing_from_the_reference_list_is_caught():
    bad = BODY.replace(
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873).",
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873) "
        "and [Snyder et al.](https://www.nature.com/articles/s42004-024-01220-4).")
    fails, _ = check(bad, LIT)
    assert any("in no reference entry" in f for f in fails), fails


def test_reading_depth_drift_between_documents_is_caught():
    """literature.md says [abstract], the manuscript says [full text]."""
    bad = BODY.replace("-- **[abstract]**", "-- **[full text]**")
    fails, _ = check(bad, LIT)
    assert any("reading depth" in f for f in fails), fails


def test_secondary_source_cited_with_a_number_is_caught():
    """The rule literature.md states and the manuscript broke: the n=50
    crossover was sourced from a [secondary] entry."""
    bad = BODY.replace(
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873).",
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873). "
        "Domain adaptation works on 4K molecules "
        "[Sultan](https://arxiv.org/abs/2503.03360).")
    bad = bad.replace(
        "1. Guo, Q.",
        "1. Sultan, A. *Domain adaptation.* <https://arxiv.org/abs/2503.03360> "
        "-- **[secondary]**.\n\n2. Guo, Q.")
    fails, _ = check(bad, LIT)
    assert any("secondary" in f for f in fails), fails


def test_a_year_alone_does_not_trip_the_secondary_rule():
    """Citing a [secondary] source with only its year is legitimate."""
    ok = BODY.replace(
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873).",
        "Splitting is discussed by [Guo et al.](https://arxiv.org/abs/2406.00873). "
        "See also [Sultan et al. 2025](https://arxiv.org/abs/2503.03360) for framing.")
    ok = ok.replace(
        "1. Guo, Q.",
        "1. Sultan, A. *Domain adaptation.* <https://arxiv.org/abs/2503.03360> "
        "-- **[secondary]**.\n\n2. Guo, Q.")
    fails, _ = check(ok, LIT)
    assert not any("secondary" in f for f in fails), fails


def test_doi_and_arxiv_links_to_the_same_paper_are_matched():
    """The body may cite a DOI where the reference lists an arXiv id."""
    body = BODY.replace("<https://arxiv.org/abs/2406.00873>\n   -- **[abstract]**.",
                        "arXiv:2406.00873 -- **[abstract]**.")
    fails, _ = check(body, LIT)
    assert not any("never cited" in f or "no reference entry" in f for f in fails), fails


def test_missing_embedded_figure_is_caught(tmp_path, monkeypatch):
    """A renumbering once left a caption pointing at a file that had moved."""
    body = BODY.replace("# Paper", "# Paper\n\n![x](../results/figures/nope.png)\n\n"
                                   "**Figure 1 — x.**")
    fails, _ = check(body, LIT)
    assert any("does not exist" in f for f in fails), fails


def test_non_sequential_figure_numbers_are_caught(tmp_path, monkeypatch):
    body = BODY.replace("# Paper", "# Paper\n\n**Figure 1 — a.**\n\n**Figure 3 — b.**")
    fails, _ = check(body, LIT)
    assert any("figure numbering" in f for f in fails), fails


@pytest.mark.parametrize("label", ["[read twice]", "[skimmed]", "[trust me]"])
def test_labels_outside_the_vocabulary_are_rejected(label):
    bad = BODY.replace("**[abstract]**", f"**{label}**")
    fails, _ = check(bad, LIT)
    assert any("outside" in f for f in fails), fails


# ---------------------------------------------------------------------------
# The claim-support register: does a source actually say what we claim?
# ---------------------------------------------------------------------------

import yaml

from scripts.verify_citations import REGISTER, _same_work, check_register


def _entry(**over):
    e = {
        "id": "x", "attributed_to": "Someone 2024",
        "source_url": "https://arxiv.org/abs/1234.56789",
        "quote_source_url": "https://arxiv.org/abs/1234.56789",
        "quantity": "50", "claim_in_manuscript": "roughly 50 molecules",
        "quote": "from 50 molecules upward things change",
        "quote_read_in": "the abstract", "second_hand": False,
        "retrieved": "2026-09-02", "status": "verified",
    }
    e.update(over)
    return e


def _run(entries, body="text with roughly 50 molecules in it", monkeypatch=None, tmp_path=None):
    f = tmp_path / "reg.yaml"
    f.write_text(yaml.safe_dump(entries))
    return check_register(body, register=f)[0]


def test_register_entry_that_supports_its_claim_passes(tmp_path, monkeypatch):
    assert _run([_entry()], monkeypatch=monkeypatch, tmp_path=tmp_path) == []


def test_quote_not_containing_the_quantity_is_caught(tmp_path, monkeypatch):
    """A quote that does not contain the number cannot support it."""
    bad = _entry(quote="machine learning is useful for molecules")
    fails = _run([bad], monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert any("does not support" in f for f in fails), fails


def test_claim_no_longer_in_the_manuscript_is_caught(tmp_path, monkeypatch):
    fails = _run([_entry()], body="unrelated prose",
                 monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert any("no longer in the manuscript" in f for f in fails), fails


def test_undeclared_second_hand_quote_is_caught(tmp_path, monkeypatch):
    """The exact 2026-09-02 error: the sentence was read in Schimunek but
    belongs to Snyder, and nothing recorded that."""
    bad = _entry(source_url="https://www.nature.com/articles/s42004-024-01220-4",
                 quote_source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12076497/",
                 second_hand=False)
    fails = _run([bad], monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert any("second_hand" in f for f in fails), fails


def test_second_hand_wrongly_declared_is_also_caught(tmp_path, monkeypatch):
    fails = _run([_entry(second_hand=True)], monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert any("second_hand" in f for f in fails), fails


def test_missing_field_is_caught(tmp_path, monkeypatch):
    bad = _entry(); del bad["quote"]
    fails = _run([bad], monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert any("missing field" in f for f in fails), fails


def test_superseded_entries_are_not_required_to_still_be_claimed(tmp_path, monkeypatch):
    """A superseded entry is kept for provenance, not enforced."""
    e = _entry(status="superseded", quote="no number here",
               claim_in_manuscript="text that is absent")
    assert _run([e], body="unrelated", monkeypatch=monkeypatch, tmp_path=tmp_path) == []


def test_a_doi_and_its_landing_page_count_as_the_same_work():
    assert _same_work("https://doi.org/10.5281/zenodo.20026661",
                      "https://zenodo.org/records/20026661")
    assert _same_work("https://arxiv.org/abs/2503.03360",
                      "https://arxiv.org/pdf/2503.03360")
    assert not _same_work("https://www.nature.com/articles/s42004-024-01220-4",
                          "https://pmc.ncbi.nlm.nih.gov/articles/PMC12076497/")


def test_the_shipped_register_is_internally_consistent():
    """The real file, not a fixture: every entry parses and is well formed."""
    entries = yaml.safe_load(REGISTER.read_text())
    assert entries, "register must not be empty"
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate register ids"
    for e in entries:
        assert e["status"] in {"verified", "superseded"}
        assert str(e["retrieved"]).startswith("2026-"), e["id"]
