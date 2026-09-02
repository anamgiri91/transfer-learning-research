"""Regression tests for the citation checker.

Every case here is a defect the 2026-09-02 citation audit actually found in
this manuscript. They are pinned as tests because all of them lived in prose,
which was the one surface `make verify` did not cover -- the audit found seven
errors by hand and the point of the checker is that the eighth is found by CI.
"""
import pytest

from scripts.verify_citations import check

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


@pytest.mark.parametrize("label", ["[read twice]", "[skimmed]", "[trust me]"])
def test_labels_outside_the_vocabulary_are_rejected(label):
    bad = BODY.replace("**[abstract]**", f"**{label}**")
    fails, _ = check(bad, LIT)
    assert any("outside" in f for f in fails), fails
