#!/usr/bin/env python
"""Which numbers in the manuscript's prose are actually derived from an artefact?

`verify_manuscript.py` checks the claims it has been given. It cannot tell you
about a number nobody ever wrote a claim for -- and that is how the compute
costs ("~160 s", "~80x", "two orders of magnitude") survived the whole project
before being measured at 185 s and 69x.

This is a reporting tool, not a gate. It classifies every numeric token in the
prose as:

  covered    the value appears as the expected value of a verify_manuscript
             claim, or inside a generated table
  structural section numbers, list markers, figure and reference numbers,
             seeds, thresholds and other design constants declared below
  UNCOVERED  a quantity asserted in prose that nothing re-derives

The UNCOVERED list is meant to be read, not to be empty: some numbers are
legitimately narrative. The point is that each one should be a deliberate
choice rather than an oversight.

Usage: python scripts/audit_number_coverage.py [--context]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MANUSCRIPT = Path("paper/manuscript.md")

# Design constants and structural values: declared, not derived.
STRUCTURAL = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",   # list/section/seeds
    "11", "12", "13", "14", "15", "16", "17", "18",           # reference numbers
    "0.05",                                                    # significance level
    "0.2", "0.1",                                              # split fractions
    "0.6", "0.7", "0.8",                                       # similarity thresholds
    "50", "100", "250", "347",                                 # training-set sizes
    "20",                                                      # early-stopping patience
    "2048", "4",                                               # fingerprint bits / radius
    "95",                                                      # CI level
    "1.0",                                                     # cliff delta
}


def register_values() -> set[str]:
    """Quantities borrowed from other papers are covered by the claim-support
    register (docs/citation-claims.yaml), not by verify_manuscript."""
    import yaml
    reg = Path("docs/citation-claims.yaml")
    if not reg.exists():
        return set()
    out: set[str] = set()
    for e in (yaml.safe_load(reg.read_text()) or []):
        q = str(e.get("quantity", ""))
        out |= {q, q.replace(",", "")}
        # the quote often carries the companion figures of the same claim
        for m in re.finditer(r"\d[\d,.]*", str(e.get("quote", ""))):
            out |= {m.group(0), m.group(0).replace(",", "")}
    return out


def claim_values() -> set[str]:
    sys.path.insert(0, "scripts")
    import verify_manuscript as vm
    vals: set[str] = set()
    for c in vm.build_claims():
        for v in (c.expected, c.actual):
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                f = float(v)
                vals |= {str(v), f"{f:g}", f"{abs(f):g}"}
                for nd in (1, 2, 3, 4):
                    vals.add(f"{abs(f):.{nd}f}")
                    vals.add(f"{abs(f):.{nd}f}".rstrip("0").rstrip("."))
                if abs(f) >= 1000:
                    vals.add(f"{abs(f):,.0f}")
                if f == int(f):
                    vals.add(str(int(abs(f))))
            else:
                vals.add(str(v))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", action="store_true", help="show the sentence")
    args = ap.parse_args()

    text = MANUSCRIPT.read_text()
    body = text[:text.index("## References")]
    tables = re.findall(r"<!-- TABLE:.*?END -->", body, flags=re.S)
    prose = re.sub(r"<!-- TABLE:.*?END -->", "", body, flags=re.S)
    prose = re.sub(r"```.*?```", "", prose, flags=re.S)
    # Strip things whose digits are identifiers or structure, not quantities:
    prose = re.sub(r"https?://\S+", " ", prose)          # URLs
    prose = re.sub(r"<[^>]*>", " ", prose)                # bare links / html
    prose = re.sub(r"10\.\d{4,9}/\S+", " ", prose)       # DOIs
    prose = re.sub(r"arXiv:\S+", " ", prose)             # arXiv ids
    prose = re.sub(r"(?m)^#{1,6} .*$", " ", prose)        # headings (section numbers)
    prose = re.sub(r"(?m)^\s*\d+\.\s", " ", prose)      # ordered-list markers
    prose = re.sub(r"§+\s?\d+(\.\d+)*", " ", prose)      # section cross-references
    prose = re.sub(r"(?:Figure|Table)\s+\d+", " ", prose)
    prose = re.sub(r"CHEMBL\d+|PMC\d+|\b[A-Z]\d[A-Z0-9]{3,}\b", " ", prose)
    in_tables = " ".join(tables)

    known = claim_values() | register_values()
    flat = re.sub(r"\s+", " ", prose)

    uncovered: list[tuple[str, str]] = []
    n_cov = n_struct = 0
    seen: set[str] = set()
    for m in re.finditer(r"(?<![\w.§\-])(\d+(?:[.,]\d+)*)(?![\w])", prose):
        tok = m.group(1)
        if tok in seen:
            continue
        seen.add(tok)
        if tok in STRUCTURAL or re.fullmatch(r"(19|20)\d\d", tok):
            n_struct += 1
        elif tok in known or tok.replace(",", "") in known or tok in in_tables:
            n_cov += 1
        else:
            i = flat.find(tok)
            ctx = flat[max(0, i - 90):i + 60] if i >= 0 else ""
            uncovered.append((tok, ctx))

    print(f"prose numeric tokens: {n_cov} covered, {n_struct} structural, "
          f"{len(uncovered)} UNCOVERED\n")
    for tok, ctx in uncovered:
        print(f"  {tok:>10}")
        if args.context:
            print(f"             …{ctx}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
