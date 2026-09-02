#!/usr/bin/env python
"""Verify the manuscript's citations the way verify_manuscript.py verifies its numbers.

Every error found in the 2026-09-02 citation audit lived in prose, and prose was
the one surface this project did not machine-check: a fabricated author initial,
a figure attributed to two papers that were not its source, a rule about
[secondary] sources stated in one document and broken in another, references
left author-less as "unretrievable" when open preprints existed, and -- while
fixing all that -- a reference list that briefly contained two entries numbered
10. None of it would have been caught by anything in `make verify`.

This script closes that gap. It checks structure and internal consistency
offline; `--online` additionally checks that every cited URL still resolves.

  1. numbering      reference numbers are unique and contiguous from 1
  2. labels         every reference carries a reading-depth label from the
                    vocabulary literature.md defines
  3. orphans        every reference is actually cited in the body
  4. dangling       every body citation appears in the reference list
  5. depth drift    a source's reading-depth label agrees between
                    manuscript.md and literature.md
  6. secondary rule literature.md says [secondary] entries are "not citable for
                    a specific number"; flag body sentences that cite one
                    alongside a quantity. This is the rule the audit found broken.

Run: python scripts/verify_citations.py [--online]
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MANUSCRIPT = Path("paper/manuscript.md")
LITERATURE = Path("docs/literature.md")

# The vocabulary literature.md defines. A label outside it is a failure, because
# an unlabelled or ad-hoc-labelled source is one whose provenance nobody checked.
LABELS = ["full text", "abstract", "secondary"]
LABEL_RE = re.compile(r"\*\*\[([^\]]+)\]\*\*")

MD_LINK = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
BARE_LINK = re.compile(r"<(https?://[^>\s]+)>")
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.I)
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})", re.I)
ARXIV_ID = re.compile(r"arXiv:([0-9]{4}\.[0-9]{4,5})", re.I)

# A "quantity" for the [secondary] rule: a bare number, percentage, p-value or
# count. Section references (§5.2), reference numbers and years are excluded.
QUANTITY = re.compile(r"(?<![§\w.])\d+(?:[.,]\d+)?\s*(?:%|K\b|M\b|-fold|×)|"
                      r"\bp\s*[<=]\s*0?\.\d+|\bRMSE\s+\d|\b\d+(?:\.\d+)?\s*(?:molecules|compounds)\b")
YEAR = re.compile(r"\b(19|20)\d{2}\b")


def identifiers(url: str) -> set[str]:
    """Canonical keys for a URL, so a DOI link and an arXiv link to the same
    paper are recognised as the same reference."""
    keys = {url.rstrip("/.,;").lower()}
    for m in DOI_RE.findall(url):
        keys.add("doi:" + m.rstrip("/.,;").lower())
    for m in ARXIV_RE.findall(url):
        keys.add("arxiv:" + m.lower())
    return keys


def entry_identifiers(text: str) -> set[str]:
    keys: set[str] = set()
    for url in MD_LINK.findall(text) + BARE_LINK.findall(text):
        keys |= identifiers(url)
    for m in DOI_RE.findall(text):
        keys.add("doi:" + m.rstrip("/.,;").lower())
    for m in ARXIV_ID.findall(text):
        keys.add("arxiv:" + m.lower())
    return keys


def split_manuscript(text: str) -> tuple[str, str]:
    i = text.index("## References")
    return text[:i], text[i:]


def parse_references(refs: str) -> list[tuple[int, str, str]]:
    """-> [(number, section heading, entry text)]"""
    out, section = [], ""
    lines = refs.splitlines()
    cur_n, buf = None, []
    for line in lines + ["### __end__"]:
        if line.startswith("###"):
            if cur_n is not None:
                out.append((cur_n, section, "\n".join(buf))); cur_n, buf = None, []
            section = line.lstrip("#").strip()
            continue
        m = re.match(r"^(\d+)\.\s", line)
        if m:
            if cur_n is not None:
                out.append((cur_n, section, "\n".join(buf)))
            cur_n, buf = int(m.group(1)), [line]
        elif cur_n is not None:
            buf.append(line)
    return out


def sentences(body: str) -> list[str]:
    flat = re.sub(r"\s+", " ", body)
    return re.split(r"(?<=[.!?])\s+", flat)


def check(text: str, lit: str, online: bool = False) -> tuple[list[str], list[tuple[str, int]]]:
    """Run every citation check. Returns (failures, per-check item counts).

    Pure in its inputs so the regression tests can feed it the exact defects
    the 2026-09-02 audit found.
    """
    body, refs = split_manuscript(text)
    entries = parse_references(refs)
    fails: list[str] = []

    # 1. numbering -----------------------------------------------------------
    nums = [n for n, _, _ in entries]
    if nums != list(range(1, len(nums) + 1)):
        dupes = {n for n in nums if nums.count(n) > 1}
        fails.append(f"reference numbering is not 1..N (n={len(nums)}); "
                     f"duplicates={sorted(dupes) or 'none'}; got {nums}")

    # 2. reading-depth labels ------------------------------------------------
    SOFTWARE = "Software, models and data resources"
    for n, section, entry in entries:
        if section == SOFTWARE:
            continue
        found = LABEL_RE.findall(entry)
        if not found:
            fails.append(f"reference {n} has no reading-depth label")
        elif not any(any(v in f for v in LABELS) for f in found):
            fails.append(f"reference {n} label {found!r} is outside {LABELS}")

    # 3/4. orphans and dangling citations ------------------------------------
    body_keys: set[str] = set()
    for url in MD_LINK.findall(body) + BARE_LINK.findall(body):
        body_keys |= identifiers(url)

    ref_key_sets = []
    for n, section, entry in entries:
        keys = entry_identifiers(entry)
        ref_key_sets.append((n, section, keys))
        if section == SOFTWARE:
            continue
        if keys and not (keys & body_keys):
            fails.append(f"reference {n} is never cited in the body "
                         f"(orphan; cite it or drop it)")

    all_ref_keys = set().union(*(k for _, _, k in ref_key_sets)) if ref_key_sets else set()
    for url in sorted(set(MD_LINK.findall(body) + BARE_LINK.findall(body))):
        if not (identifiers(url) & all_ref_keys):
            fails.append(f"body cites {url} but it is in no reference entry")

    # 5. reading-depth drift between manuscript and literature.md ------------
    for n, section, entry in entries:
        if section == SOFTWARE:
            continue
        keys = entry_identifiers(entry)
        m_label = next((f for f in LABEL_RE.findall(entry)
                        if any(v in f for v in LABELS)), None)
        for block in re.split(r"\n### ", lit):
            if entry_identifiers(block) & keys:
                l_label = next((f for f in LABEL_RE.findall(block)
                                if any(v in f for v in LABELS)), None)
                if l_label is None:
                    l_label = next((f for f in re.findall(r"\[([^\]]+)\]", block.split("\n")[0])
                                    if any(v in f for v in LABELS)), None)
                if m_label and l_label:
                    m_base = next(v for v in LABELS if v in m_label)
                    l_base = next(v for v in LABELS if v in l_label)
                    if m_base != l_base:
                        fails.append(
                            f"reference {n}: reading depth '{m_base}' in manuscript "
                            f"but '{l_base}' in literature.md")
                break

    # 6. the [secondary] rule -------------------------------------------------
    secondary_keys: set[str] = set()
    for n, section, entry in entries:
        lbl = next((f for f in LABEL_RE.findall(entry) if "secondary" in f), None)
        if lbl:
            secondary_keys |= entry_identifiers(entry)
    for sent in sentences(body):
        urls = MD_LINK.findall(sent) + BARE_LINK.findall(sent)
        if not urls:
            continue
        keys: set[str] = set()
        for u in urls:
            keys |= identifiers(u)
        if not (keys & secondary_keys):
            continue
        stripped = YEAR.sub("", sent)
        if QUANTITY.search(stripped):
            fails.append("a [secondary] source is cited in a sentence carrying a "
                         f"quantity, which literature.md forbids: {sent[:150]!r}")

    # 7. optional liveness ----------------------------------------------------
    if online:
        import requests
        urls = sorted({u for _, _, e in entries
                       for u in MD_LINK.findall(e) + BARE_LINK.findall(e)})
        for u in urls:
            try:
                r = requests.head(u, timeout=25, allow_redirects=True)
                if r.status_code >= 400 and r.status_code not in (403, 405, 429):
                    r = requests.get(u, timeout=25, allow_redirects=True)
                if r.status_code >= 400 and r.status_code not in (403, 405, 429):
                    fails.append(f"cited URL returns {r.status_code}: {u}")
            except Exception as exc:                      # noqa: BLE001
                fails.append(f"cited URL unreachable ({type(exc).__name__}): {u}")

    checks = [("numbering", len(nums)), ("reading-depth labels", len(entries)),
              ("orphan references", len(entries)),
              ("dangling body citations", len(set(MD_LINK.findall(body)))),
              ("depth drift vs literature.md", len(entries)),
              ("[secondary]-with-a-number rule", len(secondary_keys))]
    return fails, checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true",
                    help="also check that every cited URL resolves")
    args = ap.parse_args()

    entries = parse_references(split_manuscript(MANUSCRIPT.read_text())[1])
    fails, checks = check(MANUSCRIPT.read_text(), LITERATURE.read_text(), args.online)
    for name, n in checks:
        mark = " .. " if fails else "ok  "
        print(f"  [{mark}] {name:<32} {n:3d} items")

    if fails:
        print(f"\n{len(fails)} CITATION PROBLEM(S):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"\nAll citation checks passed over {len(entries)} references.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
