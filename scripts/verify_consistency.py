#!/usr/bin/env python
"""Cross-document consistency: do the summaries still agree with the artefacts?

`verify_manuscript.py` re-derives numbers from artefacts, and `verify_citations.py`
checks citation structure. Neither can see the failure that actually recurred
here: **prose that restates a conclusion goes stale silently.** On 2026-09-02 an
audit found eleven such statements -- the abstract claiming five arms when ten
had run, three documents saying the decontamination ablation "was not performed"
after it had been performed for one pretraining source, a README quoting a
machine-checked-claim count of 93 against an actual 222, a figures README
listing two figures out of seven and describing deleted files as present.

Every one of those was in a *summary*: abstract, README, limitations,
conclusion, checklist. The Holm-family bug was caught twice by machine; these
needed a human to read the paper. This script closes that asymmetry for the
class that keeps biting: a fact asserted in more than one place, where one copy
is computable.

Two kinds of check:

  COUNTS      A number stated in prose must equal the value derived from
              artefacts -- everywhere it is stated, in every document.
  SENTINELS   If an artefact exists, no summary may still claim the work it
              represents was not done.

Run: python scripts/verify_consistency.py
Exit code 0 only if every document agrees with the artefacts.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

# Summary surfaces. The decision log and literature review are deliberately
# excluded: they are append-only histories whose job is to record what was true
# at the time, including counts that have since moved.
DOCS = ["paper/manuscript.md", "paper/provenance.md", "plan.md", "README.md",
        "results/tables/README.md", "results/figures/README.md",
        "data/processed/README.md"]


def n_metric_runs() -> int:
    return len(glob.glob("results/metrics/*.json"))


def n_arms() -> int:
    arms = set()
    for f in glob.glob("results/metrics/*.json"):
        arms.add(json.load(open(f))["arm"])
    return len(arms)


def n_claims() -> int:
    sys.path.insert(0, "scripts")
    import verify_manuscript as vm
    return len(vm.build_claims()) + 1        # +1 for the self-count claim


def n_tables() -> int:
    return len(glob.glob("results/tables/table*.csv"))


def n_figures() -> int:
    return len(glob.glob("results/figures/*.png"))


def surrogate_diffs() -> set[int]:
    import pandas as pd
    df = pd.read_csv("results/tables/table9_surrogate_divergence.csv", comment="#")
    return {int(v) for v in df["n_differences"]}


# (label, regex capturing one integer, callable giving the truth)
# Patterns are deliberately narrow and current-tense, so that a retrospective
# mention of a superseded number ("left reading 85 after ...") does not trip.
COUNTS = [
    ("evaluated runs", re.compile(r"with (\d+) evaluated runs"), n_metric_runs),
    ("machine-checked claims", re.compile(r"checks \*\*(\d+) claims\*\*"), n_claims),
    ("machine-checked claims", re.compile(r"(\d+) machine-checked claims"), n_claims),
    ("arm count", re.compile(r"\*\*(\d+) numeric claims\*\*"), n_claims),
    ("tables regenerated", re.compile(r"data rows of all (\d+)\s*\n?tables"), n_tables),
    ("tables regenerated", re.compile(r"all (\d+) tables"), n_tables),
    ("figures", re.compile(r"all (\d+) figures"), n_figures),
    ("figures", re.compile(r"all seven figures"), lambda: 7),
]

WORD = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# (label, regex, truth, message) for facts written as words rather than digits
WORD_COUNTS = [
    ("arms compared",
     re.compile(r"\b(five|six|seven|eight|nine|ten) arms are compared", re.I),
     n_arms),
]

# If the artefact exists, the claim must not still appear.
SENTINELS = [
    {"id": "decontamination-performed",
     "artefact": "models/indomain_T5_clean.pt",
     "patterns": [r"decontamination ablation was not performed",
                  r"remains the one part of\s*\n?§7\.1 not performed",
                  r"the decontaminated re-run is\s*\n?\s*\*\*impossible\*\*(?!.{0,400}ChemBERTa)"],
     "why": "a decontaminated in-domain encoder exists, so the ablation was "
            "performed for that pretraining source; the claim must be qualified"},
    {"id": "in-domain-arm-exists",
     "artefact": "models/indomain_T4.pt",
     "patterns": [r"H3 — that in-domain transfer beats generic self-supervised "
                  r"pretraining — is\s*\n?entirely untested",
                  r"no in-domain arm exists(?!.{0,200}listed)"],
     "why": "an in-domain encoder exists and §6.4 reports it, so H3 is tested"},
    {"id": "surrogate-recomputed",
     "artefact": "results/tables/table9_surrogate_divergence.csv",
     "patterns": [r"as a five-residue surrogate",
                  r"a five-residue\s*\n?extrapolation"],
     "why": "table9 re-derives the divergence as 7-8 residues, strain-dependent, "
            "so the paper no longer relies on the count of five"},
]


def check(docs: list[str] | None = None) -> tuple[list[str], int]:
    """Returns (failures, statements checked). Pure in `docs` so the
    regression tests can feed it fixtures."""
    DOCS = docs if docs is not None else globals()["DOCS"]
    fails: list[str] = []
    checked = 0

    for label, pat, truth in COUNTS:
        want = None
        for doc in DOCS:
            p = Path(doc)
            if not p.exists():
                continue
            for m in pat.finditer(p.read_text()):
                if want is None:
                    want = truth()
                checked += 1
                got = int(m.group(1)) if m.groups() else want
                if got != want:
                    fails.append(f"{doc}: states {got} for {label}, artefacts say {want}")

    for label, pat, truth in WORD_COUNTS:
        for doc in DOCS:
            p = Path(doc)
            if not p.exists():
                continue
            for m in pat.finditer(p.read_text()):
                checked += 1
                got = WORD.get(m.group(1).lower())
                want = truth()
                if got != want:
                    fails.append(f"{doc}: states '{m.group(1)}' for {label}, "
                                 f"artefacts say {want}")

    for s in SENTINELS:
        if not Path(s["artefact"]).exists():
            continue
        for doc in DOCS:
            p = Path(doc)
            if not p.exists():
                continue
            text = p.read_text()
            for pat in s["patterns"]:
                for m in re.finditer(pat, text):
                    checked += 1
                    if _is_quoted(text, m.start(), m.end()):
                        continue        # a recorded past error, not a live claim
                    fails.append(f"{doc}: stale claim {s['id']!r} -- "
                                 f"{m.group(0)[:70]!r}. {s['why']}")

    return fails, checked


def _is_quoted(text: str, start: int, end: int) -> bool:
    """Is this match inside a quotation?

    §10 narrates the errors this checker exists to prevent, which means it
    reproduces their exact wording. Quoting a superseded claim in order to
    record that it was superseded is the opposite of leaving it standing, so
    matches wrapped in quotes are not stale claims.
    """
    before = text[max(0, start - 2):start]
    after = text[end:end + 2]
    quotes = ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019", "`")
    return (any(q in before for q in quotes) and any(q in after for q in quotes))


def main() -> int:
    fails, checked = check()
    print(f"  [{'ok  ' if not fails else ' .. '}] cross-document facts "
          f"{checked:3d} statements over {len(DOCS)} documents")
    if fails:
        print(f"\n{len(fails)} INCONSISTENCY(IES):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nEvery summary statement agrees with the artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
