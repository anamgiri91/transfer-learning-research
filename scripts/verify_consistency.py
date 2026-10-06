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

Three kinds of check:

  COUNTS      A number stated in prose must equal the value derived from
              artefacts -- everywhere it is stated, in every document.
  SENTINELS   If an artefact exists, no summary may still claim the work it
              represents was not done.
  DERIVED     If the artefacts do not support a verdict, no summary may state
              it. Unlike a sentinel, the trigger is a value read out of a
              results table, not the mere existence of a file -- so a claim
              that becomes true when the numbers change stops being forbidden.

**A pattern that matches nothing is a check that passes vacuously**, which is
how the abstract's run count went unverified for the life of the project: the
regex required `\\d+` and the manuscript writes `1,040`. `check()` therefore
reports any `required` pattern that found no match in the shipped documents.
A fixture-driven test cannot catch that -- the fixtures were written to match.

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
# `reproduction-coverage.md` is a live summary, not a history -- it was missing
# from this list and carried "all 23 tables" unchallenged for three tables'
# worth of drift, which is precisely the failure the `all (\d+) tables` pattern
# below exists to catch.
DOCS = ["paper/manuscript.md", "paper/provenance.md", "plan.md", "README.md",
        "results/tables/README.md", "results/figures/README.md",
        "data/processed/README.md", "docs/revised_plan.md",
        "docs/reproduction-coverage.md"]

TABLES = Path("results/tables")


def n_metric_runs() -> int:
    return len(glob.glob("results/metrics/*.json"))


def complete_cells(folder: str, arms: dict[str, list[int]], split="scaffold") -> bool:
    """A single artifact cannot establish that a whole sweep finished."""
    for arm, sizes in arms.items():
        for size in sizes:
            for seed in range(10):
                path = Path(folder) / f"{arm}__{split}__seed{seed}__n{size}.json"
                if not path.exists():
                    return False
                record = json.loads(path.read_text())
                if not record.get("metrics") or record.get("error"):
                    return False
    return True


def check_current_status(text: str, doc: str) -> list[str]:
    """Check current assertions while retaining dated protocol amendments."""
    if Path(doc).name == "plan.md":
        text = text.split("## 10. Execution checklist", 1)[-1]
    if Path(doc).name == "revised_plan.md":
        return []  # explicitly superseded history
    text = re.sub(r"[*`]", "", text)
    text = re.sub(r"\s+", " ", text)
    rules = [
        ("amended-complete", complete_cells("results/metrics_ft", {
            "T2v": [50, 100, 250, 347], "T4ft": [347], "T5ft": [347]}), [
                r"Amended fine-tuning arms \(in progress\)",
                r"Status: incomplete at the time of writing",
                r"H3 arms are still running",
                r"no in-domain fine-tune exists",
                r"H3 is tested in a substituted form only",
            ]),
        ("b3-complete", complete_cells("results/metrics_b3", {"B3": [50, 100, 250, 347]}), [
            r"B3\s*(?:\([^)]*\))?\s*(?:was |is |has )?(?:not run|never run|not been run)",
            r"B3 \(D-MPNN from scratch\), T3 .{0,150}T6 .{0,80}were not run",
        ]),
        ("t2-cross-split-complete", all(complete_cells("results/metrics", {
            "T2_chemberta_full_finetune": [347]}, split) for split in ("random", "butina")), [
                r"T2 was not run on this split",
                r"T2's replication elsewhere is untested",
                r"fine-tune T2 was run on the scaffold split only",
        ]),
        ("structure-rows-not-assay-replicates", Path("data/processed/master.csv").exists(), [
            r"133 carry more than one measurement",
            r"The fidelity claim is supported",
            r"Wilcoxon ranks signs, not magnitudes",
        ]),
    ]
    return [f"{doc}: stale claim {label!r}: {match.group(0)!r}"
            for label, active, patterns in rules if active
            for pattern in patterns for match in re.finditer(pattern, text, re.I)]


def n_arms() -> int:
    arms = set()
    for f in glob.glob("results/metrics/*.json"):
        arms.add(json.load(open(f))["arm"])
    return len(arms)


def n_claims() -> int:
    # Assembled by verify_manuscript, not recounted here: this function used to
    # say `len(build_claims()) + 1` and went stale as soon as a second
    # self-referential claim was added.
    sys.path.insert(0, "scripts")
    import verify_manuscript as vm
    return len(vm.all_claims())


def n_tables() -> int:
    return len(glob.glob("results/tables/table*.csv"))


def n_figures() -> int:
    return len(glob.glob("results/figures/*.png"))


def n_split_files() -> int:
    return len(glob.glob("data/processed/splits/eva71_2a/*.json"))


def n_reproducible_artefacts() -> int:
    """How many committed artefacts `make verify-repro` actually diffs.

    §10 states this number, and a number stated in prose about a checker is
    exactly the kind that goes stale when the checker's coverage changes.
    """
    sys.path.insert(0, "scripts")
    import verify_reproducibility as vr
    return len(vr.files())


def coverage(key: str):
    """Read a coverage number from the artefact the RUN wrote.

    Not re-derived here. An earlier version recomputed these by a parallel
    calculation in this file, which is the duplicated-count drift the whole
    checker exists to prevent -- and it drifted within a day of being written.
    The number the paper quotes is now the number a passing full-tier run
    actually reported.
    """
    def _get() -> int:
        d = json.loads(Path("docs/reproduction-coverage.json").read_text())
        return int(d[key])
    return _get


def n_regenerated_artefacts() -> int:
    """How many of the watched artefacts a `verify-repro` run actually rewrites.

    Distinct from `n_reproducible_artefacts`, and the distinction is the point:
    §10 claimed one number for both until 2026-09-11, which read as though the
    transfer arms were being re-fitted. They are not. This counts the artefacts
    owned by a stage -- the datasets, the split files, the baseline metric
    files a stage clears and rewrites, and every table and figure -- so the
    prose cannot quietly re-inflate its coverage.
    """
    sys.path.insert(0, "scripts")
    import verify_reproducibility as vr
    owned = set()
    for _label, _argv, _net, _slow, clears in vr.STAGES:
        if clears:
            owned |= set(vr.ROOT.glob(clears))
    # Two tables are not rewritten by a run, for different reasons, and both
    # are excluded so the stated number matches what the run reports:
    #   table5_contamination -- measure_contamination.py needs PubChem and is
    #     not a stage at all, so this one is genuinely NOT covered.
    #   table0_split_audit   -- audit_splits.py recomputes it every run but
    #     writes only when a data row changes (a rewrite-on-every-run left a
    #     timestamp-only diff, which trains people to discard diffs unread).
    #     It IS covered; it is simply verified in place.
    not_rewritten = {vr.ROOT / "results/tables/table5_contamination.csv",
                     vr.ROOT / "results/tables/table0_split_audit.csv"}
    for pat in ("results/tables/*.csv", "results/figures/*.png",
                "data/processed/eva71_2a.csv", "data/processed/eva71_2a.curation.json",
                "data/processed/indomain_3c.csv",
                "data/processed/indomain_3c.curation.json"):
        owned |= set(vr.ROOT.glob(pat))
    return len(owned - not_rewritten)


def surrogate_diffs() -> set[int]:
    import pandas as pd
    df = pd.read_csv("results/tables/table9_surrogate_divergence.csv", comment="#")
    return {int(v) for v in df["n_differences"]}


def _primary_contrast(arm: str, reference: str) -> dict:
    """One row of the primary (full-data RMSE) family of table 10."""
    import pandas as pd
    df = pd.read_csv(TABLES / "table10_indomain_contrasts.csv", comment="#")
    row = df[(df.arm == arm) & (df.reference == reference)
             & (df.n_train == 347) & (df.metric == "rmse")
             & (df.family.str.startswith("primary"))]
    if len(row) != 1:
        raise AssertionError(
            f"table10 has {len(row)} primary full-data RMSE rows for "
            f"{arm} vs {reference}; expected exactly 1")
    return row.iloc[0].to_dict()


def holm_significant(arm: str, reference: str, alpha: float = 0.05) -> bool:
    return float(_primary_contrast(arm, reference)["p_holm"]) < alpha


# (label, regex capturing one integer, callable giving the truth, required)
# `required` says the fact IS stated in the shipped documents, so a pattern that
# stops matching is a rotted check rather than an absent claim.
#
# Patterns are deliberately narrow and current-tense, so that a retrospective
# mention of a superseded number ("left reading 85 after ...") does not trip.
# Thousands separators are allowed everywhere a count can exceed 999: omitting
# them is what silently disabled the run-count check.
# A separator-tolerant integer that must still START with a digit: `[\d,]+`
# alone matches a bare comma, which int() then rejects.
NUM = r"(\d[\d,]*)"

COUNTS = [
    ("evaluated runs", re.compile(rf"\({NUM} evaluated runs\)"),
     n_metric_runs, True),
    ("runs with stored endpoints", re.compile(rf"stored for all {NUM} runs"),
     n_metric_runs, True),
    ("machine-checked claims", re.compile(r"checks \*\*(\d+) claims\*\*"),
     n_claims, True),
    ("machine-checked claims", re.compile(rf"{NUM} machine-checked claims"),
     n_claims, False),
    ("machine-checked claims", re.compile(r"\*\*(\d+) numeric claims\*\*"),
     n_claims, False),
    ("tables regenerated", re.compile(r"data rows of all (\d+)\s*\n?tables"),
     n_tables, True),
    # `[Aa]ll 23 result tables` sat in the Declarations table for three tables'
    # worth of drift because this pattern required "tables" to follow the digit
    # immediately. One optional qualifier closes that.
    ("tables regenerated",
     re.compile(r"all (\d+) (?:result )?tables", re.I), n_tables, True),
    ("figures", re.compile(r"all (\d+) figures"), n_figures, True),
    ("split files", re.compile(r"(\d+) split files"), n_split_files, True),
    ("artefacts diffed by verify-repro",
     re.compile(rf"full-tier run \*\*reconstructs {NUM}\*\*"),
     coverage("reconstructed"), True),
    ("artefacts watched by verify-repro",
     re.compile(rf"[Ww]atching \*\*{NUM}\*\*\s+artefacts in total"),
     coverage("watched"), True),
    ("artefacts watched by verify-repro",
     re.compile(rf"watching {NUM} committed artefacts"),
     coverage("watched"), True),
    ("artefacts regenerated by verify-repro",
     re.compile(rf"\*\*reconstructs {NUM}\*\*"),
     coverage("reconstructed"), True),
    ("supplied inputs",
     re.compile(rf"[Ss]upplied inputs [—-] {NUM} files"),
     coverage("supplied_inputs"), True),
    ("artefacts with no committed baseline",
     re.compile(rf"further \*\*{NUM} artefacts are regenerated"),
     coverage("newly_produced_no_baseline"), True),
    ("compared only",
     re.compile(rf"[Cc]ompared only [—-] {NUM} files"),
     coverage("compared_only"), True),
]

WORD = {"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40}
_WORDS = "|".join(WORD)

# (label, regex, truth, required) for facts written as words rather than digits
WORD_COUNTS = [
    ("arms compared",
     re.compile(rf"\b({_WORDS}) arms were compared", re.I), n_arms, True),
    ("split files",
     re.compile(rf"\b({_WORDS}) split files", re.I), n_split_files, True),
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
                  r"no in-domain arm exists(?!.{0,200}listed)",
                  # 8.4 kept describing T4 as the experiment still to run, in a
                  # paper whose 6.4 reports it. Both halves of that sentence.
                  r"pre-registration and neither run",
                  r"highest-value remaining experiment"],
     "why": "an in-domain encoder exists and §6.4 reports it, so H3 is tested"},
    {"id": "surrogate-recomputed",
     "artefact": "results/tables/table9_surrogate_divergence.csv",
     "patterns": [r"as a five-residue surrogate",
                  r"a five-residue\s*\n?extrapolation",
                  # README asserted the source's count as our own finding, and
                  # asserted the stronger 'near the active site' form with it.
                  r"the two differ at five residues",
                  r"differ at five residues, none"],
     "why": "table9 re-derives the divergence as 7-8 residues, strain-dependent, "
            "so the paper no longer relies on the count of five"},
]

# If the predicate holds -- i.e. the artefacts do NOT support the verdict --
# none of the patterns may appear in a summary.
DERIVED = [
    {"id": "h3-not-holm-significant",
     "predicate": lambda: not holm_significant("T5", "T1"),
     "patterns": [r"H3 is\s*\n?supported", r"H3 (?:is|was) (?:now )?confirmed",
                  r"supported for the chained arm"],
     "why": "H3's strongest contrast (T5 vs T1, full-data RMSE) clears "
            "Benjamini-Hochberg and not Holm in table 10, so the paper's own "
            "standard makes it suggestive, not supported"},
    {"id": "h4-bounded-not-answered",
     "predicate": lambda: not holm_significant("T4c", "T4r"),
     "patterns": [r"H4 is\s*\n?answered", r"H4 (?:is|was) (?:now )?settled"],
     "why": "the contrast that would decide H4 (decontaminated vs the "
            "size-matched control) does not survive Holm in table 10, so H4 is "
            "bounded rather than answered"},
]


def _is_quoted(text: str, start: int, end: int) -> bool:
    """Is this match inside a quotation?

    §10 narrates the errors this checker exists to prevent, which means it
    reproduces their exact wording. Quoting a superseded claim in order to
    record that it was superseded is the opposite of leaving it standing, so
    matches wrapped in quotes are not stale claims.
    """
    before = text[max(0, start - 2):start]
    after = text[end:end + 2]
    quotes = ('"', "'", "“", "”", "‘", "’", "`")
    return (any(q in before for q in quotes) and any(q in after for q in quotes))


def _forbid(docs, patterns, fail_prefix, why, fails, ):
    """Report every unquoted occurrence of any pattern. Returns matches tried."""
    checked = 0
    for doc in docs:
        p = Path(doc)
        if not p.exists():
            continue
        text = p.read_text()
        for pat in patterns:
            for m in re.finditer(pat, text):
                checked += 1
                if _is_quoted(text, m.start(), m.end()):
                    continue        # a recorded past error, not a live claim
                fails.append(f"{doc}: {fail_prefix} -- {m.group(0)[:70]!r}. {why}")
    return checked


def check(docs: list[str] | None = None) -> tuple[list[str], int]:
    """Returns (failures, statements checked). Pure in `docs` so the
    regression tests can feed it fixtures.

    When `docs` is None the shipped documents are checked, and `required`
    patterns that matched nothing are reported: a check nobody's prose triggers
    is not a passing check.
    """
    shipped = docs is None
    DOCS = docs if docs is not None else globals()["DOCS"]
    fails: list[str] = []
    checked = 0

    for doc in DOCS:
        if Path(doc).exists():
            fails.extend(check_current_status(Path(doc).read_text(), doc))

    for label, pat, truth, required in COUNTS:
        want = None
        hits = 0
        for doc in DOCS:
            p = Path(doc)
            if not p.exists():
                continue
            for m in pat.finditer(p.read_text()):
                if want is None:
                    want = truth()
                checked += 1
                hits += 1
                got = int(m.group(1).replace(",", "")) if m.groups() else want
                if got != want:
                    fails.append(f"{doc}: states {got} for {label}, artefacts say {want}")
        if shipped and required and hits == 0:
            fails.append(
                f"dead pattern: /{pat.pattern}/ ({label}) matched nothing in any "
                f"document. A check that matches nothing passes vacuously; fix "
                f"the pattern or drop its `required` flag.")

    for label, pat, truth, required in WORD_COUNTS:
        hits = 0
        for doc in DOCS:
            p = Path(doc)
            if not p.exists():
                continue
            for m in pat.finditer(p.read_text()):
                checked += 1
                hits += 1
                got = WORD.get(m.group(1).lower())
                want = truth()
                if got != want:
                    fails.append(f"{doc}: states '{m.group(1)}' for {label}, "
                                 f"artefacts say {want}")
        if shipped and required and hits == 0:
            fails.append(
                f"dead pattern: /{pat.pattern}/ ({label}) matched nothing in any "
                f"document. A check that matches nothing passes vacuously; fix "
                f"the pattern or drop its `required` flag.")

    for s in SENTINELS:
        if not Path(s["artefact"]).exists():
            continue
        checked += _forbid(DOCS, s["patterns"],
                           f"stale claim {s['id']!r}", s["why"], fails)

    for d in DERIVED:
        try:
            triggered = d["predicate"]()
        except FileNotFoundError:
            continue                 # the results table is not built yet
        if not triggered:
            continue
        checked += _forbid(DOCS, d["patterns"],
                           f"unsupported verdict {d['id']!r}", d["why"], fails)

    return fails, checked


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
