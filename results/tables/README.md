# results/tables

**Tracked, not gitignored** (`.gitignore` negates `results/tables/**`): the
manuscript's tables are rendered from these CSVs by
`scripts/render_manuscript_tables.py`, and its prose numbers are re-derived
from them by `scripts/verify_manuscript.py`, so a reviewer needs them in the
repository to check either.

Regenerate everything with:

```bash
python scripts/make_report.py --require-seeds 10   # tables 0-4
python scripts/measure_contamination.py            # table 5   (PubChem, network)
python scripts/analyse_cliffs.py                   # tables 6-8
python scripts/verify_surrogate.py                 # table 9   (UniProt, cached)
python scripts/analyse_indomain.py                 # tables 10-11
python scripts/prepare_indomain.py                 # table 12
python scripts/analyse_tuning.py                   # table 13
python scripts/analyse_endpoints.py                # table 14
```

Verified 2026-09-11: a full regeneration reproduces the **data rows of all 27
tables byte-identically**. Only the `#`-prefixed provenance header changes,
because it carries a generation timestamp.

| Table | Contents | Section |
|---|---|---|
| 0 | split leakage + nearest-neighbour audit | §5.1 |
| 1 | learning curves per split | §5.2 |
| 2 | data-efficiency ratios | §5.3 |
| 3 | paired tests vs B1 (**pre-registered arm family only**) | §5.4 |
| 4 | split difficulty (R², skill vs B0) | §5.5 |
| 5 | pretraining-corpus contamination upper bound | §6.1 |
| 6 | per-stratum cliff performance | §6.3 |
| 7 | dataset cliff-pair census | §6.3 |
| 8 | per-stratum paired tests (**pre-registered family only**) | §6.3 |
| 9 | CVA16/EV-A71 2A divergence, from UniProt | §3.1 |
| 10 | in-domain contrasts and controls | §6.4 |
| 11 | untrained-encoder sensitivity to the random draw | §6.4 |
| 12 | corpus vs evaluation-set chemistry | §6.4 |
| 13 | tuned fine-tune vs **both** baseline bases | §6.2 |
| 14 | every pre-registered endpoint, incl. the two omitted | §5.7 |

**Two tables pin their arm family on purpose.** Tables 3 and 8 are
Holm-corrected across arms, so admitting an arm added later silently re-corrects
the p-values of the pre-registered comparisons. Both happened here — table 3
moved T2 from 0.074 to 0.111, table 8 moved T1's `distant` p from 0.018 to
0.035 — and both are now pinned to the five arms frozen in `plan.md` §4. Arms
added afterwards are compared in table 10 as pre-specified pairwise contrasts.
