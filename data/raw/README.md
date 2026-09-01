# data/raw — immutable source records

One subdirectory per source. Each needs a `*.manifest.json` recording the exact
query, retrieval date, and row count, so a rebuild is checkable.

| Source | Target | Access | Licence | Status |
|---|---|---|---|---|
| ChEMBL | EV-A71 3C protease (`CHEMBL####`) | public API | CC BY-SA 3.0 | to fetch |
| ChEMBL | EV-A71 2A protease | public API | CC BY-SA 3.0 | to fetch |
| PubChem BioAssay | EV-A71 antiviral / protease AIDs | public API | public domain | to fetch |
| BindingDB | picornaviral 3C / 3CL relatives | public download | CC BY 3.0 | to fetch |
| PDB | EV-A71 2A/3C holo structures | public | CC0 | to fetch |
| Literature curation | hand-extracted IC50/Ki from primary papers | manual | per-publisher | not started |

## Private inputs

Any restricted dataset lives in `private/data/` and **must** be listed here
with its absence documented — never silently omitted.

| File (in private/data) | Source | Why restricted | Affects |
|---|---|---|---|
| _(none yet)_ | | | |

## Inclusion criteria (enforced in `src/evapro/data/curate.py`)

- Activity type in {IC50, Ki, Kd, EC50}, `standard_relation == '='` (censored
  values are kept in a separate file for a sensitivity analysis, not dropped
  silently)
- Units convertible to nM; `pActivity = 9 - log10(nM)`
- Single-protein assay, confidence score >= 8 where the source provides one
- Structure parses in RDKit, desalted, neutralised, no mixtures
- Duplicate (InChIKey, target, activity type) collapsed to the median, and
  **discarded if the spread exceeds 1 log unit** — this is the "high-fidelity"
  criterion the paper's title claims, so it is a hard filter, not a nudge
