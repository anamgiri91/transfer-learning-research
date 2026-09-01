# Provenance map

Every table, figure and dataset-level number in `manuscript.md` resolves through
this map to a **script** and an **output file on disk**. No number is typed by
hand into the manuscript; if a value is not reachable from this table it does not
belong in the paper.

## Data artefacts

| Artefact | Produced by | Output file | Read from |
|---|---|---|---|
| Raw structure–affinity release | — (third party) | `data/raw/OpenBind_EV-A71_2A.zip`, `data/processed/master.csv` | Zenodo 10.5281/zenodo.20026661 |
| Compound-level curated set | `scripts/prepare_openbind.py` | `data/processed/eva71_2a.csv` | `data/processed/master.csv` |
| Curation funnel counts | `scripts/prepare_openbind.py` | `data/processed/eva71_2a.curation.json` | as above |
| Split assignments | `scripts/build_splits.py` | `data/processed/splits/eva71_2a/{scaffold,random}__seed{0..9}.json` | `eva71_2a.csv` |
| Per-run metrics | `scripts/run_arms.py` | `results/metrics/<arm>__<split>__seed<N>__n<size>.json` | curated set + split files |

## Tables and figures

| Manuscript object | Produced by | Output file |
|---|---|---|
| Table 1 — learning curves | `scripts/make_report.py` | `results/tables/table1_learning_curves__<split>.csv` |
| Table 2 — data-efficiency ratio | `scripts/make_report.py` | `results/tables/table2_der__<split>.csv` |
| Table 3 — paired tests vs baseline | `scripts/make_report.py` | `results/tables/table3_paired_tests__<split>.csv` |
| Table 4 — split difficulty (R², skill vs B0) | `scripts/make_report.py` | `results/tables/table4_split_difficulty.csv` |
| Figure 1 — RMSE learning curves | `scripts/make_report.py` | `results/figures/fig1_learning_curves__<split>.png` |
| Figure 2 — Spearman ranking curves | `scripts/make_report.py` | `results/figures/fig2_ranking__<split>.png` |
| Table 0 — split leakage + NN-similarity audit | `scripts/audit_splits.py` | `results/tables/table0_split_audit.csv` |

## Reproduction order

```bash
python scripts/prepare_openbind.py                     # -> eva71_2a.csv
python scripts/build_splits.py --target eva71_2a       # -> splits/, asserts no leakage
python scripts/audit_splits.py                         # -> table0
python scripts/run_arms.py --arms B0 B1 B2 T1 --splits scaffold random butina
python scripts/run_arms.py --arms T2 --splits scaffold   # fine-tune, ~2 h on CPU
python scripts/make_report.py --require-seeds 10         # -> tables + figures
```

## Excluded from the paper

`results/legacy_prior_run/` — two single-seed metrics files from an earlier
pipeline, computed on the complex-level table. Not comparable; see the README
in that directory.
