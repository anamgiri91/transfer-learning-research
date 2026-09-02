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
| Per-compound predictions | `scripts/run_arms.py --save-preds` | `results/predictions/<arm>__<split>__seed<N>__n<size>.npz` | as above |

## Tables and figures

| Manuscript object | Produced by | Output file |
|---|---|---|
| Table 1 — learning curves | `scripts/make_report.py` | `results/tables/table1_learning_curves__<split>.csv` |
| Table 2 — data-efficiency ratio | `scripts/make_report.py` | `results/tables/table2_der__<split>.csv` |
| Table 3 — paired tests vs baseline | `scripts/make_report.py` | `results/tables/table3_paired_tests__<split>.csv` |
| Table 4 — split difficulty (R², skill vs B0) | `scripts/make_report.py` | `results/tables/table4_split_difficulty.csv` |
| Table 5 — contamination upper bound | `scripts/measure_contamination.py` | `results/tables/table5_contamination.csv` |
| Tuning ablation table (§6.2) | `scripts/tune_arms.py` + `render_manuscript_tables.py` | `results/tuned_metrics/*.json` |
| Figure 1 — RMSE learning curves | `scripts/make_report.py` | `results/figures/fig1_learning_curves__<split>.png` |
| Figure 2 — Spearman ranking curves | `scripts/make_report.py` | `results/figures/fig2_ranking__<split>.png` |
| Table 0 — split leakage + NN-similarity audit | `scripts/audit_splits.py` | `results/tables/table0_split_audit.csv` |
| Table 6 — per-stratum performance (§6.3) | `scripts/analyse_cliffs.py` | `results/tables/table6_activity_cliffs.csv` |
| Table 7 — dataset cliff-pair census (§6.3) | `scripts/analyse_cliffs.py` | `results/tables/table7_cliff_pairs.csv` |
| Table 8 — per-stratum paired tests (§6.3) | `scripts/analyse_cliffs.py` | `results/tables/table8_cliff_paired.csv` |

## Verification

| Check | Script | What it guarantees |
|---|---|---|
| Tables are generated, not typed | `scripts/render_manuscript_tables.py` | Every results table in the manuscript is written from `results/tables/*.csv`; `--check` fails if stale |
| Prose numbers are re-derived | `scripts/verify_manuscript.py` | Every numeric claim re-computed from artefacts; non-zero exit on mismatch. The *number* of claims is not restated here — it is asserted against manuscript §10 by the script itself, so there is one place for it to drift and it is checked |
| Re-running reproduces committed metrics | `scripts/run_arms.py --save-preds` | Re-runs each completed cell and fails if any stored metric moves by more than 1e-9 |
| Cliff stratification edge cases | `tests/test_cliffs.py` | `distant` is never merged into `smooth`; only training compounds can create a cliff |
| Split integrity | `tests/test_splits.py` | No compound or scaffold straddles train/test |
| All of the above | `make verify` | Runs tests + freshness check + claim verification |

Bit-reproducibility was verified by re-running each stage and comparing
checksums: `eva71_2a.csv` and all 30 split files are byte-identical, and
re-running B1/B2/T1/T2 cells reproduces their metrics exactly.

## Reproduction order

```bash
python scripts/prepare_openbind.py                     # -> eva71_2a.csv
python scripts/build_splits.py --target eva71_2a       # -> splits/, asserts no leakage
python scripts/audit_splits.py                         # -> table0
python scripts/run_arms.py --arms B0 B1 B2 T1 --splits scaffold random butina
python scripts/run_arms.py --arms T2 --splits scaffold   # fine-tune, ~2 h on CPU
python scripts/make_report.py --require-seeds 10         # -> tables + figures
python scripts/measure_contamination.py                  # -> table5 (PubChem lookup)
python scripts/tune_arms.py --arms B1 B2 T1 --trials 32  # -> tuned_metrics/
python scripts/tune_arms.py --arms T2 --sizes 50 --trials 6
python scripts/run_arms.py --arms B0 B1 B2 T1 --splits scaffold --save-preds
python scripts/run_arms.py --arms T2 --splits scaffold --sizes 347 --save-preds
python scripts/analyse_cliffs.py                         # -> tables 6, 7, 8
python scripts/render_manuscript_tables.py               # -> manuscript tables
python scripts/verify_manuscript.py                      # -> checks every claim
```

## Excluded from the paper

`results/legacy_prior_run/` — two single-seed metrics files from an earlier
pipeline, computed on the complex-level table. Not comparable; see the README
in that directory.
