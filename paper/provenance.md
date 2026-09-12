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
| Split assignments | `scripts/build_splits.py` | `data/processed/splits/eva71_2a/{scaffold,butina,random}__seed{0..9}.json` (30 files) | `eva71_2a.csv` |
| Per-run metrics | `scripts/run_arms.py` | `results/metrics/<arm>__<split>__seed<N>__n<size>.json` | curated set + split files |
| Per-compound predictions | `scripts/run_arms.py --save-preds` | `results/predictions/<arm>__<split>__seed<N>__n<size>.npz` | as above |
| In-domain corpus (raw) | `scripts/fetch_indomain.py` | `data/raw/indomain_<target>.csv` + `.manifest.json` | ChEMBL activity API, 8 targets |
| In-domain corpus (curated) | `scripts/prepare_indomain.py` | `data/processed/indomain_3c.csv`, `.curation.json` | the 8 raw pulls + `eva71_2a.csv` for overlap flags |
| 2A surrogate divergence | `scripts/verify_surrogate.py` | `results/tables/table9_surrogate_divergence.csv` | UniProt Q66478 / Q65900 / Q9QF31 |
| In-domain encoders | `scripts/pretrain_indomain.py` | `models/indomain_{T4,T5}[_clean].pt` + `.json` | `indomain_3c.csv` |

## Tables and figures

| Manuscript object | Produced by | Output file |
|---|---|---|
| Table 1 — learning curves | `scripts/make_report.py` | `results/tables/table1_learning_curves__<split>.csv` |
| Table 2 — data-efficiency ratio | `scripts/make_report.py` | `results/tables/table2_der__<split>.csv` |
| Table 3 — paired tests vs baseline | `scripts/make_report.py` | `results/tables/table3_paired_tests__<split>.csv` |
| Table 4 — split difficulty (R², skill vs B0) | `scripts/make_report.py` | `results/tables/table4_split_difficulty.csv` |
| Table 5 — contamination upper bound | `scripts/measure_contamination.py` | `results/tables/table5_contamination.csv` |
| Table 0 — split leakage + NN-similarity audit | `scripts/audit_splits.py` | `results/tables/table0_split_audit.csv` |
| Table 6 — per-stratum performance (§6.3) | `scripts/analyse_cliffs.py` | `results/tables/table6_activity_cliffs.csv` |
| Table 7 — dataset cliff-pair census (§6.3) | `scripts/analyse_cliffs.py` | `results/tables/table7_cliff_pairs.csv` |
| Table 8 — per-stratum paired tests (§6.3) | `scripts/analyse_cliffs.py` | `results/tables/table8_cliff_paired.csv` |
| Table 9 — CVA16/EV-A71 2A divergence (§3.1) | `scripts/verify_surrogate.py` | `results/tables/table9_surrogate_divergence.csv` |
| Table 10 — in-domain and decontamination contrasts (§6.4, §6.5) | `scripts/analyse_indomain.py` | `results/tables/table10_indomain_contrasts.csv` |
| Table 11 — untrained-encoder random draws (§6.4) | `scripts/analyse_indomain.py` | `results/tables/table11_random_encoder_draws.csv` |
| Table 12 — corpus vs evaluation-set chemistry (§6.4) | `scripts/prepare_indomain.py` | `results/tables/table12_corpus_chemistry.csv` |
| Table 13 — tuned fine-tune vs both baseline bases (§6.2) | `scripts/analyse_tuning.py` (over `results/tuned_metrics/*.json` from `scripts/tune_arms.py`) | `results/tables/table13_tuned_comparison.csv` |
| Table 14 — every pre-registered endpoint (§5.7) | `scripts/analyse_endpoints.py` | `results/tables/table14_all_endpoints.csv` |
| Table 15 — H2's interaction term (§5.3) | `scripts/analyse_h2.py` | `results/tables/table15_h2_interaction.csv` |
| Table 16 — DER per seed, with censoring (§5.3) | `scripts/analyse_h2.py` | `results/tables/table16_der_uncertainty.csv` |
| Table 17 — amended fine-tuning arms (§6.6) | `scripts/run_finetune.py` → `analyse_amended.py` | `results/tables/table17_amended_finetune.csv` *(pending: 15 of 60 cells)* |
| Table 18 — amended contrasts, incl. H3's pre-registered form (§6.6) | `scripts/analyse_amended.py` | `results/tables/table18_amended_contrasts.csv` *(pending)* |
| Supplied-input checksums | `scripts/verify_reproducibility.py --write-input-manifest` | `docs/input-checksums.json` |
| Reproduction coverage, written by the passing run | `scripts/verify_reproducibility.py --tier full` | `docs/reproduction-coverage.json`, `docs/reproduction-coverage.md` |
| Figures 1, 3, 4 — RMSE learning curves | `scripts/make_report.py` | `results/figures/fig1_learning_curves__<split>.png` |
| Figure 2 — Spearman ranking curves | `scripts/make_report.py` | `results/figures/fig2_ranking__<split>.png` |
| Figure 5 — in-domain vs generic (§6.4) | `scripts/make_report.py` | `results/figures/fig_indomain__<split>.png` |

The manuscript numbers figures by order of appearance, so the file names and the
visible Figure numbers do not line up one-to-one; `results/figures/README.md`
carries the mapping and `scripts/verify_citations.py` checks that every embedded
image exists and that the visible numbers run 1..N.

## Verification

| Check | Script | What it guarantees |
|---|---|---|
| Tables are generated, not typed | `scripts/render_manuscript_tables.py` | Every results table in the manuscript is written from `results/tables/*.csv`; `--check` fails if stale |
| Which prose numbers are derived at all | `scripts/audit_number_coverage.py` | Reporting tool: classifies every numeric token in the prose as covered by a claim, structural, or UNCOVERED. Written after the compute costs ("~160 s", "~80x") survived the whole project unmeasured; it cut the uncovered set from 26 to 11 |
| Summaries match the artefacts | `scripts/verify_consistency.py` | Counts stated in prose (runs, arms, claims, tables, figures) must equal the derived value in every document; sentinels fire when a summary still says work was not done that an artefact shows was done. Added because all eleven errors of the second audit were in prose that restates rather than computes |
| Citations are checked | `scripts/verify_citations.py` | Reference numbering, reading-depth labels, orphan references, dangling body citations, depth drift against `literature.md`, and the `[secondary]`-with-a-number rule. `--online` also checks every cited URL resolves. Added after a manual audit found seven prose citation errors that no existing check could see |
| Borrowed numbers carry their source sentence | `docs/citation-claims.yaml` + `scripts/verify_citations.py` | Each externally-sourced quantity records the verbatim quote, the URL it was read at and the date; the checker enforces that the quote contains the number, that the claim is still made, and that second-hand quotes are declared |
| The surrogate claim is re-derived, not cited | `scripts/verify_surrogate.py` | §3.1's CVA16/EV-A71 comparison is recomputed from UniProt-annotated 2A chains into `table9_surrogate_divergence.csv`; 13 claims check against it |
| The citation checker is itself checked | `tests/test_citations.py` | Each of the eight defects the audit found is pinned as a regression test |
| Prose numbers are re-derived | `scripts/verify_manuscript.py` | Every numeric claim re-computed from artefacts; non-zero exit on mismatch. The *number* of claims is not restated here — it is asserted against manuscript §10 by the script itself, so there is one place for it to drift and it is checked |
| Re-running reproduces committed metrics | `scripts/run_arms.py --save-preds` | Re-runs each completed cell and fails if any stored metric moves by more than 1e-9 |
| The whole pipeline reproduces | `scripts/verify_reproducibility.py` (`make verify-repro`) | Runs all fourteen offline stages inside a throwaway git worktree at an immutable ref, watching 1787 committed artefacts and **reconstructs 1495** of them; 21 supplied inputs are verified by SHA-256 rather than rebuilt, 292 are compared only, and 329 (figures, and predictions outside the scaffold split) have no committed baseline. Tolerances are 1e-9 relative for metrics and tables, 1e-6 for torch predictions, and exact for row identities, shapes, missingness and every metric file's identity fields. The working tree is never written to. Not covered: the ChemBERTa fine-tune's 40 cells, the six encoders' pretraining, the 41 tuned cells, and table5 (needs PubChem). |
| Cliff stratification edge cases | `tests/test_cliffs.py` | `distant` is never merged into `smooth`; only training compounds can create a cliff |
| In-domain corpus membership | `tests/test_indomain.py` | Capsid, RNA-polymerase and papain-like assays cannot enter a '3C-like' corpus on target name alone; every exclusion carries a stated reason |
| Split integrity | `tests/test_splits.py` | No compound or scaffold straddles train/test |
| All of the above | `make verify` | Runs tests + freshness check + claim verification + citation checks |
| Plus link liveness | `make verify-online` | As above, and every cited URL must still resolve |

Bit-reproducibility is verified on every `make verify-repro`, not once by hand:
`eva71_2a.csv`, `indomain_3c.csv` and all 30 split files are byte-identical,
the 360 re-fitted baseline metric files match to ~1e-15 relative, and the data
rows of all 23 tables are byte-identical. The nine figures are
regenerated but **not verified**: they are gitignored as regenerable, so a
reconstruction has no committed copy to diff against, and matplotlib PNG output
is not byte-stable across environments in any case. The transfer arms'
metrics are **not** re-fitted by that run — the figures below for T1/T2/T0r/T4/T5
come from the original hand check and from `run_arms.py --save-preds`, which
re-runs a completed cell and fails if a stored metric moves by more than 1e-9. The full stage-level
table is in `manuscript.md` §10, including the one caveat — the random-forest
arm reproduces to ~1e-16 rather than bit-identically, because `n_jobs=-1` lets
the order of the floating-point reduction across threads vary.

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
python scripts/analyse_tuning.py                         # -> table13
python scripts/run_arms.py --arms B0 B1 B2 T1 --splits scaffold --save-preds
python scripts/run_arms.py --arms T2 --splits scaffold --sizes 347 --save-preds
python scripts/analyse_cliffs.py                         # -> tables 6, 7, 8

# In-domain arms T4/T5 (H3), and the decontamination ablation (H4)
python scripts/fetch_indomain.py                         # -> data/raw/indomain_*
python scripts/prepare_indomain.py                       # -> indomain_3c.csv, table12
python scripts/pretrain_indomain.py --arm T4             # random init
python scripts/pretrain_indomain.py --arm T5             # from ChemBERTa
python scripts/pretrain_indomain.py --arm T4 --decontaminate
python scripts/pretrain_indomain.py --arm T5 --decontaminate
python scripts/pretrain_indomain.py --arm T4 --drop-random   # size-matched control
python scripts/pretrain_indomain.py --arm T5 --drop-random   # -- 6.5 turns on it
# T0r/T4/T5 run on all three splits (5.5, 5.7); the ablation arms on scaffold
python scripts/run_arms.py --arms T0r T4 T5 --splits scaffold random butina --save-preds
python scripts/run_arms.py --arms T4c T4r T5c T5r --splits scaffold --save-preds
python scripts/analyse_indomain.py                       # -> tables 10, 11
python scripts/analyse_endpoints.py                      # -> table14
python scripts/analyse_h2.py                             # -> tables 15, 16
python scripts/run_finetune.py --arms T2v                # Amendment 4 (~3.7 h)
python scripts/run_finetune.py --arms T4ft T5ft --sizes 347   # H3 (~3.1 h)
python scripts/analyse_amended.py                        # -> tables 17, 18
python scripts/render_manuscript_tables.py               # -> manuscript tables
make verify                                              # -> every check below
```

Omitting the two `--drop-random` encoders is not a shortcut: §6.5's conclusion
is the size-matched control, and without those runs the same ablation reports a
significant effect with the causal arrow reversed.

## Excluded from the paper

`results/legacy_prior_run/` — two single-seed metrics files from an earlier
pipeline, computed on the complex-level table. Not comparable; see the README
in that directory.
