# results/figures — regenerated, not tracked

All figures here are produced by `scripts/make_report.py` and are gitignored
because they are fully derived from `results/tables/` and `results/metrics/`,
which are tracked. Regenerate with:

```bash
python scripts/make_report.py --require-seeds 10
```

Verified byte-identical across regeneration.

## Current figures

| File | Manuscript ref | Content |
|---|---|---|
| `fig1_learning_curves__<split>.png` | Figure 1 | RMSE vs training-set size, median + IQR band |
| `fig2_ranking__<split>.png` | Figure 2 | Spearman ρ vs training-set size |

`<split>` ∈ {scaffold, random, butina}. The manuscript uses the scaffold
versions as the primary endpoint.

## Stale — not from this pipeline

`data_efficiency_rmse.png` and `data_efficiency_pearsonr.png` were produced by
`scripts/prototype/generate_reports.py` from the single-seed, complex-level
run now archived in `results/legacy_prior_run/`. They are **not** referenced by
the manuscript and are not comparable to the current figures. Delete them, or
regenerate the prototype outputs, but do not read them as current results.
