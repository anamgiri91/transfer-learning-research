# results/figures — regenerated, not tracked

Produced by `scripts/make_report.py` and gitignored, because they are fully
derived from `results/tables/` and `results/metrics/`, which are tracked.

```bash
python scripts/make_report.py --require-seeds 10
```

Verified 2026-09-02: all nine figures are **byte-identical** across
regeneration.

## Current figures

File names describe content; the manuscript numbers them by order of
appearance, so the two do not line up one-to-one. `verify_citations.py` checks
that every embedded image exists and that the visible Figure numbers run 1..N.

| File | Manuscript | Content |
|---|---|---|
| `fig1_learning_curves__scaffold.png` | Figure 1 | RMSE vs training-set size, median + IQR, primary endpoint |
| `fig2_ranking__scaffold.png` | Figure 2 | Spearman ρ vs training-set size |
| `fig1_learning_curves__random.png` | Figure 3 | as Figure 1, random split (optimism reference) |
| `fig1_learning_curves__butina.png` | Figure 4 | as Figure 1, Butina split (stricter check) |
| `fig_indomain__scaffold.png` | Figure 5 | §6.4 in-domain vs generic, focused comparison |
| `fig_indomain__random.png` | — | generated, not embedded |
| `fig_indomain__butina.png` | — | generated, not embedded |
| `fig2_ranking__random.png` | — | generated, not embedded |
| `fig2_ranking__butina.png` | — | generated, not embedded |

**Two deliberate restrictions.** Figures 1–4 show only the five arms frozen in
`plan.md` §4: adding the §6.4 arms put eight series into a 0.17 RMSE band with
overlapping ribbons and detached labels. Figure 5 carries the in-domain
comparison instead, and omits `T2`, whose n = 50 RMSE of 1.23 stretches the
axis until the arms it exists to separate become indistinguishable.

The two `data_efficiency_*.png` files from the prototype pipeline were deleted
on 2026-09-02: they came from a single-seed, complex-level run (archived in
`results/legacy_prior_run/`), were never referenced by the manuscript, and were
not comparable to anything current.
