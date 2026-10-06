# data/processed — model-ready

| File | What it is |
|---|---|
| `master.csv` | labelled complex rows rebuilt from the pinned OpenBind ZIP (925 → 649) |
| `eva71_2a.csv` | the evaluation set: 494 compounds, one row each, `pactivity` = pK_D |
| `eva71_2a.label_audit.json` | label reuse across complexes; independent assay replication is not identifiable |
| `eva71_2a.curation.json` | the curation funnel — counts in, dropped, out |
| `indomain_3c.csv` | the §6.4 in-domain pretraining corpus, 2,974 measurements |
| `indomain_3c.curation.json` | its funnel, including the overlap flags §6.5 uses |
| `splits/eva71_2a/<strategy>__seed<N>.json` | InChIKey → fold, 30 files |

**There is no EV-A71 3C dataset here, and there never was.** `plan.md` was
written for 3C; Amendment 1 records that the available data is 2A. Nothing in
this directory is a parquet file either — the curated sets are CSV so that a
change in a committed number shows up as a reviewable diff.

Splits are files, not code paths: every arm reads the same JSON so no experiment
can quietly reshuffle its own test set.

The legacy `replicate_spread` column measures label disagreement across crystal
rows. It is not an estimate of independent assay replication or measurement noise.
