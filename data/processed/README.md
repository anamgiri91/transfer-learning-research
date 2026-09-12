# data/processed — model-ready

| File | What it is |
|---|---|
| `master.csv` | the OpenBind release flattened to one row per crystal complex |
| `eva71_2a.csv` | the evaluation set: 494 compounds, one row each, `pactivity` = pK_D |
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
