# data/processed — model-ready

`eva71_3c.parquet`, `eva71_2a.parquet`, plus `splits/*.json` mapping InChIKey →
fold for each split strategy. Splits are files, not code paths: every arm reads
the same JSON so no experiment can quietly reshuffle its own test set.
