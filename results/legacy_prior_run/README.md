# Prior run (preserved, not used in the paper)

Two metrics files predating the current pipeline, kept for provenance:
`rf_ecfp4__scaffold__seed42.json`, `chemberta__scaffold__seed42.json`.

They are excluded from `scripts/make_report.py` aggregation because they are not
comparable to the current results:

- Computed on the **complex-level** table (649 rows), not the compound-level one
  (494 unique compounds). The random split at complex level leaks 22 compounds
  between train and test.
- Single seed (42) only, so no variance estimate.
- Different JSON schema (`{size: {RMSE, R2, PearsonR}}`).
- Train sizes include 500, which exceeds the compound-level train fold (347).

Their headline signal — **negative R² at every training size for both models** —
is consistent with the current results and is discussed in the paper as
motivation, not as a reported benchmark number.
