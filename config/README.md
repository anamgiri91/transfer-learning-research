# `config/` — superseded, kept as history

**These YAML files do not describe the study that was run.** They were written
on 2026-09-01, before the data was inspected, and were driven by
`scripts/run_benchmark.py` — now in `scripts/prototype/`, retired on 2026-09-11
(`plan.md` Amendment 3). Neither is on any path that produces a number in the
paper.

Where they contradict the executed study:

| These configs say | The study |
|---|---|
| `data.target: eva71_3c` | `eva71_2a` — 3C has no dataset (Amendment 1) |
| `B1` is `lightgbm` | scikit-learn `HistGradientBoostingRegressor`; LightGBM cannot load here |
| `arm_t1_chemlm_ft` = full fine-tune, `arm_t2_chemlm_probe` = probe | **transposed** in the executed study: `T1` is the probe, `T2` the fine-tune (Amendment 2) |
| `arm_b3_dmpnn`, `representation: {type: graph}` | `B3` was never run; no `graph` representation exists in `evapro.features` |
| `arm_t4_indomain` is a D-MPNN | the executed `T4` is a transformer encoder, pretrained then **frozen and probed**, not fine-tuned |
| `curve.train_sizes: [50, 100, 250, 500, null]` | `500` exceeds the 347-compound training fold; the sizes run are 50/100/250/347 |
| `tuning.n_trials: 32` for every arm | delivered to `B1` only; the fine-tune got 6 trials at n=50 and 4 at full data (§6.2, limitation 4) |
| `eval.bootstrap_resamples` | the reported CIs resample **seeds**, not test rows |

The executed pipeline is the command sequence in `README.md`, driven by
`scripts/run_arms.py`, whose arm table is the authority on what each arm does.

Repair these against the executed study or delete them; leaving them here
undocumented is the one option that misleads.
