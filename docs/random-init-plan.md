# Proposed matched random-initialization control

Prepared 2026-10-04 after the existing results were inspected. **Not executed.**
This would be an amended experiment, not an original preregistered analysis.
The current manuscript remains scoped to the experiments already evaluated.

## Design to freeze before evaluation

- Arm `T0ft`: fully trainable randomly initialized ChemBERTa backbone, with
  the pinned 77M-MTR architecture and tokenizer; no pretrained encoder weights.
- Comparator: existing `T2v`, using the same `scripts/run_finetune.py` training
  procedure, mean-pooled linear readout and AdamW optimizer.
- Scaffold split only; seeds 0–9; budgets 50, 100, 250 and 347: 40 cells.
- Exactly the same training subsamples, 15% internal validation partition,
  test rows, tokenization, batches, maximum 60 epochs and patience 10 as T2v.
- Select learning rate from {1e-5, 3e-5, 1e-4} on validation RMSE and restore
  the best validation checkpoint. No new hyperparameter search after inspecting
  test results. This estimates pretraining benefit under this shared procedure;
  it does not establish the optimal achievable from-scratch transformer score.
- Primary outcome: paired RMSE difference `T0ft - T2v`, positive meaning lower
  error for pretrained initialization. Report median paired difference, a
  seed-bootstrap interval, all per-seed scores and two-sided signed-rank p.
- Freeze one four-comparison family, one contrast per budget, corrected by
  Holm. Report secondary endpoints descriptively. Do not promote them after
  seeing their results or add them to old confirmatory families retroactively.
- No significance-based stopping. Require all ten seeds before reporting a
  budget's contrast. Log every failure; reruns retain the frozen settings.
- Before the sweep, test architecture equality, trainability, absence of
  pretrained encoder tensors, train/validation/test pairing and non-overwriting
  of T2v artifacts. Record the source revision and environment for all new runs.

Expected cost from the existing T2v path is roughly 3–4 CPU-hours, but random
initialization can change early-stopping times. The actual cost must be measured.
No results or conclusion about this arm are included in the manuscript.
