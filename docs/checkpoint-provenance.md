# ChemBERTa checkpoint attribution and reproducible identity

Recorded 2026-10-05 for the accompanying repository and review materials.

| Item | Evidence or status |
|---|---|
| Evaluated checkpoint | `DeepChem/ChemBERTa-77M-MTR` |
| Checkpoint and tokenizer revision | `66b895cab8adebea0cb59a8effa66b2020f204ca` |
| Executable pin | `src/evapro/models/pretrained.py`; all executed ChemBERTa loaders use `checkpoint_kwargs` |
| Model-card discrepancy | The [model page](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR) displays “No model card”; the [pinned file listing](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/tree/66b895cab8adebea0cb59a8effa66b2020f204ca) contains weights, configuration and tokenizer files but no README/model card |
| Paper attribution | Inferred to be the ChemBERTa-2 MTR release, based on the checkpoint name and the multi-task regression objective described by [Ahmad et al.](https://arxiv.org/abs/2209.01712); the model card does not confirm the mapping |
| Maintainer confirmation | Not obtained; no correspondence is claimed |
| Distribution | This record is included in the local source repository and review materials. Public deposition of the revised materials remains pending |

The manuscript identifies the evaluated artifact independently of its inferred
paper label. Reproduction uses the pinned repository and revision, not a moving
model name or an assumed equivalence between releases. A future maintainer
clarification should be recorded with its source and date; a different checkpoint
would constitute a new evaluation rather than an undocumented substitution.

This record documents the discrepancy; it does not reconstruct undisclosed
pretraining-corpus membership or establish the absence of contamination.
