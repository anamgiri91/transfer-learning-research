"""Multitask in-domain pretraining on related 3C / 3C-like proteases (plan.md §4, T4/T5).

A shared SMILES encoder with one linear head per protease target, trained to
regress pActivity. The point of the multitask setup rather than pooling all
targets into one column is that the targets are different enzymes measured in
different labs: pooling them would ask the model to predict a quantity that
does not exist, whereas separate heads let a shared representation form while
each target keeps its own scale and offset.

Two initialisations, which is what separates the two arms:
  T4  random init  -- in-domain pretraining *instead of* generic pretraining.
                      This is the arm plan.md §4 specifies, and the H3 contrast
                      against T1 (generic pretraining, frozen probe).
  T5  ChemBERTa    -- in-domain adaptation *on top of* generic pretraining;
                      plan.md's "chained" arm, and the configuration §2 reports
                      the literature credits for most real transfer gains.

Both are evaluated exactly as T1 is -- frozen encoder, mean-pooled, ridge probe
-- so any difference is attributable to the pretraining, not the adaptation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

CHEMBERTA = "DeepChem/ChemBERTa-77M-MTR"
MAX_LEN = 128


class MultitaskRegressor(nn.Module):
    """Shared encoder, one linear head per target."""

    def __init__(self, task_names: list[str], pretrained: bool,
                 model_name: str = CHEMBERTA):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            # Same architecture and tokenizer, random weights: the comparison
            # against T5 is then purely about what pretraining contributed.
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        h = self.encoder.config.hidden_size
        self.task_names = list(task_names)
        self.heads = nn.ModuleDict({t: nn.Linear(h, 1) for t in self.task_names})

    def embed(self, input_ids, attention_mask):
        """Mean-pooled token states -- identical pooling to arm T1."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1)

    def forward(self, input_ids, attention_mask, task_idx):
        z = self.embed(input_ids, attention_mask)
        # One head per row; gather each row's prediction from its own task head.
        preds = torch.stack([h(z).squeeze(-1) for h in self.heads.values()], dim=1)
        return preds.gather(1, task_idx.unsqueeze(1)).squeeze(1)


def tokenize(smiles, model_name: str = CHEMBERTA, max_len: int = MAX_LEN):
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(list(smiles), padding="max_length", truncation=True,
              max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def embed_smiles(model: MultitaskRegressor, smiles, batch_size: int = 64,
                 model_name: str = CHEMBERTA) -> np.ndarray:
    """Frozen-encoder embeddings for the downstream ridge probe."""
    model.eval()
    ids, mask = tokenize(smiles, model_name)
    out = []
    for i in range(0, len(ids), batch_size):
        out.append(model.embed(ids[i:i + batch_size], mask[i:i + batch_size]).numpy())
    return np.vstack(out).astype(np.float32)
