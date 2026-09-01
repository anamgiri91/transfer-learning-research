import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ChemBERTaRegressor(nn.Module):
    """
    ChemBERTa-based model for predicting affinity (pKD) from SMILES.
    Using DeepChem's pretrained ChemBERTa model.
    """
    def __init__(self, model_name="DeepChem/ChemBERTa-10M-MTR", dropout=0.2, freeze_base=True):
        super(ChemBERTaRegressor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze the last layer
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
                
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output).squeeze(-1)

    def prepare_batch(self, smiles_list, device):
        """Tokenize a list of SMILES and move to device."""
        inputs = self.tokenizer(
            smiles_list, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device)
        }
