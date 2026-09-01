import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from copy import deepcopy

class AffinityDataset(Dataset):
    def __init__(self, smiles_list, pkd_list):
        self.smiles = list(smiles_list)
        self.pkd = list(pkd_list)
        
    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        return self.smiles[idx], self.pkd[idx]

def train_chemberta(model, train_loader, val_loader, device, epochs=30, lr=1e-3, weight_decay=1e-2):
    # Only optimize parameters that require gradients
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    # Using L1 loss / Huber loss is sometimes more stable for affinity prediction than pure MSE
    criterion = nn.HuberLoss()
    
    best_loss = float('inf')
    best_model_state = None
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for smiles, targets in train_loader:
            targets = torch.tensor(targets, dtype=torch.float32).to(device)
            inputs = model.prepare_batch(smiles, device)
            
            optimizer.zero_grad()
            preds = model(**inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for smiles, targets in val_loader:
                targets = torch.tensor(targets, dtype=torch.float32).to(device)
                inputs = model.prepare_batch(smiles, device)
                preds = model(**inputs)
                val_loss += criterion(preds, targets).item()
                
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_chemberta(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for smiles, targets in test_loader:
            inputs = model.prepare_batch(smiles, device)
            preds = model(**inputs)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    pearson_r, _ = pearsonr(all_targets, all_preds)
    
    return {
        "RMSE": float(rmse),
        "R2": float(r2),
        "PearsonR": float(pearson_r)
    }

def train_schnet(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()
    
    best_loss = float('inf')
    best_model_state = None
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.z, batch.pos, batch.batch)
            loss = criterion(preds.view(-1), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.z, batch.pos, batch.batch)
                val_loss += criterion(preds.view(-1), batch.y.view(-1)).item()
                
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_schnet(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch.z, batch.pos, batch.batch)
            
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_targets.extend(batch.y.view(-1).cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if len(all_targets) < 2:
        return {"RMSE": float('nan'), "R2": float('nan'), "PearsonR": float('nan')}
        
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    pearson_r, _ = pearsonr(all_targets, all_preds)
    
    return {
        "RMSE": float(rmse),
        "R2": float(r2),
        "PearsonR": float(pearson_r)
    }
