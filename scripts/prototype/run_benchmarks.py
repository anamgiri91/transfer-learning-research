import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from src.evapro.features.fingerprints import get_morgan_fingerprint
from src.evapro.models.baselines import RandomForestBaseline

def prepare_dataset(df, idx_list):
    """Convert dataframe subset into ML-ready arrays."""
    subset = df.loc[idx_list]
    
    X = []
    y = []
    valid_indices = []
    
    for i, row in subset.iterrows():
        fp = get_morgan_fingerprint(row['SMILES'])
        if fp is not None:
            X.append(fp)
            y.append(row['pKD'])
            valid_indices.append(i)
            
    return np.array(X), np.array(y)

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    master_csv = PROJECT_ROOT / "data" / "processed" / "master.csv"
    splits_json = PROJECT_ROOT / "data" / "interim" / "splits.json"
    
    if not master_csv.exists() or not splits_json.exists():
        print("Required data files not found. Run 'make splits' first.")
        return
        
    df = pd.read_csv(master_csv)
    # Ensure index matches what is stored in splits
    df.index = df.index.astype(int)
    
    with open(splits_json, 'r') as f:
        splits = json.load(f)
        
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # We will test over learning budgets
    budgets = [50, 100, 250, 500, "full"]
    
    train_idx = splits['scaffold']['train']
    val_idx = splits['scaffold']['val']
    test_idx = splits['scaffold']['test']
    
    # ---------------------------------------------------------
    # 1. Random Forest Baseline
    # ---------------------------------------------------------
    # print("--- Running Random Forest Baseline on Scaffold Split ---")
    # rf_results = {}
    # for budget in budgets:
    #     print(f"\nRF Budget: {budget}")
    #     current_train_idx = train_idx if budget == "full" else train_idx[:budget]
            
    #     X_train, y_train = prepare_dataset(df, current_train_idx)
    #     X_test, y_test = prepare_dataset(df, test_idx)
        
    #     if len(X_train) == 0:
    #         continue
            
    #     model = RandomForestBaseline(n_estimators=200, random_state=42)
    #     model.train(X_train, y_train)
        
    #     metrics = model.evaluate(X_test, y_test)
    #     rf_results[str(budget)] = metrics
        
    # with open(metrics_dir / "rf_ecfp4__scaffold__seed42.json", "w") as f:
    #     json.dump(rf_results, f, indent=4)
        
    # ---------------------------------------------------------
    # 2. ChemBERTa Transfer Learning
    # ---------------------------------------------------------
    # print("\n--- Running ChemBERTa on Scaffold Split ---")
    # import torch
    # from torch.utils.data import DataLoader
    # from src.evapro.models.chemberta import ChemBERTaRegressor
    # from src.evapro.evaluation.trainer import AffinityDataset, train_chemberta, evaluate_chemberta
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # print(f"Using device for ChemBERTa: {device}")
    
    # # Validation and Test sets are constant
    # val_subset = df.loc[val_idx]
    # test_subset = df.loc[test_idx]
    
    # val_dataset = AffinityDataset(val_subset['SMILES'].values, val_subset['pKD'].values)
    # test_dataset = AffinityDataset(test_subset['SMILES'].values, test_subset['pKD'].values)
    
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # chemberta_results = {}
    
    # for budget in budgets:
    #     print(f"\nChemBERTa Budget: {budget}")
    #     current_train_idx = train_idx if budget == "full" else train_idx[:budget]
    #     train_subset = df.loc[current_train_idx]
        
    #     train_dataset = AffinityDataset(train_subset['SMILES'].values, train_subset['pKD'].values)
    #     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
    #     model = ChemBERTaRegressor(model_name="DeepChem/ChemBERTa-10M-MTR")  # Using a smaller model for faster benchmarking iteration
        
    #     model = train_chemberta(model, train_loader, val_loader, device)
    #     metrics = evaluate_chemberta(model, test_loader, device)
        
    #     print(f"Results: {metrics}")
    #     chemberta_results[str(budget)] = metrics
        
    # with open(metrics_dir / "chemberta__scaffold__seed42.json", "w") as f:
    #     json.dump(chemberta_results, f, indent=4)
        
    # ---------------------------------------------------------
    # 3. SchNet 3D Geometric Deep Learning
    # ---------------------------------------------------------
    print("\n--- Running SchNet on Scaffold Split ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for SchNet: {device}")
    
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from src.evapro.features.graph3d import sdf_to_pyg_data
    from src.evapro.models.geometric import SchNetRegressor
    from src.evapro.evaluation.trainer import train_schnet, evaluate_schnet
    import os
    
    class PyGDataset(torch.utils.data.Dataset):
        def __init__(self, df, root_dir):
            self.df = df
            self.root = root_dir
            self.data_list = []
            
            for _, row in df.iterrows():
                group = row['compound_group']
                cname = row['Complex']
                # Search for the ligand sdf
                sdf_path1 = self.root / group / cname / f"{cname}_ligand_prepared.sdf"
                sdf_path2 = self.root / group / cname / f"{cname}_ligand_ref.sdf"
                
                sdf_path = sdf_path1 if sdf_path1.exists() else (sdf_path2 if sdf_path2.exists() else None)
                if sdf_path:
                    data = sdf_to_pyg_data(str(sdf_path), row['pKD'])
                    if data is not None:
                        self.data_list.append(data)
                        
        def __len__(self):
            return len(self.data_list)
            
        def __getitem__(self, idx):
            return self.data_list[idx]

    structures_dir = PROJECT_ROOT / "data" / "raw" / "OpenBind_EV-A71_2A" / "structures"
    
    val_subset = df.loc[val_idx]
    test_subset = df.loc[test_idx]
    
    val_pyg = PyGDataset(val_subset, structures_dir)
    test_pyg = PyGDataset(test_subset, structures_dir)
    
    if len(val_pyg) > 0 and len(test_pyg) > 0:
        val_loader_pyg = PyGDataLoader(val_pyg, batch_size=16, shuffle=False)
        test_loader_pyg = PyGDataLoader(test_pyg, batch_size=16, shuffle=False)
        
        schnet_results = {}
        
        for budget in budgets:
            print(f"\nSchNet Budget: {budget}")
            current_train_idx = train_idx if budget == "full" else train_idx[:budget]
            train_subset_pyg = df.loc[current_train_idx]
            
            train_pyg = PyGDataset(train_subset_pyg, structures_dir)
            if len(train_pyg) == 0:
                continue
                
            train_loader_pyg = PyGDataLoader(train_pyg, batch_size=16, shuffle=True)
            
            # Very small schnet for quick benchmarking
            schnet_model = SchNetRegressor(hidden_channels=32, num_filters=32, num_interactions=2)
            
            schnet_model = train_schnet(schnet_model, train_loader_pyg, val_loader_pyg, device, epochs=15, lr=5e-4)
            metrics = evaluate_schnet(schnet_model, test_loader_pyg, device)
            
            print(f"Results: {metrics}")
            schnet_results[str(budget)] = metrics
            
        with open(metrics_dir / "schnet__scaffold__seed42.json", "w") as f:
            json.dump(schnet_results, f, indent=4)
    else:
        print("Could not load sufficient 3D PDB/SDF poses for SchNet benchmarking.")
        
    print(f"\nSaved all metrics to {metrics_dir}")

if __name__ == "__main__":
    main()
