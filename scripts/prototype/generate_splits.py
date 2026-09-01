import os
import json
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from collections import defaultdict

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split(df, smiles_col='SMILES', frac_train=0.8, frac_val=0.1, random_state=42):
    """
    Splits the dataframe based on Murcko Scaffolds to prevent scaffold leakage.
    """
    scaffolds = defaultdict(list)
    for i, row in df.iterrows():
        scaffold = generate_scaffold(row[smiles_col])
        scaffolds[scaffold].append(i)
        
    scaffold_sets = list(scaffolds.values())
    # Sort by size (largest scaffolds first) for reproducible distribution
    scaffold_sets.sort(key=lambda x: (len(x), x[0]), reverse=True)
    
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = int(frac_train * len(df))
    val_cutoff = int((frac_train + frac_val) * len(df))
    
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
            
    return train_idx, val_idx, test_idx

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    master_csv = PROJECT_ROOT / "data" / "processed" / "master.csv"
    
    if not master_csv.exists():
        print(f"File {master_csv} not found. Waiting for data processing to complete.")
        return
        
    df = pd.read_csv(master_csv)
    if 'SMILES' not in df.columns:
        print("SMILES column missing in master.csv")
        return
        
    # Generate Scaffold Splits
    print("Generating Scaffold Splits...")
    train_idx, val_idx, test_idx = scaffold_split(df)
    
    splits = {
        "scaffold": {
            "train": [int(i) for i in df.iloc[train_idx].index],
            "val": [int(i) for i in df.iloc[val_idx].index],
            "test": [int(i) for i in df.iloc[test_idx].index],
        }
    }
    
    # Generate Random Splits as baseline
    print("Generating Random Splits...")
    indices = df.index.tolist()
    train_rand, temp = train_test_split(indices, test_size=0.2, random_state=42)
    val_rand, test_rand = train_test_split(temp, test_size=0.5, random_state=42)
    
    splits["random"] = {
        "train": [int(i) for i in train_rand],
        "val": [int(i) for i in val_rand],
        "test": [int(i) for i in test_rand],
    }
    
    interim_dir = PROJECT_ROOT / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    with open(interim_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=4)
        
    print(f"Splits successfully saved to {interim_dir / 'splits.json'}")

if __name__ == "__main__":
    main()
