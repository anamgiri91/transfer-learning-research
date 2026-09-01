import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    rf_file = metrics_dir / "rf_ecfp4__scaffold__seed42.json"
    chemberta_file = metrics_dir / "chemberta__scaffold__seed42.json"
    schnet_file = metrics_dir / "schnet__scaffold__seed42.json"
    
    records = []
    
    if rf_file.exists():
        with open(rf_file, 'r') as f:
            rf_data = json.load(f)
            for budget, metrics in rf_data.items():
                records.append({
                    "Model": "Random Forest (ECFP4)",
                    "Budget": 649 if budget == "full" else int(budget),
                    "PearsonR": metrics["PearsonR"],
                    "RMSE": metrics["RMSE"]
                })
                
    if chemberta_file.exists():
        with open(chemberta_file, 'r') as f:
            cb_data = json.load(f)
            for budget, metrics in cb_data.items():
                records.append({
                    "Model": "ChemBERTa (2D Pretrained)",
                    "Budget": 649 if budget == "full" else int(budget),
                    "PearsonR": metrics["PearsonR"],
                    "RMSE": metrics["RMSE"]
                })
                
    if schnet_file.exists():
        with open(schnet_file, 'r') as f:
            sn_data = json.load(f)
            for budget, metrics in sn_data.items():
                records.append({
                    "Model": "SchNet (3D GNN)",
                    "Budget": 649 if budget == "full" else int(budget),
                    "PearsonR": metrics["PearsonR"],
                    "RMSE": metrics["RMSE"]
                })
                
    if not records:
        print("No metrics found to plot. Run benchmarks first.")
        return
        
    df = pd.DataFrame(records)
    
    # 1. Plot Data Efficiency Curve (Pearson R)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Budget", y="PearsonR", hue="Model", marker="o", linewidth=2)
    plt.title("Data Efficiency on EV-A71 Scaffold Split (Pearson R)")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Pearson Correlation Coefficient (R)")
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / "data_efficiency_pearsonr.png", dpi=300)
    plt.close()
    
    # 2. Plot Data Efficiency Curve (RMSE)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Budget", y="RMSE", hue="Model", marker="o", linewidth=2)
    plt.title("Data Efficiency on EV-A71 Scaffold Split (RMSE)")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Root Mean Square Error (pKD)")
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / "data_efficiency_rmse.png", dpi=300)
    plt.close()
    
    print(f"Figures successfully generated in {figures_dir}")

if __name__ == "__main__":
    main()
