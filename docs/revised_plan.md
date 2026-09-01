# Revised Research Plan: Benchmarking Transfer Learning on EV-A71 2A Protease

**Target Venue:** ICML AI for Science, NeurIPS Datasets & Benchmarks, or J. Chem. Inf. Model.
**Core Objective:** Systematically evaluate whether large-scale pretraining (sequence, 2D graph, and 3D geometric) improves data efficiency and OOD generalization on a low-N, high-fidelity affinity prediction task (EV-A71 2A protease).

## 1. Experimental Design (Publishable Benchmark)

We will compare models at various fine-tuning data budgets ($N \in \{50, 100, 250, 500, \text{full}\}$).
*   **Baselines (From-Scratch):**
    *   **RF + ECFP4:** Classic chemoinformatics baseline (Random Forest with Morgan fingerprints).
    *   **GCN/GAT (2D):** Standard Graph Neural Network trained without pretraining.
*   **Transfer Learning Arms (Pretrained):**
    *   **ChemBERTa / MolBERT (Sequence):** Fine-tuning a language model trained on millions of SMILES.
    *   **Uni-Mol / MolCLR (2D/3D Graph):** Self-supervised graph pretraining.
    *   **EGNN/SchNet (3D Structural):** Using explicit 3D poses (from the OpenBind structures) pretrained on PDBbind or PCQM4Mv2.

## 2. Dataset Setup
*   **Source:** Zenodo Record 20026661 (`OpenBind_EV-A71_2A.zip`) containing 925 structural events and 601 affinity labels.
*   **Splits:** We will rigorously enforce the official splits (Random, Scaffold/Murcko, Temporal) from the `OpenBind-Consortium/EV-A71_2A_benchmark` repository to prevent data leakage and assess true OOD performance.

## 3. Project Architecture & Tooling

To ensure the reproducibility required for a top-tier venue, we will scaffold the project precisely as outlined in the `README.md`.

*   `src/evapro`: The core modular Python package.
*   `Makefile`: Entrypoints for end-to-end reproducibility (`make data`, `make splits`, `make bench`, `make report`).
*   `config/`: YAML-based configuration for each benchmarking "arm".

## 4. Phase-by-Phase Execution

### Phase 1: Data Engineering & Scaffolding (Current)
*   [x] Initialize the Python package `src/evapro` and setup `pyproject.toml`.
*   [x] Write a `Makefile` covering data fetching and execution pipelines.
*   [x] Fetch the Zenodo dataset (ID: 20026661) into `data/raw`.
*   [x] Clean data (filtering missing affinities, linking SMILES to PDB poses) and produce `data/processed/master.csv`.

### Phase 2: Baselines & Splits
*   [x] Generate and persist splits into `data/interim/splits.json` ensuring no scaffold-leakage.
*   [x] Implement RF + ECFP4 baseline and establish the non-deep learning benchmark score (RMSE & Pearson R).

### Phase 3: Transfer Learning Models
*   [x] Set up PyTorch Dataloaders for sequence modeling.
*   [x] Integrate ChemBERTa (HuggingFace) and write a regression head.
*   [ ] Integrate a 3D architecture using the provided PDB poses.
*   [x] Train models across all specified data budgets $\{50, 100, 250, 500, \text{full}\}$.

### Phase 4: Evaluation & Reporting
*   [x] Write standardized evaluation scripts to dump metrics into `results/metrics/`.
*   [x] Produce learning curves (Data Efficiency Ratio) and OOD scatter plots.
*   [ ] Draft the methods, results, and discussion for the manuscript in `private/manuscript/`.
