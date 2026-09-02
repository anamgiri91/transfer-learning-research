# Decision log

Every deviation from `plan.md` gets an entry. This is what separates a
pre-registered protocol from a post-hoc story: if the analysis changed, the
change is dated and justified here rather than absorbed silently.

| Date | Decision | Rationale | Affects |
|---|---|---|---|
| 2026-09-01 | Protocol drafted; arm list and analysis frozen | Pre-registration before any data is seen | `plan.md` |
| 2026-09-01 | Scaffold split is the primary endpoint; random is reference only | Random splits overstate generalisation for congeneric series | §3.3 |
| 2026-09-01 | Replicate spread > 1 log ⇒ discard, not average | "High-fidelity" is the title's claim, so it must be a gate | §3.2 |
| 2026-09-01 | N < 300 triggers a shift to baseline-focused reporting | Fixed in advance to prevent post-hoc rescue of deep arms | §3.2 |
| 2026-09-01 | **Target changed: EV-A71 3C → EV-A71/CVA16 2A protease.** Source changed ChEMBL → OpenBind Zenodo release | The available data is the OpenBind 2A structure–affinity release, not a 3C activity set. The protocol was written before the data was inspected | `plan.md` §2, §3.1 |
| 2026-09-01 | Affinities are **CVA16** 2A^pro, a surrogate differing from EV-A71 at 5 non-active-site residues | Documented by Lithgo et al. 2024; stated in the manuscript body, not the limitations | `paper/manuscript.md` §3.1 |
| 2026-09-01 | Modelling unit changed from crystal complex (649 rows) to **compound** (494) | The release is one row per complex; 150 rows are repeat structures of the same ligand, which would leak across folds | `scripts/prepare_openbind.py` |
| 2026-09-01 | `B1` uses sklearn `HistGradientBoostingRegressor`, not LightGBM | LightGBM cannot load in this environment (missing `libomp`). Same algorithm class; equal-budget rule preserved | `plan.md` §4 |
| 2026-09-01 | Scaffold-split group ordering randomised per seed (was deterministic smallest-first) | **Bug**: the deterministic order produced an identical split for all 10 seeds, giving the primary endpoint zero variance and making the paired tests meaningless | `src/evapro/data/splits.py` |
| 2026-09-01 | Held-out folds reject a group that would exceed quota | One large scaffold group could inflate the val fold to 2× target, making seeds non-comparable | `src/evapro/data/splits.py` |
| 2026-09-01 | Temporal split dropped for this dataset | The OpenBind release carries no publication/deposition year per compound | `plan.md` §3.3 |
| 2026-09-01 | Decontamination ablation (§7.1) **not performed** | The ChemBERTa PubChem-77M pretraining corpus was not retrieved. Transfer results are therefore an upper bound, stated as limitation 3 | `paper/manuscript.md` §7 |
| 2026-09-01 | Fidelity gate passed with nothing removed (max spread 0.49 log) | Reported as a passed audit rather than an active filter, to avoid implying it did work | `paper/manuscript.md` §3.2 |
| 2026-09-01 | Butina cluster split added and run (plan.md §3.3) | Guo et al. 2024 show scaffold splits still leak; measured here as 29% of scaffold-split test compounds having a training neighbour at Tanimoto >= 0.7 | `table0`, `table4` |
| 2026-09-01 | Cross-split comparison reports R² and skill-vs-B0, never raw RMSE | Stricter splits yield lower-variance test folds (SD 0.87 -> 0.66), so raw RMSE inverts the apparent difficulty ordering | `paper/manuscript.md` §5.5 |
| 2026-09-01 | Bootstrap CIs resample **seeds**, not test rows | The seed is the unit of replication (independent split + fit); resampling rows would understate split variance | `scripts/make_report.py` |
| 2026-09-01 | T2 interim result (rho = 0.680, 3 seeds) withdrawn; completed sweep gives 0.613 | The first three seeds included T2's best. Withheld from tables by `--require-seeds 10`; recorded in §5.6 as a near-miss | `paper/manuscript.md` §5.6 |
| 2026-09-01 | T2 run on the scaffold split only | ~160 s per run on CPU; the primary endpoint was prioritised over the reference splits | `paper/manuscript.md` §6 |
| 2026-09-01 | Contamination ablation (§7.1) **performed** as a PubChem-membership upper bound | The 77M ChemBERTa-2 corpus is not redistributable; PubChem is a strict superset, so absence proves non-membership. 53% of test compounds could have been seen | `table5`, `paper/manuscript.md` §6.1 |
| 2026-09-01 | Hyperparameter search with validation-fold selection added (`tune_arms.py`) | `run_arms.py` used fixed defaults and never read the val fold — the most serious threat to a negative transfer result | `paper/manuscript.md` §6.2 |
| 2026-09-01 | **§5.6 interpretation retracted.** The "calibration collapse" (R² −1.26 at n=50) was an artefact of a fixed 40-epoch schedule at a 10x-too-low learning rate | Tuned, the same arm reaches R² +0.30. Section retained with a retraction notice rather than silently deleted | `paper/manuscript.md` §5.6 |
| 2026-09-01 | Baselines confirmed NOT under-tuned: 32 trials move B1 by +0.004 RMSE, helping in 11/26 cells | Establishes that the comparison was fair to the baselines even though it was unfair to the fine-tune | `paper/manuscript.md` §6.2 |
| 2026-09-01 | Tuned T2 evaluated at n=50 (10 seeds) and n=347 (5 seeds, 4 trials) | Full 32-trial HPO for a 160 s/run arm was not affordable; the reduced budget is disclosed rather than presented as equal | `paper/manuscript.md` §7 |
| 2026-09-02 | Activity-cliff ablation (§7.2) **performed**; test folds stratified three ways (cliff / smooth / distant) rather than cliff / non-cliff | A two-way split pools compounds the model can interpolate with compounds whose neighbourhood it never saw. The median predictor alone spans RMSE 0.72–1.32 across the three strata, so the confound is real, not hypothetical | `table6`, `table8`, `paper/manuscript.md` §6.3 |
| 2026-09-02 | Cliff threshold reused from the split audit (ECFP4 Tanimoto ≥ 0.7, Δp > 1) rather than newly chosen | A fresh threshold here would be a free researcher degree of freedom on an ablation with a small subgroup; 0.7 was already fixed for Table 0. 0.6 and 0.8 reported as sensitivity | `src/evapro/evaluation/cliffs.py` |
| 2026-09-02 | Tanimoto ≥ 0.8 cliff stratum reported but not tested | Only 3 of 10 seeds clear the 5-compound floor; the paired test is withheld rather than run on 3 seeds | `paper/manuscript.md` §7, limitation 10 |
| 2026-09-02 | `run_arms.py --save-preds` re-runs completed cells and asserts stored metrics are unchanged (tol 1e-9) | The manuscript's determinism claim was verified once by hand. Making it an assertion on every prediction run turns it into a standing check; 160 committed cells passed | `scripts/run_arms.py`, `paper/provenance.md` |
| 2026-09-02 | §7 Limitations written; H3 and H4 recorded as **untested** rather than answered | No in-domain arm was run (H3 has no evidence either way), and decontamination was bounded rather than performed (H4 likewise). Leaving §7 empty let the abstract's scope caveats stand in for a limitations section | `paper/manuscript.md` §7 |
| 2026-09-02 | The claim count stated in §10 is itself verified against the checker's own total | It had already drifted (85 stated, 111 actual) — the exact failure mode §10 claims to have eliminated | `scripts/verify_manuscript.py` |
