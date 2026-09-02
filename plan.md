# Research Protocol

**Benchmarking Transfer Learning Efficacy on High-Fidelity Viral Protease
Datasets: A Case Study on EV-A71**

Status: **amended; execution complete for the arms that were run.**
Written 2026-09-01; see Amendment 1 and the checklist in §10.


> ## ⚠ Amendment 1 — 2026-09-01, after data inspection
>
> **This protocol was written before the dataset was inspected, and several of
> its assumptions turned out to be wrong.** It is preserved as written below;
> corrections are recorded in [`docs/decision-log.md`](docs/decision-log.md)
> rather than edited into the text, so the pre-registration stays honest.
>
> The material divergences:
>
> | Protocol says | Reality |
> |---|---|
> | Primary target EV-A71 **3C** protease | Data available is EV-A71/CVA16 **2A** protease |
> | Source: ChEMBL / PubChem / BindingDB activity records | Source: OpenBind Zenodo structure–affinity release (CC0) |
> | Label: pActivity from IC50/Ki/Kd/EC50 | Label: pK_D from a single Creoptix WAVEsystem assay |
> | Protein assayed is EV-A71 | Protein assayed is **CVA16** 2A^pro (5-residue surrogate) |
> | Temporal split available | No per-compound year in the release — split dropped |
> | Expected N a few hundred–2k for 3C | N = 494 compounds, 272 scaffolds (passes the N ≥ 300 rule) |
>
> Sections §1 (question), §4 (arms), §6 (statistics) and §8 (threats) survive
> the change and still govern the study. §2, §3 and §7.1 are superseded as
> tabulated above; §7.1 (decontamination) is **bounded but not performed** —
> the corpus is not redistributable, so only a PubChem-membership upper bound
> was obtainable. See `paper/manuscript.md` §6.1 and §7, limitation 3.


This document is pre-registration-shaped on purpose: the analysis plan below is
fixed *before* results exist, and any deviation gets an entry in
`docs/decision-log.md`.

---

## 1. Question

Molecular transfer learning is usually justified on large, noisy benchmarks
(MoleculeNet, ChEMBL bulk). Viral protease inhibitor datasets are the opposite
regime: **hundreds to low thousands of measurements, but curated to a tight
assay standard.** It is not obvious that pretraining helps here — the classic
result is that count-based fingerprints plus gradient boosting are extremely
hard to beat below ~10^3 examples.

> **RQ.** On small, high-fidelity EV-A71 protease datasets, does transfer
> learning improve predictive performance and data efficiency over a
> from-scratch baseline of matched capacity — and does the improvement survive
> scaffold-based splitting and pretraining-corpus decontamination?

### Hypotheses

| | Statement | Decided by |
|---|---|---|
| **H1** | Transfer arms beat the best from-scratch baseline on scaffold-split RMSE | Paired Wilcoxon across seeds, Holm-corrected |
| **H2** | The advantage *grows* as training-set size shrinks (data efficiency) | Interaction term in the learning-curve model; DER (§6) |
| **H3** | In-domain transfer (related picornaviral/3CL proteases) beats generic self-supervised pretraining | Direct arm contrast T4 vs T1 |
| **H4** | Part of any apparent gain is leakage — test scaffolds present in the pretraining corpus | Decontamination ablation (§7.1) |

H4 is a hypothesis we expect to *partially confirm*. It is stated up front so a
negative headline result is still a publishable one.

---

## 2. Targets and scope

- **Primary:** EV-A71 **3C protease** (3C^pro) — chymotrypsin-like cysteine
  protease, the better-populated target.
- **Secondary:** EV-A71 **2A protease** (2A^pro) — smaller dataset; used to test
  whether conclusions hold at even lower N.
- Small-molecule inhibitors only. Peptidomimetics are retained but flagged, and
  a covalent/non-covalent indicator is carried through as a covariate.

Out of scope: structure-based / docking arms, generative design, broad-spectrum
claims beyond EV-A71.

---

## 3. Data

### 3.1 Sources
ChEMBL (3C and 2A activities), PubChem BioAssay, BindingDB, plus hand-curated
literature values. Related-protease transfer corpus: HRV 3C, CVB3 3C,
SARS-CoV-2 3CL^pro. Full table with licences in `data/raw/README.md`.

### 3.2 "High-fidelity" is a hard filter, not an adjective
The title claims fidelity, so curation is a gate, enforced in
`src/evapro/data/curate.py`:

1. Activity in {IC50, Ki, Kd, EC50}, `standard_relation == '='`.
   Censored values (`>`, `<`) are split into a separate file and used **only**
   in a sensitivity analysis — never silently dropped, never silently included.
2. Units → nM; `pActivity = 9 − log10(nM)`. Values outside [3, 11] rejected.
3. Single-protein assay; ChEMBL confidence ≥ 8.
4. RDKit-parseable, desalted, neutralised, no mixtures, MW ∈ [150, 900].
5. **Replicate agreement:** duplicates on (InChIKey, target, activity type)
   collapse to the median and are **discarded if spread > 1 log unit.**
6. Enzymatic and cell-based readouts are kept in separate columns, never pooled
   into one label.

Expected yield is a few hundred to ~2k compounds for 3C. **If the curated 3C set
falls below N = 300, the protocol changes**: the deep arms are reported as a
negative result and the paper's centre of gravity moves to the fingerprint
baselines. That decision rule is fixed now to avoid post-hoc rescue.

### 3.3 Splits
Splits are materialised as JSON (InChIKey → fold) in
`data/processed/splits/` and read by every arm, so no experiment can reshuffle
its own test set.

| Split | Role |
|---|---|
| **Bemis–Murcko scaffold** | **Primary endpoint** |
| Butina cluster (Tanimoto 0.6 on ECFP4) | Stricter generalisation check |
| Temporal (by first publication year) | Deployment realism |
| Random | Optimism reference only — never the headline |

Test fraction 0.2, validation 0.1 carved from train. 10 seeds throughout.
At the smallest training sizes the test set is held **fixed** across sizes so
learning curves are comparable.

---

## 4. Arms

Every arm is a (representation, pretraining source, adaptation) triple.
Capacity is matched to the baselines where architecture permits; where it
cannot be, parameter counts are reported alongside.

**From scratch**
- `B0` median predictor (sanity floor)
- `B1` ECFP4 count fingerprint + LightGBM ← *the arm to beat*
- `B2` RDKit descriptors + Random Forest
- `B3` D-MPNN (Chemprop-style) trained from scratch

**Transfer**
- `T1` Chemical-language MLM (ChemBERTa-class), full fine-tune
- `T2` Same encoder, **frozen**, linear probe
- `T3` Alternative pretrained encoder (MolFormer / Uni-Mol class), fine-tune
- `T4` **In-domain**: multitask pretrain on related 3C/3CL proteases → fine-tune
- `T5` **Chained**: self-supervised → related-protease multitask → EV-A71
- `T6` Ligand encoder + ESM-2 target embedding, joint 2A/3C model

`T2` matters more than it looks: if a frozen encoder plus a linear head matches
full fine-tuning, the story is about representation quality, not adaptation.

---

## 5. Training

Hyperparameters tuned on the validation fold **within each split and each
training-set size**, by fixed-budget random search (32 trials), identical budget
for every arm — baselines included. Under-tuning the baseline is the most
common way this literature manufactures wins; equal budget is the defence.
Early stopping on validation RMSE, patience 20. Seeds 0–9.

---

## 6. Endpoints and statistics

**Primary:** scaffold-split test **RMSE** on pActivity.
**Secondary:** Spearman ρ, MAE, R², precision@10% (enrichment view).
All reported as median over 10 seeds with bootstrap 95% CI (10k resamples).

**Learning curves** at N ∈ {50, 100, 250, 500, full}, fixed test set.

**Data-Efficiency Ratio** — the paper's headline quantity:

> DER = N_baseline / N_transfer, where each is the training-set size at which
> that arm first reaches a fixed target RMSE.

Obtained by interpolating each arm's learning curve; CI by bootstrapping over
seeds. DER > 1 means transfer bought data. Reporting a ratio rather than a
single-point delta is what makes H2 falsifiable.

**Testing.** Paired Wilcoxon signed-rank across seeds for each transfer arm vs
`B1`, Holm correction over the arm family. Effect sizes always reported;
significance never reported alone. Given 10 seeds the study is powered only for
fairly large effects — an underpowered null is reported as inconclusive, not as
"no difference".

---

## 7. Ablations

### 7.1 Decontamination (the load-bearing one)
Measure overlap between the pretraining corpus and each test set at three
levels: exact InChIKey, Bemis–Murcko scaffold, and ECFP4 Tanimoto ≥ 0.7.
Then **re-pretrain with the overlap removed** and re-run the affected arms.
The gap between contaminated and decontaminated performance is a reportable
result, not a footnote. This is the ablation most likely to change the paper's
conclusion.

### 7.2 Others
- Adaptation strategy: full FT vs linear probe vs LoRA vs layer-wise unfreezing
- Pretraining corpus size (10%, 50%, 100%)
- Fidelity ablation: relax the 1-log replicate filter and re-run — does clean
  data matter more than pretraining?
- Activity-cliff subset: performance restricted to matched molecular pairs with
  >1 log activity difference
- 2A vs 3C: do conclusions transfer across targets?

---

## 8. Threats to validity

| Threat | Mitigation |
|---|---|
| Pretraining contamination | §7.1, stated as its own result |
| Scaffold split still leaks via analog series | Butina cluster split as stricter check |
| Baseline under-tuned | Identical HPO budget; `B1` tuned first |
| Small test set → wide CIs | Bootstrap CIs everywhere; no single-seed claims |
| Assay heterogeneity across sources | Enzymatic vs cell-based never pooled; source as covariate |
| Multiple comparisons across many arms | Holm correction; arm list frozen here |
| Researcher degrees of freedom | This document + `docs/decision-log.md` |

---

## 9. Deliverables

1. Curated, versioned EV-A71 3C/2A benchmark with split files and manifests —
   arguably the most durable output.
2. Reproducible benchmark harness (`make bench`).
3. Learning curves + DER table across arms and splits.
4. Decontamination analysis.
5. Manuscript.

## 10. Execution checklist

Updated 2026-09-02. `[x]` done, `[~]` partial, `[ ]` not done. Partial and
undone items are each accounted for in `paper/manuscript.md` §7 — an unchecked
box here must correspond to a stated limitation there, or one of the two
documents is lying.

- [x] Fetch and curate sources — via `scripts/prepare_openbind.py`, not
      `make data`; the source changed (Amendment 1)
- [x] Report curated N per target; **apply the §3.2 N < 300 decision rule** —
      N = 494, so the rule did not fire and the deep arms stayed in scope
- [x] Build and verify splits → `make splits` (leakage tests pass; `tests/test_splits.py`)
- [~] Baselines `B0`–`B3` across seeds and sizes — `B0`–`B2` complete over
      10 seeds × 4 sizes × 3 splits; **`B3` (D-MPNN) not run** (§7.6)
- [~] Transfer arms `T1`–`T6` — `T1`, `T2`, `T4`, `T5` complete on the scaffold
      split (plus the `T0r` untrained-encoder control added in §6.4);
      **`T3` and `T6` not run**. **H3 is now tested** (§6.4): supported for the
      chained arm, inconclusive in its pre-registered `T4`-vs-`T1` form
- [~] Contamination measurement, then decontaminated re-run — measurement done
      as a PubChem-membership upper bound (53%); **the decontaminated re-run is
      impossible**, the corpus is not redistributed, so **H4 is untested** (§7.3)
- [~] Ablations §7.2 — activity cliffs done (§6.3); adaptation strategy partial
      (full FT vs linear probe only, no LoRA / layer-wise); fidelity ablation
      **vacuous**, the gate removed nothing; corpus size and 2A-vs-3C
      **impossible here** (§7.9)
- [x] Figures, tables, manuscript — draft complete including §7; tables are
      generated from source and every prose number is machine-checked
      (`make verify`; the claim count lives in one place, manuscript §10)

Three hypotheses were decided and one was not. **H1** (transfer beats the best
from-scratch baseline) and **H2** (the advantage grows as data shrinks) are
answered negatively on every arm run. **H3** (in-domain beats generic
pretraining) is answered in §6.4 — supported for the chained arm `T5`,
inconclusive in the exact `T4`-vs-`T1` form this document specifies. **H4**
still has no evidence for the ChemBERTa arms, whose corpus is not distributed;
it is answerable for the in-domain arms, whose corpus we built, and that
ablation is in progress.
