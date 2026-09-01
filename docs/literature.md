# Annotated literature review

**Reading-depth is labelled on every entry, and the label is honest.**

- **[full text]** — the complete paper was retrieved and read.
- **[abstract]** — only the abstract / landing page was retrieved.
- **[secondary]** — known only through a search-engine summary or another
  paper's description. **Not citable for a specific number** in the manuscript.

Nothing in this file is cited in `paper/manuscript.md` at a precision the
reading depth does not support. Where a claim needs a number, it comes from an
entry marked [full text], or from our own results files.

---

## 1. The dataset and its target

### 1.1 Lithgo, Tomlinson, Fairhead, Winokan, Thompson, Wild, Aschenbrenner, Balcomb, Marples, Chandran, Golding, Koekemoer, Williams, Wang, Ni, MacLean, Giroud, Godoy, Xavier, Walsh, Fearon, von Delft (2024) — [full text, read twice]
*Crystallographic Fragment Screen of Coxsackievirus A16 2A Protease identifies
new opportunities for the development of broad-spectrum anti-enterovirals.*
bioRxiv, posted 29 April 2024. CC-BY 4.0.
<https://doi.org/10.1101/2024.04.29.591684> ·
PDF: <https://www.biorxiv.org/content/10.1101/2024.04.29.591684.full.pdf> ·
PubMed: <https://pubmed.ncbi.nlm.nih.gov/38746446/>

*First pass — provenance.* Crystallographic fragment screen against
enteroviral 2A protease (2A^pro), a chymotrypsin-like cysteine protease that
self-cleaves from the viral polyprotein and is required for capsid folding and
virion maturation. Campaign identified **75 fragments** binding the protease,
including **38 unique compounds binding within the active site**. Target
selected in coordination with the **ASAP Discovery Consortium**
(<https://asapdiscovery.org>).

*Second pass — methodology, and the fact that governs our title.*
> "The 2A^pro sequence of EV-A71 and CVA16 **differ by 5 amino acids, none of
> which are near or predicted to affect** [the active site]."

All crystallography and affinity work is therefore performed on **CVA16 2A^pro
as a surrogate for EV-A71 2A^pro**. Method details: apo structure solved in C2
at **1.6 Å (PDB 8POA)**, two monomers per asymmetric unit, each with a
structural zinc-finger Zn²⁺. Fragments from the XChem libraries — DSiP, DSiP
EUbOpen Expansion, FragLites, PepLites, MiniFrags, York3D, SpotXplorer3 —
dispensed acoustically (Echo) at 25% v/v, 3 h soak at 20 °C. Processing:
DIMPLE → GRADE restraints → **PanDDA2** hit identification → REFMAC/COOT
refinement via XChemExplorer. Beamlines I04-1 and I03, Diamond Light Source.

**Consequence for this work:** the case study is titled EV-A71, but the
measured protein is CVA16. This is stated in the manuscript's Data section
rather than buried, and is a limitation, not a footnote.

### 1.2 OpenBind Consortium (2026) — [full text of record, read twice]
*OpenBind Structure–Affinity Data Release: Enterovirus A71 (EV-A71) /
Coxsackievirus A16 (CVA16) 2A protease.* Zenodo, released 5 May 2026.
Licence **CC0 1.0** (public domain).
<https://doi.org/10.5281/zenodo.20026661> ·
Record: <https://zenodo.org/records/20026661> ·
Project: <https://openbind.uk/news/blog-openbinds-first-release-a-structure-affinity-dataset-for-structure-based-ai/> ·
Benchmark code: <https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark>

*First pass — contents.* **925 crystallographic binding events** from **699
compounds**; **K_D for 601 compounds**, measured on the **Creoptix
WAVEsystem** (grating-coupled interferometry). Single archive
`OpenBind_EV-A71_2A.zip` (98.4 MB) — the file present in `data/raw/`.

*Second pass — what the release benchmarks.* Reference benchmarks are
**structure-based**: docking (AutoDock Vina), ML docking (GNINA, DiffDock),
cofolding (AlphaFold3, Boltz, OpenFold3), and affinity prediction. Reported:
redocking succeeds up to 85% (GNINA) but **cross-docking into apo structures
falls below 5%**, attributed to binding-site loop conformational change;
fine-tuning OpenFold3-p2 on EV-A71 2A data raised success 36% → 76%.

**Consequence for this work:** the official benchmark is a *structure-based*
one. This study asks a different and complementary question — whether
**ligand-only** transfer learning helps at this data scale — so it does not
duplicate the consortium's numbers and is not directly comparable to them.

---

## 2. Does molecular pretraining actually help?

### 2.1 Chithrananda, Grand, Ramsundar (2020) — [abstract]
*ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property
Prediction.* arXiv:2010.09885. <https://arxiv.org/abs/2010.09885>

RoBERTa-style masked-language-model pretraining on SMILES from PubChem;
released a curated **77M-SMILES** pretraining corpus. Reports competitive —
not dominant — MoleculeNet performance, and that performance scales with
pretraining set size.

### 2.2 Ahmad, Simon, Chithrananda, Grand, Ramsundar (2022) — [abstract]
*ChemBERTa-2: Towards Chemical Foundation Models.* arXiv:2209.01712.
<https://arxiv.org/abs/2209.01712>

**This is the correct citation for the encoder used in this study**
(`DeepChem/ChemBERTa-77M-MTR`, <https://huggingface.co/DeepChem/ChemBERTa-77M-MTR>):
the `-MTR` variant is ChemBERTa-**2**'s multi-task-regression pretraining
objective, not the original ChemBERTa MLM model. The paper compares multi-task
against self-supervised pretraining while varying hyperparameters and
pretraining-set size, on up to 77M PubChem compounds. *(We did not retrieve the
full text; the MTR-vs-MLM comparison is therefore not quoted numerically.)*

### 2.3 Domain adaptation for molecular transformers (2026) — [secondary]
*Transformers for molecular property prediction: domain adaptation efficiently
improves performance.* J. Cheminformatics, DOI 10.1186/s13321-026-01252-z.
<https://doi.org/10.1186/s13321-026-01252-z>

Reported finding: transformers beat Morgan-fingerprint baselines on most
datasets, but the margin comes mainly from **domain-adapted** pretraining
rather than from scale alone; a random forest on RDKit descriptors remains a
strong baseline. **Retrieval blocked by an authentication redirect**, so this
is cited only as motivation for arm T4 (in-domain transfer) and never for a
specific number.

### 2.4 Deep learning for low-data drug discovery — [secondary]
*Deep learning for low-data drug discovery: Hurdles and opportunities.*
Current Opinion in Structural Biology (2024).
<https://www.sciencedirect.com/science/article/pii/S0959440X24000459>

Frames the central tension this study probes: deep models fit millions of
parameters while discovery projects are structurally low-data.

### 2.5 Altae-Tran, Ramsundar, Pappu, Pande (2017) — [secondary]
*Low Data Drug Discovery with One-Shot Learning.* ACS Cent. Sci.;
arXiv:1611.03199. <https://arxiv.org/pdf/1611.03199>

### 2.6 Schimunek et al. (2025) — [secondary]
*MHNfs: Prompting In-Context Bioactivity Predictions for Low-Data Drug
Discovery.* J. Chem. Inf. Model.
<https://pubs.acs.org/doi/10.1021/acs.jcim.4c02373> ·
<https://pmc.ncbi.nlm.nih.gov/articles/PMC12076497/>

Relevant reported observation: **from roughly 50 measured molecules upward,
classical ML begins to outperform few-shot methods** — a crossover directly
comparable to the learning curves measured here.

---

## 3. Evaluation and splitting

### 3.1 Guo, Hernandez-Hernandez, Ballester (2024) — [full text of abstract/landing, read twice]
*Scaffold Splits Overestimate Virtual Screening Performance.* arXiv:2406.00873.
<https://arxiv.org/abs/2406.00873> · PDF: <https://arxiv.org/pdf/2406.00873>
Also in ICANN 2024 proceedings:
<https://link.springer.com/chapter/10.1007/978-3-031-72359-9_5>

*First pass — the claim.* Scaffold splits are widely treated as the realistic
evaluation, but **molecules with different Bemis–Murcko scaffolds are often
still similar**, so scaffold splits leave unrealistically high train/test
similarity and inflate estimated performance.

*Second pass — evidence and ranking.* **2,100 models** across **60 NCI-60
datasets** (~30k–50k molecules each). Difficulty ordering, easiest to hardest:
**random < scaffold < Butina < UMAP**. Recommends UMAP-based cluster splits;
warns that model selection on scaffold splits can produce suboptimal
prospective choices.

**Consequence for this work:** our protocol's use of scaffold split as the
*primary* endpoint is defensible but not conservative. This paper is the reason
Butina clustering is reported as the stricter check, and the reason we do not
present scaffold-split numbers as prospective estimates.

---

## 4. What this review changes in the protocol

1. **Title vs. measured protein.** §1.1 forces an explicit statement that
   affinities are CVA16 2A^pro (5 substitutions from EV-A71, none active-site).
2. **Correct model citation.** §2.2 — the encoder is ChemBERTa-2 (MTR), and
   citing Chithrananda 2020 for it would be wrong.
3. **Splitting.** §3.1 demotes scaffold split from "the honest endpoint" to
   "the optimistic-but-standard endpoint"; Butina is the stricter comparison.
4. **Expected crossover.** §2.6 predicts classical ML overtaking low-data
   methods above ~50 compounds — our learning curves start at exactly n=50.
5. **Non-duplication.** §1.2 confirms the official OpenBind benchmark is
   structure-based, so a ligand-only transfer study is complementary.
