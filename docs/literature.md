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

### 2.3 Sultan, Rausch-Dupont, Khan, Kalinina, Klakow, Volkamer (2025/2026) — [abstract]
*Transformers for molecular property prediction: domain adaptation efficiently
improves performance.* J. Cheminformatics (2026), DOI 10.1186/s13321-026-01252-z.
**Open preprint: arXiv:2503.03360** <https://arxiv.org/abs/2503.03360>

> **Correction, 2026-09-02.** This entry previously read "(2026) — [secondary]
> ... retrieval blocked by an authentication redirect", with no author list.
> The Springer version is indeed paywalled, but an **open arXiv preprint was
> available all along** and was simply not found. The label is corrected to
> [abstract] and the authors are named. The practical cost of the error: this
> is the paper motivating arm T4, and the old label barred it from carrying any
> number, so T4's design was argued without its most relevant figures.

Reported findings, from the abstract:
- Pretraining beyond roughly **400K–800K molecules does not improve
  performance** across seven datasets covering five ADME endpoints.
- **Domain adaptation** — multi-task regression of physicochemical properties
  on a small domain-specific set of **≤ 4K molecules** — significantly improves
  performance and generalisation (reported p < 0.001).
- Chemically and physically informed features consistently do better across
  model types, and **random forest remains a strong baseline**.

**Consequence for this work:** it puts a number on the corpus size at which
in-domain adaptation is reported to work (≤ 4K), which is the same order as our
own 2,743-compound in-domain corpus (§6.4). Note the difference in adaptation
signal: theirs is computed physicochemical properties, ours is measured
bioactivity across eight protease targets.

### 2.4 Deep learning for low-data drug discovery — [secondary]
*Deep learning for low-data drug discovery: Hurdles and opportunities.*
Current Opinion in Structural Biology (2024).
<https://www.sciencedirect.com/science/article/pii/S0959440X24000459>

Frames the central tension this study probes: deep models fit millions of
parameters while discovery projects are structurally low-data.

### 2.5 Altae-Tran, Ramsundar, Pappu, Pande (2017) — [abstract]
*Low Data Drug Discovery with One-Shot Learning.* ACS Cent. Sci. **3(4),
283–293**; DOI 10.1021/acscentsci.6b00367; arXiv:1611.03199.
Open full text: <https://ncbi.nlm.nih.gov/pmc/articles/PMC5408335>

Cited **only** for the existence of one-shot/few-shot approaches to this
regime. It does **not** report the ~50-molecule crossover of §2.7, and an
earlier draft of the manuscript cited it alongside that claim in error.

### 2.6 Schimunek, Luukkonen, Klambauer (2025) — [full text via PMC]
*MHNfs: Prompting In-Context Bioactivity Predictions for Low-Data Drug
Discovery.* J. Chem. Inf. Model., April 2025.
DOI 10.1021/acs.jcim.4c02373 ·
open: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12076497/>

Few-shot bioactivity prediction, developed and evaluated on the FS-Mol
benchmark (4,938 training tasks, 157 test tasks from ChEMBL27). MHNfs
outperforms other few-shot models on the FS-Mol test set.

> **Correction, 2026-09-02.** This entry previously attributed the
> ~50-molecule crossover to this paper. It does not originate here: the
> Discussion **quotes it from Snyder et al. 2024** (§2.7) — "from 50 measures
> molecules upward, classic machine learning methods start outperforming
> few-shot learning methods". Cite §2.7 for the crossover, not this entry.

### 2.7 Snyder et al. (2024) — [abstract] — **primary source for the ~50-molecule crossover**
*The Goldilocks paradigm: comparing classical machine learning, large language
models, and few-shot learning for drug discovery applications.*
Communications Chemistry 7 (2024).
<https://www.nature.com/articles/s42004-024-01220-4>

Compares classical ML, LLMs and few-shot learning across a range of dataset
sizes and diversities, identifying an optimal regime for each model type. This
is the origin of the statement quoted by Schimunek et al. (§2.6) that **from
about 50 measured molecules upward, classical ML starts to outperform few-shot
methods** — the crossover our learning curves are built to straddle, starting
at exactly n = 50. Located 2026-09-02 while auditing the review; the manuscript
previously cited the figure to §2.5 and §2.6, neither of which is its source.

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

### 3.2 van Tilborg, Alenicheva, Grisoni (2022) — [abstract] — **the activity-cliff benchmark**
*Exposing the Limitations of Molecular Machine Learning with Activity Cliffs.*
J. Chem. Inf. Model. **62(23), 5938–5951**. DOI 10.1021/acs.jcim.2c01073 ·
open preprint: <https://chemrxiv.org/engage/chemrxiv/article-details/630cc44058843b8403a19810>

The standard ML benchmark for activity cliffs (MoleculeACE): pairs of
structurally similar molecules with large potency differences, and how badly
models handle them. Reports that descriptor- and fingerprint-based models are
frequently **better** than deep models on cliff compounds.

**Two published corrections** exist (<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10091401/>,
<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11683855/>), one for a software
bug that mislabelled activity-cliff pairs in the train/test split. The
conclusions survived retraining. Worth citing for its own sake: it is the same
class of split-construction bug this project's decision log records catching in
`splits.py`.

**Consequence for this work:** added 2026-09-02, after §6.3 had already been
written and committed. §6.3 defined cliffs from scratch, on similarity rather
than fragmentation, and argued the definition without citing the convention it
departs from. Our finding — that the *pretrained* arms are no worse on cliffs
and much worse on distant compounds — sits against this paper's reported
fingerprint advantage on cliffs, and should be presented as such.

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
