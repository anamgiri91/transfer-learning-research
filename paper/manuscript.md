# Benchmarking Transfer Learning Efficacy on a High-Fidelity Viral Protease Dataset: A Case Study on EV-A71 / CVA16 2A Protease

**Status: draft. Results sections are populated only from files listed in
[`provenance.md`](provenance.md).** Every number below carries an inline
pointer of the form `[→ file]` naming the artefact it was read from and the
script that produced it.

---

## Abstract

Self-supervised pretraining is the default opening move in molecular property
prediction, but its evidence base is large, noisy benchmarks. We test whether
it helps in the opposite regime — few hundred compounds, one target, one assay
— using the OpenBind EV-A71 / CVA16 2A protease structure–affinity release
(494 curated compounds, 272 scaffolds, maximum replicate spread 0.49 log
units). Five arms (median predictor; ECFP4 + gradient boosting; RDKit
descriptors + random forest; ChemBERTa-2 frozen linear probe; ChemBERTa-2 full
fine-tune) are compared on identical materialised splits across 10 seeds and
four training-set sizes, with 640 evaluated runs.

**On this dataset, pretraining did not help.** The frozen probe is
significantly worse than the ECFP4 baseline (paired Wilcoxon, Holm-corrected
p = 0.012); the fine-tune is also worse but does not reach significance
(p = 0.074, inconclusive at 10 seeds). Neither reaches the baseline's full-data
RMSE at any training size, so both have a data-efficiency ratio of zero, and
the deficit does not shrink in the low-data regime where transfer is supposed
to pay. The two baselines are statistically indistinguishable from each other
(p = 0.56); the nominal best is a random forest on RDKit descriptors, but we do
not claim it beats gradient-boosted fingerprints.

Three secondary observations bear on how such benchmarks should be run. First,
the transfer arm degrades more than the baselines under a stricter split: from
scaffold to Butina clustering, the frozen probe retains 13% of its R² against
the baselines' 40–50%. Second, scaffold splitting was not harder than random
splitting here — it shares zero Bemis–Murcko scaffolds with training, yet 29%
of its test compounds still have a training neighbour at Tanimoto ≥ 0.7. Third,
raw RMSE is not comparable across splitting strategies, because stricter splits
yield lower-variance test folds; a variance-normalised measure is required, and
its absence inverts the apparent difficulty ordering.

Three ablations qualify this. Giving every arm an explicit search budget with
validation-fold model selection leaves the baselines essentially unchanged
(32 trials move the fingerprint baseline by a median of +0.004 RMSE) but
transforms the fine-tune at n = 50, from RMSE 1.23 / R² −1.26 to 0.74 / +0.30 —
so a benchmark that fixes hyperparameters across arms will understate transfer,
and one earlier claim of ours built on that configuration is withdrawn. Even
tuned, the fine-tune is not shown to beat the baselines (n = 50: better in 3 of
10 seeds, p = 0.16; full data: 2 of 5, p = 0.63) — indistinguishable rather than
worse, at ~80× the compute per fit. Separately, up to 53% of test compounds are present in
PubChem, an upper bound on pretraining overlap; this cannot explain transfer
losing, but it caps how much any transfer advantage here should be believed.
Third, stratifying the test fold by its relationship to training localises the
deficit: both transfer arms are significantly worse than the baseline only on
compounds with **no near training neighbour** (p = 0.018 and 0.019), and
neither is behind on activity cliffs — on cliffs both are nominally ahead. The
weakness is extrapolation to novel chemistry, not local label roughness, which
is the same conclusion the scaffold-to-Butina comparison reaches by an
independent route.

A fourth ablation locates the failure. Against an **untrained encoder of the
same architecture**, the pretrained one is statistically indistinguishable
(5 of 10 seeds, p = 0.49) — generic pretraining buys essentially nothing here.
Multitask pretraining on a 2,743-compound corpus of related 3C/3C-like
proteases does clear that control (7–8 of 10 seeds, p = 0.049 and 0.010) and
beats the generic probe in 9 of 10 seeds (p = 0.006), supporting **H3** for the
chained arm while the pre-registered T4-vs-T1 form stays inconclusive. In-domain
pretraining is doing real work; it still does not overtake count fingerprints.

These are single-target, single-assay results with one pretrained encoder. They
constrain claims about *this* regime; they are not a general verdict on
molecular pretraining.

Affinities are measured on CVA16 2A protease as a five-residue surrogate for
EV-A71. The pretraining-corpus decontamination ablation was not performed, so
the reported transfer performance is an upper bound — which strengthens rather
than weakens the negative result.

---

## 1. Introduction

Self-supervised pretraining is now the default opening move in molecular
property prediction, but the evidence for it is drawn largely from benchmarks
that are big and noisy. Real target-based discovery projects are the opposite:
a few hundred compounds, measured carefully, against one protein. Whether
pretraining pays off in *that* regime is a separate empirical question, and the
honest prior is not encouraging — count-based fingerprints with gradient
boosting remain hard to beat below roughly a thousand training examples, and
recent work finds transformer gains come mainly from domain-adapted pretraining
rather than scale alone (§2).

The OpenBind EV-A71 / CVA16 2A protease release makes the question answerable
on a clean example. It pairs a crystallographic fragment-screening campaign with
K_D measurements from a single biophysical assay, so the label noise that
usually confounds "does pretraining help?" is largely absent.

**Contributions.**

1. A compound-level, leakage-audited benchmark derived from the OpenBind 2A
   protease release, with materialised split files (§3).
2. A like-for-like comparison of from-scratch baselines against pretrained
   chemical-language-model transfer, over learning curves from 50 compounds to
   the full training fold (§4–5).
3. A quantified statement of what random splits buy you in apparent performance
   on this dataset, measured rather than asserted (Table 0, §5.1).
4. A negative-result-tolerant analysis: effect sizes are reported beside every
   p-value, and underpowered comparisons are labelled inconclusive rather than
   null (§4.4).

## 2. Related work

Full annotated review with reading-depth labels: [`../docs/literature.md`](../docs/literature.md).

**Pretraining for molecules.** ChemBERTa established RoBERTa-style masked
language modelling over SMILES at scale, releasing a 77M-compound PubChem
corpus and reporting competitive — not dominant — MoleculeNet results
([Chithrananda et al. 2020](https://arxiv.org/abs/2010.09885)). ChemBERTa-2
extended this to multi-task regression (MTR) pretraining
([Ahmad et al. 2022](https://arxiv.org/abs/2209.01712)). **We take the encoder
used in this study, `DeepChem/ChemBERTa-77M-MTR`, to be the ChemBERTa-2 MTR
model** and cite it accordingly: MTR (multi-task regression) is ChemBERTa-2's
objective, not the original ChemBERTa's masked language modelling, and 77M is
the corpus size both papers use. We flag that this is an inference from the
checkpoint's name and training objective — its HuggingFace model card is empty
— rather than a claim documented at source.

[Sultan et al. (2025)](https://arxiv.org/abs/2503.03360) report that
transformer advantages over fingerprint baselines depend more on
domain-adapted pretraining than on corpus scale: pretraining beyond roughly
**400K–800K molecules** stops helping, while domain adaptation on a
**≤ 4K-molecule** set significantly improves performance, and a random forest
remains a strong baseline. That corpus figure is the same order as the
in-domain corpus assembled in §6.4, which is why the arm is worth running.

**The low-data regime.** The tension this study probes — deep models fitting
millions of parameters against discovery projects that are structurally
low-data — is reviewed by
[van Tilborg et al. (2024)](https://doi.org/10.1016/j.sbi.2024.102818).
Few-shot and one-shot approaches were developed precisely for this setting
([Altae-Tran et al. 2017](https://doi.org/10.1021/acscentsci.6b00367);
[Schimunek et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12076497/)).
The observation that matters for our design is
[Snyder et al. (2024)](https://www.nature.com/articles/s42004-024-01220-4)'s:
**classical ML starts to outperform few-shot methods from roughly 50 measured
molecules upward** — a crossover our learning curves are designed to straddle,
starting at exactly n = 50. (We previously credited this figure to the two
few-shot papers above; it is Snyder et al.'s, quoted in Schimunek et al.'s
discussion, and the mis-attribution is recorded in
[`../docs/literature.md`](../docs/literature.md) §2.6.)

**Evaluation and splitting.** Scaffold splitting is the field's standard
concession to realism, but [Guo, Hernandez-Hernandez and Ballester
(2024)](https://arxiv.org/abs/2406.00873) show across 2,100 models on 60 NCI-60
datasets that molecules with distinct Bemis–Murcko scaffolds are frequently
still similar, so scaffold splits *also* overestimate performance; their
difficulty ordering is random < scaffold < Butina < UMAP. We therefore treat
our scaffold-split results as the standard-but-optimistic endpoint, not as a
prospective estimate, and we measure the random-split optimism directly
(Table 0).

**The target.** The 2A protease is a chymotrypsin-like cysteine protease that
self-cleaves from the enteroviral polyprotein and is required for capsid
folding and virion maturation; inhibiting it derails capsid assembly
([Lithgo et al. 2024](https://doi.org/10.1101/2024.04.29.591684)).

## 3. Data

### 3.1 Source and a necessary caveat about the title

Data come from the OpenBind Consortium structure–affinity release for EV-A71 /
CVA16 2A protease ([Zenodo, CC0](https://doi.org/10.5281/zenodo.20026661)):
925 crystallographic binding events from 699 compounds, with K_D for 601
compounds measured on a Creoptix WAVEsystem. The underlying fragment campaign
is described by [Lithgo et al. (2024)](https://doi.org/10.1101/2024.04.29.591684).

**The measured protein is Coxsackievirus A16 2A protease, not Enterovirus A71
2A protease.** CVA16 2A^pro is used as an experimental surrogate. Lithgo et al.
(2024) state that the two sequences differ at five amino acids, none of which
lie near or are predicted to affect the active site [→ Lithgo et al. 2024,
Results: "2A^pro Construct"]. Because this single sentence conditions the
entire study, we do not take it on citation — we re-derive it from primary
sequence data [→ `scripts/verify_surrogate.py` →
`results/tables/table9_surrogate_divergence.csv`].

Comparing the UniProt-annotated Protease 2A chains (150 residues) of EV-A71
BrCr/1970 (Q66478) against the two reviewed CVA16 reference strains:

<!-- TABLE:surrogate START -->
| Reference | Surrogate | differences | at catalytic site | nearest catalytic | at Zn site |
|---|---|---|---|---|---|
| EV-A71 BrCr/1970 | CVA16 G-10 | 7 | 0 | 5 residues | 0 |
| EV-A71 BrCr/1970 | CVA16 Tainan/5079/98 | 8 | 0 | 5 residues | 0 |
<!-- TABLE:surrogate END -->

Two things follow, and they point in opposite directions.

**The count of five does not reproduce, and we do not rely on it.** Against
these reference strains the divergence is **7** (G-10: F15Y, T26N, N57D, N66S,
V82I, E102V, S129N) or **8** (Tainan: F15Y, T26N, N66S, R68K, S77T, V82I,
E102V, S129N). This is not a contradiction of Lithgo et al. — enterovirus
sequences are strain-dependent and they compared the specific constructs they
crystallised — but it does mean "five residues" is a property of one strain
pair, not of EV-A71 versus CVA16 in general. We therefore describe the
surrogate gap qualitatively rather than quoting a number.

**The claim that actually matters is independently confirmed.** The catalytic
triad — His21, Asp39, Cys110 in 2A numbering — is **identical across all three
sequences**, and no differing position is a catalytic residue in either
surrogate; the nearest substitution to the triad is 5 residues away in
sequence. The four structural Zn²⁺ ligands (56, 58, 116, 118) are also
conserved. One caveat we did not find stated anywhere: in the G-10 strain the
substitution **N57D sits directly between two of those zinc ligands**. That is
a structural site rather than the catalytic one, and sequence adjacency is not
structural proximity, but "none near the active site" is a stronger claim than
"none catalytic", and only the latter is what we have verified.

The consequence for reading this paper is unchanged: every "EV-A71" result
here — including in the title — is strictly a CVA16 result extrapolated across
a handful of non-catalytic substitutions. We state it here rather than in a
limitations paragraph because it conditions the entire study.

### 3.2 Curation to compound level

The released table is one row per **crystal complex**, not per compound. Naive
modelling on complexes would place the same ligand in both training and test
folds. Curation [→ `scripts/prepare_openbind.py` → `data/processed/eva71_2a.csv`,
funnel in `data/processed/eva71_2a.curation.json`]:

1. Drop complexes flagged `suspected_artefact` or failing `pb_valid_prepared`.
2. Group by `compound_group`; discard any compound whose replicate pK_D spread
   exceeds 1 log unit (the fidelity gate).
3. Collapse survivors to the median pK_D; standardise structures with RDKit and
   key on InChIKey; merge any InChIKey collisions.

**Funnel** [→ `eva71_2a.curation.json`]: 649 complexes in → 10 dropped on
structure quality → 0 compounds dropped for replicate disagreement → **494
unique compounds out**, spanning **272 Bemis–Murcko scaffolds**.

**The fidelity claim is supported, and it is worth being precise about how.**
Of 494 compounds, 133 carry more than one measurement. The maximum replicate
spread across the entire dataset is **0.49 log units**, and **no compound
exceeds the 1-log gate** [→ `eva71_2a.curation.json`, `max_replicate_spread`].
The gate therefore removed nothing — it is reported as a passed audit, not as a
filter that did work. pK_D spans 3.44–7.94 (median 4.95, SD 0.86).

A consequence for interpretation: the dynamic range is narrow (about 4.5 log
units, mostly compressed between 4 and 6) and the assay is a fragment-screening
campaign, so most compounds are weak binders. A model that predicts the mean
well but ranks poorly will look deceptively good on RMSE. This is why Spearman ρ
is reported alongside RMSE throughout.

### 3.3 Splits

Splits are materialised to JSON and read identically by every arm, so no
experiment can reshuffle its own test set [→ `scripts/build_splits.py` →
`data/processed/splits/eva71_2a/`]. Both strategies use a 0.2 / 0.1 test / val
fraction over 10 seeds, packing whole scaffold groups so that no group straddles
folds, and rejecting a group that would push a held-out fold past its quota.

- **Scaffold (primary)** — Bemis–Murcko groups assigned whole, shuffled per seed.
- **Random (reference)** — reported only to quantify optimism, per §2.

## 4. Methods

### 4.1 Arms

| Arm | Representation | Pretraining | Adaptation |
|---|---|---|---|
| `B0_median` | — | none | predicts training median (sanity floor) |
| `B1_ecfp_histgb` | ECFP4 count fingerprint, 2048 bit | none | histogram gradient boosting |
| `B2_descriptors_rf` | RDKit descriptor block | none | random forest, 500 trees |
| `T1_chemberta_linear_probe` | ChemBERTa-2 77M-MTR, mean-pooled | PubChem 77M, MTR | frozen encoder + ridge (CV-selected α) |
| `T2_chemberta_full_finetune` | ChemBERTa-2 77M-MTR tokens | PubChem 77M, MTR | full fine-tune, regression head |

`T1` versus `T2` separates representation quality from adaptation: if a frozen
encoder with a linear head matches full fine-tuning, the story is about what
pretraining encoded, not about how it was adapted.

**Deviation from protocol, recorded.** `plan.md` specifies LightGBM for `B1`.
LightGBM could not load in this environment (missing `libomp`), so `B1` uses
scikit-learn's `HistGradientBoostingRegressor`, the same algorithm class. Logged
in [`../docs/decision-log.md`](../docs/decision-log.md).

### 4.2 Learning curves

Each arm is evaluated at n ∈ {50, 100, 250, 347} training compounds, where 347
is the full training fold. The **test fold is held fixed** as n varies, so
points on a curve are directly comparable [→ `scripts/run_arms.py`].

### 4.3 Endpoints

Primary: test RMSE on pK_D. Secondary: Spearman ρ, MAE, R², precision@10%.
Reported as median over 10 seeds with the interquartile range
[→ `scripts/make_report.py` → `results/tables/table1_*.csv`].

**Data-Efficiency Ratio.** DER = N_baseline / N_transfer, the ratio of training
set sizes at which each arm first reaches a fixed target RMSE (the baseline's
full-data score), by linear interpolation along the curve. DER > 1 means
transfer bought data [→ `table2_der__*.csv`].

### 4.4 Statistics

Paired Wilcoxon signed-rank across the 10 seeds against `B1`, Holm-corrected
over the arm family [→ `table3_paired_tests__*.csv`]. Two disciplines apply:

- **Effect size is always reported beside p.** Wilcoxon ranks signs, not
  magnitudes, so a tiny but perfectly consistent difference is "significant".
  This is asserted as a unit test
  (`tests/test_stats.py::test_wilcoxon_is_sign_based_so_tiny_consistent_deltas_are_significant`).
- **Underpowered ⇒ inconclusive, not null.** With 10 seeds only large effects
  are detectable; a non-significant comparison is reported as inconclusive.

## 5. Results

*(Populated from `results/tables/` once all arms complete. No number appears in
this section that is not present in a file named in `provenance.md`.)*

### 5.1 Split audit — what random splits buy

[→ `scripts/audit_splits.py` → `results/tables/table0_split_audit.csv`]

Across all 10 seeds and both strategies, **no compound appears in both train and
test** (compound-level curation succeeded). The strategies differ in scaffold
overlap: scaffold splits share **0** Bemis–Murcko scaffolds between train and
test by construction, whereas random splits share **23–31** scaffolds. Any
random-split advantage in §5.2 is therefore attributable to train/test scaffold
overlap of that magnitude, and is reported as optimism rather than performance.

### 5.2 Learning curves

[→ `table1_learning_curves__*.csv`, `fig1_learning_curves__*.png`]

Scaffold split, median over 10 seeds, full training fold (n = 347), with
bootstrap 95% CI on the median (resampled over seeds):

<!-- TABLE:full_data START -->
| Arm | RMSE | 95% CI | Spearman ρ | R² |
|---|---|---|---|---|
| B0 median | 0.852 | [0.805, 0.924] | — | -0.045 |
| B1 ECFP4 + HistGB | 0.603 | [0.537, 0.636] | 0.670 | 0.507 |
| B2 descriptors + RF | 0.586 | [0.539, 0.647] | 0.636 | 0.513 |
| T1 ChemBERTa probe | 0.637 | [0.607, 0.661] | 0.612 | 0.421 |
| T2 ChemBERTa fine-tune | 0.662 | [0.607, 0.688] | 0.614 | 0.405 |
<!-- TABLE:full_data END -->

Every trained arm clears the B0 floor, so all are learning, and **both
pretrained arms rank below both baselines.** T1's bootstrap CI excludes B1's
median (B1 0.603 vs T1's CI lower bound 0.607), and T2's does likewise. The
ordering is not an artefact of the endpoint: across the learning curve,

<!-- TABLE:curve START -->
| Arm (RMSE ↓) | n=50 | n=100 | n=250 | n=347 |
|---|---|---|---|---|
| B0 median | 0.875 | 0.852 | 0.855 | 0.852 |
| B1 ECFP4 + HistGB | 0.704 | 0.649 | 0.584 | 0.603 |
| B2 descriptors + RF | 0.670 | 0.635 | 0.618 | 0.586 |
| T1 ChemBERTa probe | 0.730 | 0.695 | 0.643 | 0.637 |
| T2 ChemBERTa fine-tune | 1.232 | 0.960 | 0.695 | 0.662 |
<!-- TABLE:curve END -->

T1 is worse than both baselines at every size, and T2 is worse than everything
including the B0 floor at n = 50 and n = 100.

![Learning curves on the scaffold split: test RMSE against training-set size for
all five arms, median over 10 seeds with the interquartile band.](../results/figures/fig1_learning_curves__scaffold.png)

**Figure 1 — Learning curves, scaffold split.** Test RMSE (pK_D, lower better)
against training-set size, median over 10 seeds; bands are the interquartile
range. Both baselines sit below both pretrained arms at every size. **The T2
curve here is the fixed-schedule configuration**, whose n = 50 point (RMSE 1.23)
is an artefact of training 40 unchecked epochs at a learning rate an order of
magnitude too low; tuned, the same arm reaches 0.735 at n = 50. Read this panel
together with §6.2, which retracts the interpretation of that point.
[→ `scripts/make_report.py` → `fig1_learning_curves__scaffold.png`]

![Ranking ability on the scaffold split: test Spearman rho against training-set
size for the four trained arms.](../results/figures/fig2_ranking__scaffold.png)

**Figure 2 — Ranking ability, scaffold split.** Test Spearman ρ (higher better)
against training-set size, median over 10 seeds with interquartile bands. Shown
separately from Figure 1 because a narrow, mean-dominated label distribution
(§3.2) lets RMSE flatter an arm that ranks poorly. The ordering broadly matches
Figure 1, with one difference worth noting: B2 leads at n = 50 and B1 overtakes
it by n = 250. The bands overlap heavily throughout, which is the visual form of
the §5.4 finding that the two baselines are statistically indistinguishable.
[→ `fig2_ranking__scaffold.png`]

Figures are regenerated rather than tracked (`make report`); they are derived
entirely from the committed tables and metrics, and are verified byte-identical
across regeneration.

Two honest qualifications. The baselines' curves are not monotonic — B1 reaches
0.584 at n = 250 and worsens to 0.603 at n = 347 — which is an early-stopping
artefact (the validation fraction is carved from an already-small training
fold) and lies within seed noise; it is not a real reversal. And the
CI-exclusion statement above is a comparison of a bootstrap interval against a
point estimate, which is weaker evidence than the paired test in §5.4; the
paired test is what the conclusion rests on.

### 5.3 Data efficiency

[→ `table2_der__scaffold.csv`]

Taking B1's own full-data RMSE (0.603) as the target:

<!-- TABLE:der START -->
| Arm | n to reach B1's full-data RMSE | DER vs B1 |
|---|---|---|
| B0 median | never | 0.00 |
| B1 ECFP4 + HistGB | 206 | 1.00 |
| B2 descriptors + RF | 295 | 0.70 |
| T1 ChemBERTa probe | never | 0.00 |
| T2 ChemBERTa fine-tune | never | 0.00 |
<!-- TABLE:der END -->

**Neither transfer arm ever reaches the baseline's full-data score at any
training size**, so both have a data-efficiency ratio of 0 — the quantity this
study was built to measure returns its strongest negative. B2 reaches the
target at an interpolated n = 295 (DER 0.70), i.e. it needs ~43% more data than
B1 to match B1, which is a small effect and consistent with the two being
statistically indistinguishable (§5.4).

A caveat on the DER itself: it is read off a four-point curve by linear
interpolation, so the interpolated sizes (206, 295) carry more precision than
the data supports and should be read as approximate. The "never reaches"
results are robust, since they do not depend on interpolation at all.

There is no sign of the crossover that motivates pretraining: T1's deficit does
not shrink as n falls. At n = 50 — the regime where transfer is supposed to pay
off most — T1 (0.730) is worse than both B1 (0.704) and B2 (0.671), and T2
(1.232) is worse than predicting the median.

### 5.4 Paired comparisons

[→ `table3_paired_tests__scaffold.csv`]

Paired Wilcoxon across seeds vs B1, Holm-corrected over the four-arm family.
A negative delta means the arm is worse than B1.

<!-- TABLE:paired START -->
| Arm | median ΔRMSE vs B1 | p (Holm) | Verdict |
|---|---|---|---|
| B0 median | -0.2864 | 0.0078 | significantly worse |
| T1 ChemBERTa probe | -0.0487 | 0.0117 | significantly worse |
| T2 ChemBERTa fine-tune | -0.0878 | 0.0742 | inconclusive |
| B2 descriptors + RF | -0.0118 | 0.5566 | inconclusive |
<!-- TABLE:paired END -->

The frozen probe is significantly worse than the baseline. **The fine-tune is
worse but does not clear the significance threshold (p = 0.074), and with 10
seeds this is genuinely inconclusive — it is not evidence of a tie**, and we do
not report it as one.

**A methodological note that changes the conclusion.** B2 has the lower
*marginal median* (0.586 vs 0.603), which reads as "B2 is best". The paired
statistic disagrees in sign (−0.012, favouring B1) because B2 wins in only
**4 of 10 seeds**; marginal medians are computed over different per-seed
distributions and are not a matched comparison. The paired test governs, and it
says B1 and B2 are indistinguishable. Reporting marginal medians alone would
have produced a different and wrong headline.

### 5.5 Split strictness, and why RMSE cannot be compared across splits

[→ `table4_split_difficulty.csv`, `table0_split_audit.csv`]

Raw RMSE *improves* on the strictest split — B1 scores 0.554 under Butina
versus 0.603 under scaffold — which would absurdly imply Butina is easier. It
is an artefact: stricter clustering concentrates chemically similar,
similarly-active compounds, so the **test fold's pK_D standard deviation falls
from 0.870 (random) to 0.839 (scaffold) to 0.658 (Butina)**. B0, which learns
nothing, "improves" from 0.852 to 0.700 for the same reason. Cross-split
comparison therefore requires a variance-normalised measure.

R² at full training data (variance-normalised, therefore comparable):

<!-- TABLE:splits START -->
| Arm (R² ↑) | random | scaffold | Butina |
|---|---|---|---|
| B1 ECFP4 + HistGB | 0.492 | 0.507 | 0.204 |
| B2 descriptors + RF | 0.461 | 0.513 | 0.257 |
| T1 ChemBERTa probe | 0.422 | 0.421 | 0.056 |
| T2 ChemBERTa fine-tune | — | 0.405 | — |
<!-- TABLE:splits END -->

and the corresponding train/test similarity audit:

<!-- TABLE:audit START -->
| Split | scaffolds shared train/test | median NN Tanimoto | test cmpds with NN ≥ 0.7 |
|---|---|---|---|
| random | 24 | 0.695 | 49.0% |
| scaffold | 0 | 0.648 | 29.1% |
| butina | 13 | 0.574 | 17.3% |
<!-- TABLE:audit END -->

Two findings.

**(i) Scaffold splitting bought little over random splitting here.** Scaffold
R² is not *lower* than random R² for any arm — it is higher for B1
(0.507 vs 0.492) and B2 (0.513 vs 0.461) and level for T1 (0.421 vs 0.422) —
so the scaffold split was, if anything, marginally the easier of the two. The
audit shows why: the scaffold split shares **zero** Bemis–Murcko scaffolds with
training, yet **29% of its test compounds still have a training neighbour at
ECFP4 Tanimoto ≥ 0.7** (random: 49%; Butina: 17%). Zero scaffold overlap is not
chemical dissimilarity. This is consistent with Guo et al. (2024)'s argument,
observed here on one independent dataset — a single target is not a replication
of their 60-dataset result, and we do not claim it as one.

The same curves under the two reference splits make the effect visible:

![Learning curves on the random split.](../results/figures/fig1_learning_curves__random.png)

**Figure 3 — Learning curves, random split (optimism reference).** Same axes as
Figure 1. Reported only to quantify how much apparent performance a random split
buys; it is never the headline endpoint, because it shares 23–31 Bemis–Murcko
scaffolds between train and test (§5.1).

![Learning curves on the Butina cluster split.](../results/figures/fig1_learning_curves__butina.png)

**Figure 4 — Learning curves, Butina cluster split (stricter check).** Same axes
as Figure 1. Note the trap this panel sets: every curve sits *lower* than in
Figure 1, which would read as Butina being the easier split. It is not — the
test fold's label SD falls from 0.839 to 0.658, so RMSE is measuring a
different-variance target. The R² table above is the comparable view, and
`fig2_ranking__butina.png` shows the same ordering on a scale-free metric.
T2 was not run on this split (§7.8).

**(ii) Transfer degrades more than the baselines when the test set is genuinely
novel.** Moving scaffold → Butina, B1 retains 40% of its R² and B2 50%, while
T1 retains 13% (0.421 → 0.056), i.e. close to no skill. Spearman moves the same
way (T1 0.612 → 0.251). The direction is consistent across both baselines and
is large, but it rests on **one transfer arm on one target**, and T2 was not run
on Butina (§6), so this is a suggestive result rather than an established
property of pretrained representations.

### 5.6 Fine-tuning under the default schedule (superseded by §6.2)

> **⚠ Retraction notice.** The interpretation in this section does not survive
> the hyperparameter ablation in §6.2 and **should not be cited as a finding.**
> The numbers below are correct for the training schedule used in §5, but that
> schedule was misconfigured, and the effect it produced is an artefact of our
> setup rather than a property of low-data fine-tuning. The section is retained
> unedited because retracting a claim silently is worse than recording it.

[→ `table1_learning_curves__scaffold.csv`; 40/40 runs complete]

Under the §5 schedule (fixed 40 epochs, no validation signal), T2 shows a
dissociation between ordering and scale. At n = 50 it attains **R² = −1.26** and
RMSE 1.232, far worse than predicting the test mean (B0: 0.875), while still
achieving **Spearman ρ = 0.456**. R² recovers monotonically with data
(−1.26 → −0.24 → 0.27 → 0.40).

**What §6.2 shows.** Giving the arm validation-fold early stopping and a
learning-rate search changes the n = 50 result to RMSE 0.735 and **R² = +0.301**
— positive, not −1.26. The apparent "calibration collapse" was 40 unchecked
passes over 50 examples at a learning rate 10× too low, not an insight about
transfer learning. **We withdraw the interpretation.**

What survives §6.2 is narrower and still worth stating: fine-tuning is far more
sensitive to its training schedule than the fingerprint baselines are (§6.2),
so a low-data transfer benchmark that fixes hyperparameters across arms will
understate transfer.

Also unchanged: at full data T2 reaches RMSE 0.662 against B1's 0.603 (paired
Wilcoxon, Holm p = 0.074 — inconclusive), and Spearman 0.614 against B1's 0.670,
beating B1 on ranking in only 3 of 10 seeds.

**A recorded near-miss.** An interim read of this arm at 3 of 10 completed seeds
showed ρ = 0.680 and suggested T2 was the best-ranking arm. The completed sweep
gives ρ = 0.613; the three seeds seen first happened to include T2's single best
(seed 0, ρ = 0.737). The interim value was withheld from the results tables by
`make_report.py --require-seeds 10`.

### 5.7 Summary of findings

Scoped to this dataset, this encoder, and these arms. **Points 1–4 describe the
fixed-hyperparameter benchmark; §6.2 qualifies how far point 1 can be pushed.**

1. **Neither transfer arm was shown to improve on the baselines.** The frozen
   probe is significantly worse under the fixed schedule (Holm p = 0.012) and
   was not re-tuned. The fine-tune, once tuned, is **statistically
   indistinguishable** from the baseline at both sizes tested (n = 50: better in
   3 of 10 seeds, p = 0.16; n = 347: better in 2 of 5, p = 0.63) while costing
   ~80× more compute per fit. We claim the absence of a demonstrated benefit,
   **not** a demonstrated deficit — and note the search budget still favours the
   baselines (§6.2).
2. **The two baselines are indistinguishable** (p = 0.56), and are **not
   under-tuned**: a 32-trial search changes B1 by a median of +0.004 RMSE and
   helps in only 11 of 26 matched cells (§6.2).
3. **The frozen probe degraded more than the baselines under Butina splitting**
   (§5.5): 13% R² retained vs 40–50%. One transfer arm on one target, with the
   fine-tune untested there — suggestive, not established.
4. **Scaffold splitting was not harder than random splitting here** (§5.5).
   Scaffold-split R² exceeds random for B1 (+0.014) and B2 (+0.052) and is level
   for T1 (−0.002), despite zero shared scaffolds, because 29% of scaffold-split
   test compounds still have a near neighbour in training.
5. **Fine-tuning is far more hyperparameter-sensitive than the baselines**
   (§6.2). The same 40-epoch schedule that costs B1 nothing drives the fine-tune
   from R² +0.30 to −1.26 at n = 50. **This replaces the "learns ranking before
   calibration" claim of §5.6, which we withdraw.**
6. **Up to ~53% of test compounds could have been in pretraining** (§6.1, an
   upper bound). This cannot explain transfer losing, but it means no transfer
   advantage measured here should be taken at face value.
7. **Generic pretraining is indistinguishable from no pretraining here, and
   in-domain pretraining is not** (§6.4). Against an untrained encoder of the
   same architecture, ChemBERTa's frozen embeddings win in 5 of 10 seeds
   (p = 0.49); the in-domain arms win in 7 and 8 of 10 (p = 0.049, p = 0.010).
   The chained arm T5 beats the generic probe T1 in 9 of 10 seeds (p = 0.006),
   which supports **H3**; the pre-registered T4-vs-T1 form of H3 is
   inconclusive. None of them overtakes the fingerprint baseline.
8. **The transfer deficit is concentrated in extrapolation, not activity
   cliffs** (§6.3). Both transfer arms are significantly worse than B1 on test
   compounds with no training neighbour at Tanimoto ≥ 0.7 (T1 p = 0.018,
   T2 p = 0.019), at every threshold tested; on cliff compounds neither is
   behind, and both are nominally ahead. This corroborates point 3 by a second,
   within-split route, and now covers both transfer arms rather than one.

## 6. Ablations

### 6.1 Pretraining-corpus contamination

[→ `scripts/measure_contamination.py` → `results/tables/table5_contamination.csv`]

The protocol (§7.1 of `plan.md`) requires measuring overlap between the
pretraining corpus and our test sets. ChemBERTa-2's exact 77M-compound
pretraining set is not redistributed, so it cannot be diffed directly. PubChem,
from which that corpus was drawn, **is** queryable, and it is a strict superset
of the corpus. Membership therefore gives a genuine **upper bound**: a compound
absent from PubChem cannot have been pretrained on; one present may or may not
have been.

<!-- TABLE:contamination START -->
| Scope | Split | n | in PubChem | fraction |
|---|---|---|---|---|
| all curated compounds | - | 494 | 262 | 53.0% |
| test fold (median over seeds) | scaffold | 98 | 52 | 53.6% |
| test fold (median over seeds) | random | 98 | 53 | 54.6% |
| test fold (median over seeds) | butina | 98 | 62 | 63.8% |
<!-- TABLE:contamination END -->

Roughly **half of the curated set exists in PubChem**, and the scaffold-split
test folds are no different from the dataset as a whole (53.6% vs 53.0%), so
the splits do not concentrate or dilute potentially-seen compounds. The Butina
test folds are somewhat more enriched (63.8%).

This is a real upper bound, not a clean bill of health: up to half of each test
fold could have appeared in pretraining. Two things limit the damage. The bound
is loose — PubChem contains ~119M compounds against the corpus's 77M, and many
of these are recent Enamine catalogue entries. And the direction of the bias is
known: contamination can only *flatter* the transfer arms, so it cannot explain
a transfer arm losing. It does mean any transfer *advantage* observed here
should be treated as an upper estimate.

We did not re-pretrain on a decontaminated corpus; that remains the one part of
§7.1 not performed, and it is out of reach without the corpus itself.

### 6.2 Hyperparameter budget

[→ `scripts/tune_arms.py` → `results/tuned_metrics/`]

The benchmark in §5 uses fixed sensible defaults for every arm and never reads
the validation fold. For a negative result about transfer learning this is the
most serious threat to validity: an under-tuned transfer arm loses for the
wrong reason. We therefore re-ran arms under an explicit search budget with
**model selection on the validation fold**, as `plan.md` §5 specifies.

<!-- TABLE:tuning START -->
| Arm | n | seeds | RMSE untuned | RMSE tuned | ρ untuned | ρ tuned |
|---|---|---|---|---|---|---|
| B1 ECFP4 + HistGB | 50 | 7 | 0.709 | 0.714 | 0.615 | 0.519 |
| B1 ECFP4 + HistGB | 100 | 7 | 0.665 | 0.640 | 0.623 | 0.602 |
| B1 ECFP4 + HistGB | 250 | 6 | 0.593 | 0.611 | 0.684 | 0.688 |
| B1 ECFP4 + HistGB | 347 | 6 | 0.620 | 0.592 | 0.679 | 0.696 |
| T2 ChemBERTa fine-tune | 50 | 10 | 1.232 | 0.735 | 0.456 | 0.580 |
| T2 ChemBERTa fine-tune | 347 | 5 | 0.678 | 0.620 | 0.680 | 0.695 |
<!-- TABLE:tuning END -->

The result materially qualifies §5, and in an asymmetric way.

**The baselines were not under-tuned.** A 32-trial random search over
learning rate, tree size, leaf minimum, regularisation and iteration count
changes B1's scaffold-split RMSE by a median of +0.004 (i.e. slightly worse)
— it wins in 11 of 26 matched cells, no better than chance. The fingerprint baseline is at its
ceiling on defaults, which is what makes it a fair comparator.

**The fine-tune was badly under-tuned**, and specifically by the missing
validation signal rather than by the learning rate alone. In §5 T2 trains a
fixed 40 epochs regardless of training-set size, so at n = 50 it makes 40
unchecked passes over 50 examples. Adding validation-fold early stopping with
best-checkpoint restore, plus a learning-rate search, changes its n = 50
scaffold-split result from RMSE 1.232 / ρ 0.456 to figures competitive with the
baselines.

**The §5.6 "calibration collapse" was therefore largely an artefact of our own
training schedule, not a property of low-data fine-tuning.** Section 5.6 is
retained as written because it accurately describes the untuned configuration,
but its interpretation does not survive this ablation and should not be cited
as a finding about transfer learning.

#### What the tuned comparison shows

At n = 50 (10 seeds, 6 trials) and at full data (5 seeds, 4 trials):

| Comparison | tuned T2 | B1 | T2 better in | Wilcoxon p |
|---|---|---|---|---|
| n = 50, RMSE | 0.735 | 0.704 | 3 of 10 seeds | 0.16 |
| n = 347, RMSE | 0.620 | 0.634 | 2 of 5 seeds | 0.63 |
| n = 347, Spearman ρ | 0.695 | 0.679 | 2 of 5 seeds | — |

**Both comparisons are inconclusive.** This is a real weakening of §5. Under the
fixed schedule the fine-tune was worse than the baseline at full data
(p = 0.074, itself inconclusive); tuned, its marginal median is *lower* than
B1's (0.620 vs 0.634) while still losing on the paired statistic in 3 of 5
seeds. As in §5.4, the marginal median and the paired comparison disagree, and
the paired one governs — but at 5 seeds neither supports a claim in either
direction.

**The defensible conclusion is therefore weaker than §5 states.** Tuned, we do
not show that fine-tuning is *worse* than the baselines; we show only that it is
**not demonstrably better**, at either training-set size, while costing roughly
two orders of magnitude more compute (≈160 s vs ≈2 s per fit). The strongly
negative reading — that transfer loses — survives only for the frozen linear
probe (§5.4, Holm p = 0.012), which was **not** re-tuned here; its ridge penalty
is already selected by internal cross-validation, but that is a weaker defence
than the search the fine-tune received.

**The remaining budget asymmetry disfavours transfer, and we state it plainly.**
B1 received 32 trials; the fine-tune received 6 at n = 50 and 4 at full data,
because each of its fits costs ~160 s. This is not the equal budget `plan.md` §5
specifies. The direction matters: the arm with the *smaller* search is the
transfer arm, so a fuller search could only improve it further. Our conclusion
is consequently stated as "no demonstrated benefit", not "demonstrated deficit".
B2 and T1 were not re-tuned at all.

### 6.3 Activity cliffs, and where the transfer deficit actually lives

[→ `scripts/analyse_cliffs.py` → `results/tables/table6_activity_cliffs.csv`,
`table7_cliff_pairs.csv`, `table8_cliff_paired.csv`]

`plan.md` §7.2 asks for performance restricted to matched molecular pairs
differing by more than 1 log unit. A plain cliff / non-cliff split of the test
fold would not answer it cleanly, because "non-cliff" then pools two unrelated
kinds of compound: those the model can interpolate from close training
neighbours, and those whose neighbourhood it has never seen. Each test compound
is therefore labelled by its relationship to the **training** fold:

| Stratum | Definition |
|---|---|
| **cliff** | ≥ 1 training neighbour at Tanimoto ≥ T whose \|Δ pK_D\| > 1 |
| **smooth** | ≥ 1 training neighbour at Tanimoto ≥ T, none of them a cliff |
| **distant** | no training neighbour at Tanimoto ≥ T at all |

T = 0.7 is the primary threshold, **reused rather than newly chosen** — it is
the cut already fixed for the near-neighbour split audit (Table 0, §5.1), so
the ablation introduces no new researcher degree of freedom. 0.6 and 0.8 are
reported as sensitivity. The dataset-level census:

<!-- TABLE:cliff_pairs START -->
| Tanimoto ≥ | similar pairs | cliff pairs (Δp > 1) | cliff share of similar | compounds in ≥1 cliff |
|---|---|---|---|---|
| 0.6 | 3,433 | 827 | 24.1% | 228 / 494 |
| 0.7 | 941 | 242 | 25.7% | 102 / 494 |
| 0.8 | 133 | 42 | 31.6% | 46 / 494 |
<!-- TABLE:cliff_pairs END -->

Roughly a quarter of all near-neighbour pairs are cliffs at every threshold, so
this is a genuine feature of the dataset and not a handful of outliers.

**Raw RMSE is not comparable across strata either.** The median predictor B0,
which learns nothing, scores 1.317 on cliff compounds and 0.722 on distant ones
— cliff compounds simply carry more label variance. This is the §5.5 problem
again, one level down, so skill against B0 on the *same* compounds is reported
beside RMSE:

<!-- TABLE:cliff_strata START -->
| Arm — RMSE ↓ (skill vs B0 ↑) | cliff (n≈9) | smooth (n≈20) | distant (n≈70) | all (n≈98) |
|---|---|---|---|---|
| B0 median | 1.317 (+0.00) | 0.943 (+0.00) | 0.722 (+0.00) | 0.852 (+0.00) |
| B1 ECFP4 + HistGB | 0.833 (+0.43) | 0.417 (+0.57) | 0.584 (+0.21) | 0.603 (+0.32) |
| B2 descriptors + RF | 0.725 (+0.43) | 0.471 (+0.52) | 0.582 (+0.20) | 0.586 (+0.32) |
| T1 ChemBERTa probe | 0.745 (+0.46) | 0.519 (+0.49) | 0.642 (+0.09) | 0.637 (+0.27) |
| T2 ChemBERTa fine-tune | 0.676 (+0.49) | 0.488 (+0.40) | 0.666 (+0.07) | 0.662 (+0.25) |
<!-- TABLE:cliff_strata END -->

and the paired tests, Holm-corrected over the arm family within each stratum:

<!-- TABLE:cliff_paired START -->
| Arm — median ΔRMSE vs B1 (Holm p) | cliff | smooth | distant | all |
|---|---|---|---|---|
| B2 descriptors + RF | -0.0385 (p=0.922) | -0.0443 (p=0.129) | -0.0094 (p=0.557) | -0.0118 (p=0.557) |
| T1 ChemBERTa probe | +0.0526 (p=0.387) | -0.0517 (p=0.111) | -0.0545 (p=0.018) | -0.0487 (p=0.012) |
| T2 ChemBERTa fine-tune | +0.0368 (p=0.252) | -0.1121 (p=0.129) | -0.0954 (p=0.019) | -0.0878 (p=0.074) |
<!-- TABLE:cliff_paired END -->

**An internal consistency check first.** The `all` column reproduces §5.4
exactly — T1 −0.0487 at Holm p = 0.012, T2 −0.0878 at p = 0.074 — despite being
recomputed from saved per-compound predictions by a different script. The
stratification is a partition of the same numbers the headline rests on.

**The finding.** Both transfer arms are significantly worse than the
fingerprint baseline on **distant** compounds — T1 p = 0.018, T2 p = 0.019 —
and **neither is behind on cliffs**. On cliffs both are nominally *ahead* of
B1 (T1 +0.053, T2 +0.037), and in skill terms the fine-tune is the best arm in
the table (+0.49 against B1's +0.43). Neither cliff comparison is significant,
so the claim is the absence of a deficit, not the presence of an advantage.

This is stable where it matters. Across all three thresholds, the deficit on
**distant** is significant every time (T1 p = 0.018 / 0.018 / 0.012 at
T = 0.6 / 0.7 / 0.8; T2 p = 0.027 / 0.019 / 0.039) and the cliff comparison
never favours the baseline. The **smooth** stratum is the unstable one — both
transfer arms are significantly worse there at T = 0.6 but inconclusive at
T = 0.7 — so we claim only that the deficit is *concentrated* in extrapolation,
not that it is exclusive to it.

**Why this matters beyond the ablation.** §5.5(ii) reported that the frozen
probe degrades more than the baselines under a stricter split, but had to
qualify it: one transfer arm, and T2 was never run on Butina. §6.3 measures the
same phenomenon a second and independent way — inside a single split, by
partitioning the test fold rather than changing it — and both transfer arms
show it. Two different cuts of the data, one varying the split and one holding
it fixed, agree that the pretrained representation's weakness is novel
chemistry rather than local label roughness.

**Against the activity-cliff benchmark.** The reference point here is
[van Tilborg, Alenicheva and Grisoni (2022)](https://doi.org/10.1021/acs.jcim.2c01073)'s
MoleculeACE, which reports that descriptor- and fingerprint-based models are
frequently *better* than deep models on cliff compounds. Our result does not
reproduce that contrast: on this dataset the pretrained arms are, if anything,
the stronger ones on cliffs, and their deficit is elsewhere. Two differences
plausibly matter and we cannot separate them — their benchmark spans 30 ChEMBL
targets of lead-like compounds, while this is a single fragment screen with 9
cliff compounds per fold; and their cliff definition combines substructure and
scaffold criteria with a similarity cut, where ours is the single ECFP4
threshold already fixed by Table 0. We therefore report ours as a discrepant
observation on one dataset, not as a correction to theirs.

The intuition behind that literature is that learned representations are needed
*because* fingerprints cannot separate cliff pairs, whose ECFP4 vectors are
near-identical. This dataset is consistent with the premise — cliffs are the
hardest stratum in raw RMSE for every arm — but not with the conclusion that
pretraining is what fixes it. The pretrained arms are no worse on cliffs and
much worse elsewhere.

Two caveats, both real. A median of **9 cliff compounds per test fold** at
T = 0.7 is a thin basis for a null, and at T = 0.8 only 3 of 10 seeds clear the
five-compound floor, so the paired test is withheld there rather than run on
three seeds. And the cliff definition is similarity-based, not a fragmentation
MMP: it asks whether a close ECFP4 neighbour exists, not whether the pair
differs by a single well-defined transformation.

### 6.4 In-domain pretraining, and what generic pretraining is worth (H3)

[→ `scripts/fetch_indomain.py`, `prepare_indomain.py`, `pretrain_indomain.py`,
`analyse_indomain.py` → `results/tables/table10_indomain_contrasts.csv`,
`table11_random_encoder_draws.csv`]

H3 — that in-domain transfer beats generic self-supervised pretraining — was
listed in §7.6 as untested, because no in-domain arm existed. This section
runs it.

**The corpus.** Eight ChEMBL 3C / 3C-like protease targets. ChEMBL files
picornaviral proteases under the *whole genome polyprotein*, whose component
synonyms cover 2A, 3C, capsid and RNA polymerase alike, so a target id is no
evidence of what was assayed; every record is screened on its assay
description instead. That screen dropped **2,022 non-3C and 757 wrong-enzyme
records of 7,634**, and two otherwise-obvious targets were excluded outright —
poliovirus CHEMBL5127 is RNA-polymerase data and HCoV-NL63 CHEMBL3232683 is
the papain-like protease PLP2, a different fold. After the §3.2 fidelity gate:
**2,974 measurements on 2,743 compounds**, of which 2,829 are coronaviral and
145 picornaviral.

**Two things about this corpus have to be said before any result.** Both were
recorded in the decision log before the arms were run.

First, it is **in-domain by protein family and out-of-domain by chemistry**.
The median maximum ECFP4 Tanimoto from a 2A evaluation compound to the entire
corpus is **0.247**, and no evaluation compound has any corpus neighbour at
≥ 0.5. Median molecular weight is 329 for the evaluation set against 470 for
the corpus, and median pActivity 4.95 against 6.30: fragments versus optimised
leads. Second, `plan.md`'s transfer corpus was specified when the target was
EV-A71 **3C**; Amendment 1's change to **2A** silently demoted it from "same
protease, related viruses" to "related fold, different viruses". H3 is
therefore tested in a harder configuration than it was written for.

**The arms.** `T4` initialises the encoder randomly and multitask-pretrains it
on the corpus — in-domain pretraining *instead of* generic. `T5` starts from
ChemBERTa-2 and adapts it on the same corpus — in-domain *on top of* generic,
`plan.md`'s chained arm. Both are then frozen and probed with ridge, exactly as
`T1` is, at the same 384 dimensions, so a difference is attributable to what
the encoder was pretrained on and not to how it was adapted.

**The control that makes this readable.** `T4` is a randomly-initialised
transformer. A randomly-initialised transformer is a random projection of SMILES
tokens, and random features are a real baseline — so `T0r` takes the same
architecture, **never trains it at all**, and probes it identically. Without
`T0r`, a positive `T4` would be uninterpretable.

<!-- TABLE:indomain_curve START -->
| Arm (RMSE ↓) | n=50 | n=100 | n=250 | n=347 | ρ | R² |
|---|---|---|---|---|---|---|
| B1 ECFP4 + HistGB | 0.704 | 0.649 | 0.584 | 0.603 | 0.670 | 0.507 |
| T0r untrained encoder probe | 0.716 | 0.681 | 0.663 | 0.648 | 0.596 | 0.417 |
| T1 ChemBERTa probe | 0.730 | 0.695 | 0.643 | 0.637 | 0.612 | 0.421 |
| T2 ChemBERTa fine-tune | 1.232 | 0.960 | 0.695 | 0.662 | 0.614 | 0.405 |
| T4 in-domain probe | 0.694 | 0.701 | 0.634 | 0.603 | 0.601 | 0.440 |
| T5 chained probe | 0.744 | 0.674 | 0.636 | 0.612 | 0.634 | 0.466 |
<!-- TABLE:indomain_curve END -->

and the contrasts H3 actually needs — pairwise, pre-specified, uncorrected:

<!-- TABLE:indomain_contrasts START -->
| Contrast (full data) | median ΔRMSE | arm better in | p | verdict |
|---|---|---|---|---|
| T4 vs T1 | +0.0351 | 7/10 | 0.3750 | inconclusive |
| T5 vs T1 | +0.0252 | 9/10 | 0.0059 | arm better |
| T1 vs T0r | +0.0038 | 5/10 | 0.4922 | inconclusive |
| T2 vs T0r | -0.0050 | 4/10 | 0.6250 | inconclusive |
| T4 vs T0r | +0.0299 | 7/10 | 0.0488 | arm better |
| T5 vs T0r | +0.0299 | 8/10 | 0.0098 | arm better |
| T4 vs T5 | +0.0074 | 6/10 | 0.8457 | inconclusive |
| T4 vs B1 | -0.0471 | 2/10 | 0.0645 | inconclusive |
| T5 vs B1 | -0.0135 | 0/10 | 0.0020 | reference better |
<!-- TABLE:indomain_contrasts END -->

**H3 is supported for the chained arm and inconclusive for the pure one.**
`T5` beats `T1` at full data by 0.025 RMSE, winning in **9 of 10 seeds**
(p = 0.006), and on Spearman ρ as well (+0.028, 8 of 10, p = 0.020). `T4` beats
`T1` by a similar margin (+0.035) but in only 7 of 10 seeds (p = 0.375), which
is inconclusive. The pre-registered form of H3 is the `T4` vs `T1` contrast, and
on that exact test the answer is **not established**; on the chained variant it
is. We report both rather than the one that reads better.

**The control result is the more important one.** Generic pretraining is
**statistically indistinguishable from no pretraining at all** on this task.
`T1` beats `T0r` by 0.004 RMSE at full data, winning in 5 of 10 seeds
(p = 0.49); the fine-tune `T2` is *behind* the untrained encoder at full data
and far behind it at n = 50 (−0.534, 0 of 10 seeds, p = 0.002). The in-domain
arms, by contrast, clear the control decisively: `T4` +0.030 (7 of 10,
p = 0.049) and `T5` +0.030 (8 of 10, p = 0.010).

This is not an artefact of one lucky random initialisation. Re-drawing the
untrained encoder five times moves its median RMSE between 0.6438 and 0.6549
(SD 0.0039):

<!-- TABLE:random_draws START -->
| Random encoder draw | median RMSE over 10 eval seeds |
|---|---|
| 0 | 0.6477 |
| 1 | 0.6481 |
| 2 | 0.6526 |
| 3 | 0.6438 |
| 4 | 0.6549 |
| **median** | **0.6481** |
<!-- TABLE:random_draws END -->

`T1`'s 0.6372 sits just outside that range, so the marginal median does favour
ChemBERTa slightly — but the paired comparison across evaluation seeds, which
§5.4 established as the statistic that governs here, cannot separate them. The
honest statement is that **generic pretraining buys at most about 0.01 RMSE
over a randomly-initialised encoder of the same architecture, and in-domain
adaptation buys three times that and is significant.**

**None of it overtakes the fingerprint baseline.** `T4` matches `B1`'s marginal
median almost exactly (0.6028 vs 0.6031) but loses the paired test in 8 of 10
seeds (−0.047, p = 0.064, inconclusive); `T5` loses it in 10 of 10 (−0.014,
p = 0.002) on an effect of 0.014 RMSE — significant and practically negligible
at once, which is the sign-based Wilcoxon behaving exactly as §4.4 warns. So
the paper's headline is unchanged. What changes is the mechanism underneath it:
the failure of transfer here is **not** a failure of pretrained representations
in general, it is specifically a failure of *generic* pretraining, which on
this task does no better than random initialisation.

## 7. Limitations

Ordered by how much each one constrains the conclusions. Every item here is
either recorded in [`../docs/decision-log.md`](../docs/decision-log.md) or
follows from a number reported above.

**1. One target, one assay, one pretrained encoder.** Every result is a single
point in a three-dimensional space of (target, assay, encoder). ChemBERTa-2
77M-MTR is one chemical language model; MolFormer, Uni-Mol, graph-pretrained
encoders and domain-adapted variants are untested here, and §2 notes that
domain adaptation is precisely where the literature reports transfer gains
concentrate. **Nothing in this paper licenses a general claim about molecular
pretraining.** It constrains what to expect on a small, single-target,
single-assay regression problem of this shape.

**2. The measured protein is CVA16 2A^pro, not EV-A71 2A^pro.** Stated in §3.1
rather than deferred here because it conditions the whole study: every
"EV-A71" result, including the title, is a CVA16 result with a five-residue
extrapolation.

**3. Pretraining decontamination was bounded, not performed.** `plan.md` §7.1
makes re-pretraining on a decontaminated corpus the load-bearing ablation, and
H4 — that part of any apparent transfer gain is leakage — is the hypothesis it
was to decide. The ChemBERTa-2 77M corpus is not redistributed, so it could not
be diffed or rebuilt. What §6.1 supplies instead is a PubChem-membership
**upper bound**: up to 53% of test compounds could have been seen. **H4 is
therefore untested, not answered.** The bound's direction is the saving grace —
contamination can only flatter a transfer arm, so it cannot explain transfer
losing, and the negative result survives. The same asymmetry means the reverse
does not hold: had transfer won, this study could not have told you whether the
win was real.

**4. The search budget is unequal, and unequal in the direction that
disfavours transfer.** `plan.md` §5 specifies an identical fixed budget for
every arm. §6.2 delivers 32 trials to B1 but only 6 to the fine-tune at n = 50
and 4 at full data, because each fine-tune fit costs ~160 s against the
baseline's ~2 s. B2 and the frozen probe T1 were not re-tuned at all — and T1
is the arm carrying the one significant negative result (§5.4). Its ridge
penalty is chosen by internal cross-validation, so it is not untuned, but that
is weaker than the search its comparator received. This is why §5.7 claims **no
demonstrated benefit** rather than a demonstrated deficit.

**5. The study is powered only for large effects.** Ten seeds for the main
sweep, and five for the tuned full-data comparison. Several of the comparisons
that matter most — the fine-tune against B1 untuned (p = 0.074) and tuned
(p = 0.16 at n = 50, p = 0.63 at full data) — are inconclusive rather than null,
and are reported that way throughout (§4.4). A larger seed budget could move any
of them in either direction.

**6. Most of the pre-registered arm list was not run.** `plan.md` §4 freezes
four baselines and six transfer arms; this paper reports three baselines
(B0–B2) and two transfer arms (T1, T2). B3 (D-MPNN from scratch), T3
(alternative encoder) and T6 (ligand + ESM-2 target embedding) were not run.
T4 and T5 **were** run (§6.4), so H3 is tested — but on a corpus that Amendment
1 made harder than the protocol intended: the in-domain set was specified for
an EV-A71 3C target and is being transferred to 2A, and it is chemically
near-disjoint from the evaluation set (median nearest-neighbour Tanimoto 0.247).
H3 is supported for the chained arm T5 and inconclusive for T4, which is the
contrast `plan.md` names. A stronger in-domain corpus — picornaviral rather
than 95% coronaviral, and overlapping the fragment chemistry — could still
change the answer.

**7. The dataset is a fragment screen, and its regime is narrow.** pK_D spans
3.44–7.94 with SD 0.86, mostly compressed between 4 and 6, and the compounds
are small fragments rather than an optimised lead series. Two consequences.
Predicting near the mean scores deceptively well on RMSE, which is why
Spearman ρ is carried alongside throughout (§3.2). And the chemistry is not
where transfer is usually deployed: whether these findings hold on a
lead-optimisation series with a wide potency range is an open question this
dataset cannot answer.

**8. Split coverage is incomplete.** The temporal split of `plan.md` §3.3 was
dropped — the OpenBind release carries no per-compound year — so the
deployment-realism endpoint is absent. The fine-tune T2 was run on the scaffold
split only, at ~160 s per fit, so the Butina degradation result of §5.5(ii)
rests on the frozen probe alone; whether full fine-tuning degrades the same way
under a stricter split is untested.

**9. Four of the five §7.2 ablations are missing, for three different
reasons.** The adaptation-strategy sweep is partial: full fine-tune and linear
probe are compared (T1 vs T2), but LoRA and layer-wise unfreezing were not run
— affordable in principle, simply not done. Pretraining-corpus size and the 2A
vs 3C transfer check are **impossible here**: the first needs the corpus, the
second needs a 3C dataset, and neither exists in this environment. The fidelity
ablation is **vacuous rather than skipped**: relaxing the 1-log replicate gate
cannot change the dataset, because the gate removed nothing at all (maximum
observed spread 0.49 log units, §3.2), so there is no relaxed variant to
compare against. Of the five, only the activity-cliff ablation (§6.3) was
carried out in full.

**10. The cliff strata are small.** §6.3 rests on a median of 9 cliff compounds
per test fold at the primary threshold, against 20 smooth and 70 distant. Ten
seeds of 9 compounds is a thin basis for the claim that transfer is *not*
behind on cliffs, and at Tanimoto ≥ 0.8 only 3 of 10 seeds clear the
five-compound floor. The finding is reported as consistent across two
thresholds and mechanistically plausible, not as established.

## 8. Discussion

### 8.1 What the negative result is, stated precisely

The claim this paper supports is narrow and worth stating without slippage:
**on a 494-compound, single-target, single-assay regression problem, a
generically pretrained chemical language model did not outperform count
fingerprints with gradient boosting, and gave no data-efficiency advantage at
any training-set size tested.** The frozen probe was significantly worse
(Holm p = 0.012); the fine-tune, once given a search budget, was
indistinguishable rather than worse. Neither ever reached the baseline's
full-data RMSE, so both have a data-efficiency ratio of zero — the quantity H2
was framed to falsify.

What the paper does **not** support is the sentence it would be easiest to
extract from it. It is not evidence that molecular pretraining does not work —
and §6.4 makes that sharper rather than softer. **Generic** pretraining is what
fails here, and it fails completely: ChemBERTa's frozen representation is
statistically indistinguishable from a randomly-initialised encoder of the same
architecture. In-domain pretraining on 2,743 related-protease compounds is
distinguishable from random initialisation, and beats the generic probe. It
still does not overtake count fingerprints.

### 8.2 Where the deficit lives, and what that suggests

The more informative result is not that transfer lost but *where* it lost.
Two independent measurements agree:

- **Across splits** (§5.5): moving from scaffold to Butina clustering, the
  baselines retain 40–50% of their R² and the frozen probe retains 13%.
- **Within a single split** (§6.3): stratifying the scaffold test fold by its
  relationship to training, the probe's significant deficit is confined to
  compounds with **no near training neighbour**. On activity cliffs it is not
  behind at all — nominally ahead of the fingerprint baseline, at both
  thresholds where the comparison is powered enough to run.

These are different cuts of the data — one varies the split, one holds the
split fixed and partitions the test fold — and they point the same way. The
pretrained representation is not failing at the thing fingerprints are
structurally bad at. ECFP4 vectors for a cliff pair are nearly identical, so a
fingerprint model is close to forced into predicting the same value for both; a
learned representation is under no such constraint, and §6.3 is consistent with
it exploiting that. What it fails at is **generalising to chemistry unlike its
training fold** — precisely the regime a small project cares about.

We offer that as a direction, not a mechanism. Establishing it would need the
fine-tune stratified the same way, more than 9 cliff compounds per fold, and
more than one encoder. It does, however, suggest that "does pretraining help?"
is the wrong granularity of question, and that per-stratum reporting would
separate two effects that a single RMSE silently averages.

### 8.3 What a practitioner should do with this

Concretely, for a project with a few hundred measurements on one target:

1. **Start with ECFP4 counts + gradient boosting, or RDKit descriptors + a
   random forest.** They were statistically indistinguishable from each other
   here (p = 0.56), both beat both transfer arms, and they fit in ~2 s against
   the fine-tune's ~160 s. On this evidence a pretrained encoder is not the
   first thing to reach for.
2. **If you do evaluate a transfer arm, tune it, and tune it separately at each
   training-set size.** This is the single most consequential finding for how
   such comparisons are run: a fixed 40-epoch schedule cost the baselines
   nothing and drove the fine-tune from R² +0.30 to −1.26 at n = 50 (§6.2). A
   benchmark that fixes hyperparameters across arms will not merely understate
   transfer, it will manufacture a catastrophic-looking failure that is an
   artefact of the harness. We published such a claim internally and retracted
   it (§5.6); the retraction is the finding.
3. **Do not treat a scaffold split as the conservative one — measure it.**
   Ours shared **zero** Bemis–Murcko scaffolds with training and was still no
   harder than a random split, because 29% of its test compounds had a training
   neighbour at Tanimoto ≥ 0.7 (§5.5). Nearest-neighbour similarity is cheap to
   compute and tells you what scaffold counting does not.
4. **Never compare RMSE across splitting strategies.** Stricter splits produced
   lower-variance test folds here (label SD 0.87 → 0.66), which inverts the
   apparent difficulty ordering and would let you report a stricter split as
   easier. Use R², or skill against a median predictor fitted on the same fold.
5. **Report a data-efficiency curve, not a full-data delta.** The interesting
   claim about transfer is almost always about the low-data end, and a single
   full-data comparison cannot address it.

### 8.4 What we would need to change our minds

Two experiments, both specified in the pre-registration and neither run:

- **H3, in-domain pretraining (arm T4).** Multitask pretraining on related
  3C / 3C-like proteases, then fine-tuning on this target. This is the
  highest-value remaining experiment, because it tests the mechanism §2
  identifies as the actual source of transfer gains. A positive T4 against T1
  would convert this paper's headline from "pretraining did not help" to
  "generic pretraining did not help, in-domain pretraining did" — a materially
  stronger and more useful claim. The prior is better than it might look:
  [Sultan et al. (2025)](https://arxiv.org/abs/2503.03360) report domain
  adaptation working on sets of **≤ 4K molecules**, and report pretraining
  corpus size saturating at **400K–800K** — so the relevant scale for this
  intervention is one a single-target project can actually assemble, and is the
  same order as the corpus §6.4 builds.
- **H4, decontamination.** Not answerable with an encoder whose corpus is not
  distributed; §6.1 could only bound test-set overlap at 53%. It becomes
  answerable by *changing the design*: pretrain a smaller encoder on a corpus
  under our own control, and decontamination is available by construction
  rather than by inference.

Both would also address the asymmetry that currently limits this study's
positive claims: because contamination can only flatter a transfer arm, this
design can refute transfer but could never have credited it.

## 9. Conclusion

On a small, high-fidelity, single-target protease dataset, a generically
pretrained chemical language model did not beat count fingerprints with
gradient boosting — not at full data, not at 50 compounds, and not on any
data-efficiency measure. The frozen probe was significantly worse; the
fine-tune, properly tuned, was merely indistinguishable, at roughly two orders
of magnitude more compute per fit. For projects in this regime the classical
baseline remains the right default, and the burden of proof sits with the
pretrained model.

Three methodological points generalise further than the headline. Transfer arms
are far more sensitive to their training schedule than the baselines are, so
equalising hyperparameters across arms — the intuitive fairness move —
systematically disadvantages transfer, and cost us a claim we had to retract.
Raw RMSE is not comparable across splitting strategies, because stricter splits
change the variance of the target. And zero scaffold overlap is not chemical
novelty: our scaffold split was no harder than a random one. Where transfer's
deficit actually concentrated was not on activity cliffs but on compounds
unlike anything in the training fold — a distinction a single aggregate score
hides, and one worth reporting separately.

Finally, the scope. This is one encoder on one assay against one protein that
is itself a five-residue surrogate for the protein in the title, with two of
the four pre-registered hypotheses untested. The result constrains what to
expect in this regime; it is not a verdict on molecular pretraining, and the
experiment that would test the mechanism most likely to overturn it (§8.4) has
not been run.

## 10. Reproduction and verification

See [`provenance.md`](provenance.md) for the artefact map and command sequence.

**Tables are generated, not transcribed.** Every results table in this document
sits between `<!-- TABLE:name START/END -->` markers and is written by
`scripts/render_manuscript_tables.py` directly from `results/tables/*.csv`. A
hand-typed number cannot drift from its source, because no number is
hand-typed. `render_manuscript_tables.py --check` fails if the manuscript is
stale relative to the CSVs.

**Prose numbers are machine-checked.** `scripts/verify_manuscript.py` re-derives
every numeric claim made in the body text — dataset counts, per-arm scores,
p-values, seed-win counts, similarity fractions — from the artefacts and exits
non-zero on any mismatch. It currently checks **207 claims** across sections 3.1
through 6.4. That count is itself one of the claims: the script parses this
sentence and fails if the stated total disagrees with the number of checks it
actually ran, so the one hand-typed number in a section arguing that no number
is hand-typed cannot go stale either. It has already caught two errors: a count
of compounds with replicate measurements taken from the raw table (137) rather
than the curated one (133), and this sentence itself, left reading 85 after the
ablations of §6 added checks.

Forty-three of those checks belong to §6.3, and six of them are **cross-checks
between artefacts rather than against the prose**: the per-stratum paired tests
are recomputed from saved per-compound predictions by a different script than
the headline tests, and the `all` stratum must reproduce Table 3's deltas and
Holm p-values exactly. If the stratification ever stopped being a partition of
the numbers §5.4 rests on, the check would fail.

**Citations are machine-checked too, and were not always.** Every check above
covers numbers derived from artefacts. Citations are prose, and prose was
unguarded — which showed. An audit on 2026-09-02 re-retrieved every source in
the reference list and found **seven errors**, all of them invisible to
`make verify` as it then stood:

| Error | Consequence |
|---|---|
| A fabricated author initial (`Guo, X.` for `Guo, Q.`) | Wrong attribution in the reference list |
| The ~50-molecule crossover credited to Altae-Tran and Schimunek | It is **Snyder et al. (8)**'s, only quoted by Schimunek. This figure justifies starting the learning curves at n = 50 |
| A `[secondary]` source cited for a specific number | Breaks the rule `literature.md` itself states |
| Two references left author-less as "unretrievable" | Both had open preprints that were never looked for |
| `Schimunek et al.` for a three-author paper | Avoidable vagueness |
| The ChemBERTa-2 attribution asserted as fact | It is an inference; the checkpoint's model card is empty |
| §6.3 written without citing the activity-cliff benchmark it departs from | Fixed in §6.3 against MoleculeACE (11) |

The costly one is the fourth. [Sultan et al. (2025)](https://arxiv.org/abs/2503.03360)
is the paper motivating the in-domain arm; marked `[secondary]`, it was barred
from carrying any figure, so that arm's corpus was assembled without knowing
their reported domain-adaptation set size. An eighth error — two reference
entries briefly numbered 10 — was introduced *while fixing the other seven*.

`scripts/verify_citations.py` now runs inside `make verify` and checks
reference numbering, reading-depth labels against a fixed vocabulary, orphan
references, body citations missing from the list, reading-depth drift between
this document and `literature.md`, and the `[secondary]`-with-a-number rule.

**Every number borrowed from another paper carries the sentence that supports
it.** Structural checks cannot tell whether a source actually says what we
claim — the crossover error was found by reading a third paper's discussion,
not by any rule. So `docs/citation-claims.yaml` records, for each borrowed
quantity, the verbatim supporting sentence, the URL it was read at, and the
date. The checker then enforces that the quoted sentence **contains the number
it is cited for**, that the claim is still in this manuscript, and that a quote
read in a different work from the one cited is **declared second-hand** — which
is precisely how the crossover error hid, since that sentence lives in
Schimunek et al.'s discussion but belongs to Snyder et al. Any body sentence
citing an external source alongside a borrowed quantity must have an entry.
This does not make attribution automatic; it makes an unchecked attribution
impossible to leave implicit.
`make verify-online` additionally checks that every cited URL resolves.
`tests/test_citations.py` pins each of the eight defects above as a regression
test, so the checker is verified against the errors that motivated it rather
than only against a clean document. It found one further orphan reference on
its first run.

**The one claim we could not re-verify is now checked against primary data
instead.** §3.1's five-residue CVA16/EV-A71 statement could not be re-retrieved
during the audit — the sentence sits in bioRxiv body text, which rate-limited
us, and the paper has no PMC or Europe PMC full-text mirror. Rather than keep
trying to confirm that a paper says something,
`scripts/verify_surrogate.py` re-derives the underlying fact from the
UniProt-annotated 2A chains and `make verify` checks 13 claims against it. That
is a stronger guarantee than the citation it replaces, and it changed what §3.1
says: the count of five does not reproduce against the reviewed reference
strains (7 and 8), while the claim that matters — no catalytic residue differs
— is independently confirmed. The paper's author list and abstract figures also
re-verified cleanly via Europe PMC.

**The pipeline is bit-reproducible.** Verified by re-running from scratch and
comparing checksums:

| Stage | Result |
|---|---|
| `prepare_openbind.py` → `eva71_2a.csv` | byte-identical |
| `build_splits.py` → 30 split files | byte-identical (all 30) |
| `run_arms.py` B1 / B2 / T1 re-runs | metrics identical to < 1e-12 |
| `run_arms.py` T2 (torch fine-tune) | metrics identical to < 1e-9 |

Determinism comes from seeding Python, NumPy and torch per run
(`evapro.utils.seeding.set_seed`, with `torch.use_deterministic_algorithms`)
and from splits being read from committed files rather than recomputed.

One caveat, stated precisely: the random-forest arm is reproducible to ~1e-16
rather than bit-identically, because `n_jobs=-1` makes the order of the
floating-point reduction across threads vary between runs. This is far below
any reported precision (we report 3–4 decimals) but it is not literal bit
equality, and we do not claim it is.

```bash
make verify   # tests + table freshness + claim verification
```

**What is not reproducible from this repository alone:** the 495 MB raw release
is referenced by DOI rather than vendored, and the ChemBERTa-2 checkpoint is
fetched from HuggingFace at run time. Both are external dependencies whose
future availability we do not control.

## References

**Reading depth is labelled on every entry and the label is honest**, carried
over from the annotated review in
[`../docs/literature.md`](../docs/literature.md):

- **[full text]** the paper was retrieved and read;
- **[abstract]** only the abstract or landing page was retrieved;
- **[secondary]** known through a search summary or another paper's
  description — **never cited for a specific number**.

No claim in this manuscript is cited at a precision its reading depth does not
support. Two entries below carry no author list because retrieval was blocked;
they are listed by title rather than given an invented authorship.

### Dataset and target

1. Lithgo, R. M., Tomlinson, C. W. E., Fairhead, M., Winokan, M., Thompson, W.,
   Wild, C., Aschenbrenner, J. C., Balcomb, B. H., Marples, P. G., Chandran,
   A. V., Golding, M., Koekemoer, L., Williams, E. P., Wang, S., Ni, X.,
   MacLean, E. M., Giroud, C., Godoy, A. S., Xavier, M. A., Walsh, M. A.,
   Fearon, D., von Delft, F. (2024). *Crystallographic fragment screen of
   Coxsackievirus A16 2A protease identifies new opportunities for the
   development of broad-spectrum anti-enterovirals.* bioRxiv 2024.04.29.591684.
   <https://doi.org/10.1101/2024.04.29.591684> — **[full text]**. Source of the
   five-residue CVA16/EV-A71 surrogate statement in §3.1.

2. OpenBind Consortium (2026). *OpenBind structure–affinity data release:
   Enterovirus A71 (EV-A71) / Coxsackievirus A16 (CVA16) 2A protease.* Zenodo,
   CC0 1.0. <https://doi.org/10.5281/zenodo.20026661> — **[full text of
   record]**. The dataset analysed here. Its own reference benchmarks are
   structure-based (docking, cofolding), so the ligand-only question asked in
   this paper is complementary rather than a re-run of theirs.

### Molecular pretraining

3. Chithrananda, S., Grand, G., Ramsundar, B. (2020). *ChemBERTa: large-scale
   self-supervised pretraining for molecular property prediction.*
   arXiv:2010.09885. <https://arxiv.org/abs/2010.09885> — **[abstract]**.
   Cited for the existence and provenance of the 77M PubChem corpus, **not**
   for the encoder used here (see 4).

4. Ahmad, W., Simon, E., Chithrananda, S., Grand, G., Ramsundar, B. (2022).
   *ChemBERTa-2: towards chemical foundation models.* arXiv:2209.01712.
   <https://arxiv.org/abs/2209.01712> — **[abstract]**. **The correct citation
   for the encoder in this study**: `DeepChem/ChemBERTa-77M-MTR` is
   ChemBERTa-2's multi-task-regression variant, not the original MLM model.
   The MTR-vs-MLM comparison is not quoted numerically because the full text
   was not retrieved.

5. Sultan, A., Rausch-Dupont, M., Khan, S., Kalinina, O., Klakow, D.,
   Volkamer, A. (2025). *Transformers for molecular property prediction: domain
   adaptation efficiently improves performance.* arXiv:2503.03360; Journal of
   Cheminformatics (2026), <https://doi.org/10.1186/s13321-026-01252-z> —
   **[abstract]**, read from the open preprint. Source of the 400K–800K
   pretraining-saturation figure and the ≤ 4K-molecule domain-adaptation set
   cited in §2 and §6.4. The journal version is paywalled; an earlier draft of
   this manuscript recorded the source as unretrievable and author-less, which
   was wrong — the preprint was open throughout.

### The low-data regime

6. Altae-Tran, H., Ramsundar, B., Pappu, A. S., Pande, V. (2017). *Low data
   drug discovery with one-shot learning.* ACS Central Science 3(4), 283–293.
   <https://doi.org/10.1021/acscentsci.6b00367>; open at
   <https://ncbi.nlm.nih.gov/pmc/articles/PMC5408335> — **[abstract]**. Cited
   for the existence of one-shot approaches to this regime, **not** for the
   ~50-molecule crossover (8), which an earlier draft wrongly attributed here.

7. Schimunek, J., Luukkonen, S., Klambauer, G. (2025). *MHNfs: prompting
   in-context bioactivity predictions for low-data drug discovery.* Journal of
   Chemical Information and Modeling. DOI 10.1021/acs.jcim.4c02373; open at
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC12076497/> — **[full text via
   PMC]**. Cited for few-shot methods in this regime. **Not** the source of the
   ~50-molecule crossover: its discussion quotes that from (8).

8. Snyder, S., et al. (2024). *The Goldilocks paradigm: comparing classical
   machine learning, large language models, and few-shot learning for drug
   discovery applications.* Communications Chemistry 7.
   <https://www.nature.com/articles/s42004-024-01220-4> — **[abstract]**.
   Origin of the ~50-molecule crossover above which classical ML overtakes
   few-shot methods; our learning curves begin at exactly n = 50 for that
   reason. Reported as their finding, not re-derived here.

9. van Tilborg, D., et al. (2024). *Deep learning for low-data drug discovery:
   hurdles and opportunities.* Current Opinion in Structural Biology 86,
   102818. <https://doi.org/10.1016/j.sbi.2024.102818>; open preprint
   <https://chemrxiv.org/engage/chemrxiv/article-details/65b154a166c13817292fad82>
   — **[secondary]**. Cited only as framing.

### Evaluation and splitting

10. Guo, Q., Hernandez-Hernandez, S., Ballester, P. J. (2024). *Scaffold splits
    overestimate virtual screening performance.* arXiv:2406.00873; ICANN 2024,
    LNCS. <https://arxiv.org/abs/2406.00873> — **[full text of abstract and
    landing page]**. Source of the random < scaffold < Butina < UMAP difficulty
    ordering across 2,100 models (700 per splitting algorithm) on 60 NCI-60
    datasets, and the reason Butina clustering is reported here as the stricter
    check. Our §5.5 result is consistent with their argument on one further
    dataset; a single target is not a replication of their 60-dataset study and
    is not claimed as one.

11. van Tilborg, D., Alenicheva, A., Grisoni, F. (2022). *Exposing the
    limitations of molecular machine learning with activity cliffs.* Journal of
    Chemical Information and Modeling 62(23), 5938–5951.
    DOI 10.1021/acs.jcim.2c01073; open preprint
    <https://chemrxiv.org/engage/chemrxiv/article-details/630cc44058843b8403a19810>
    — **[abstract]**. The standard ML benchmark for activity cliffs
    (MoleculeACE); §6.3 states how our stratification differs from it and how
    our result sits against theirs. Two corrections were subsequently published,
    one for a bug that mislabelled cliff pairs across the train/test split.

### Software, models and data resources

12. RDKit: open-source cheminformatics. <https://www.rdkit.org> — fingerprints,
    descriptors, Bemis–Murcko scaffolds, standardisation.
13. Pedregosa, F., et al. (2011). *Scikit-learn: machine learning in Python.*
    JMLR 12, 2825–2830. <https://scikit-learn.org> — `HistGradientBoosting`
    (B1), `RandomForest` (B2), `RidgeCV` (T1, T4, T5).
14. Paszke, A., et al. (2019). *PyTorch: an imperative style, high-performance
    deep learning library.* NeurIPS 32. <https://pytorch.org> — arms T2, T4, T5.
15. Wolf, T., et al. (2020). *Transformers: state-of-the-art natural language
    processing.* EMNLP System Demonstrations, 38–45.
    <https://huggingface.co/docs/transformers> — encoder loading.
16. `DeepChem/ChemBERTa-77M-MTR` model checkpoint.
    <https://huggingface.co/DeepChem/ChemBERTa-77M-MTR> — the pretrained
    encoder evaluated in this study. Its model card is empty; the attribution
    to ChemBERTa-2 (4) is our inference from the checkpoint name and objective.
17. ChEMBL. Activity records for the eight 3C / 3C-like protease targets of
    §6.4, retrieved 2026-09-02 via the ChEMBL API.
    <https://www.ebi.ac.uk/chembl/> — per-target manifests in
    `data/raw/indomain_*.manifest.json`.
18. PubChem PUG REST. Kim, S., et al. (2023). *PubChem 2023 update.* Nucleic
    Acids Research 51(D1), D1373–D1380.
    <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest> — membership queries for
    the contamination upper bound of §6.1.
