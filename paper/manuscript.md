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
four training-set sizes, with 520 evaluated runs.

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
([Ahmad et al. 2022](https://arxiv.org/abs/2209.01712)); **the encoder used in
this study, `DeepChem/ChemBERTa-77M-MTR`, is the ChemBERTa-2 MTR model**, and we
cite it accordingly. Subsequent work reports that transformer advantages over
Morgan-fingerprint baselines depend more on domain-adapted pretraining than on
corpus scale, with random forests over RDKit descriptors remaining strong
baselines ([J. Cheminform. 2026](https://doi.org/10.1186/s13321-026-01252-z);
abstract only — not cited for a specific figure).

**The low-data regime.** Few-shot and one-shot approaches were developed
precisely for this setting ([Altae-Tran et al. 2017](https://arxiv.org/pdf/1611.03199);
[Schimunek et al. 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02373)),
with the reported observation that **classical ML overtakes few-shot methods
from roughly 50 measured molecules upward** — a crossover our learning curves
are designed to straddle, starting at exactly n = 50.

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
2A protease.** CVA16 2A^pro is used as an experimental surrogate; the two
sequences differ at five amino acids, none of which lie near or are predicted
to affect the active site [→ Lithgo et al. 2024, Results: "2A^pro Construct"].
This is a deliberate and documented choice by the data generators, but it means
every "EV-A71" result in this paper — including in its title — is strictly a
CVA16 result with a five-residue extrapolation. We state this here rather than
in a limitations paragraph because it conditions the entire study.

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

**(ii) Transfer degrades more than the baselines when the test set is genuinely
novel.** Moving scaffold → Butina, B1 retains 40% of its R² and B2 50%, while
T1 retains 13% (0.421 → 0.056), i.e. close to no skill. Spearman moves the same
way (T1 0.612 → 0.251). The direction is consistent across both baselines and
is large, but it rests on **one transfer arm on one target**, and T2 was not run
on Butina (§6), so this is a suggestive result rather than an established
property of pretrained representations.

### 5.6 Fine-tuning: calibration fails before ranking does

[→ `table1_learning_curves__scaffold.csv`, `table3_paired_tests__scaffold.csv`;
40/40 runs complete]

The fully fine-tuned encoder (T2) shows a clear dissociation between the two
things a regression model must do — order the compounds, and place them on the
right scale. At n = 50 it attains **R² = −1.26** and RMSE 1.232, far worse than
simply predicting the test mean (B0: 0.875), while still achieving **Spearman
ρ = 0.456**. It has learned real ordering signal while its outputs sit on the
wrong scale entirely. R² then recovers monotonically with data
(−1.26 → −0.24 → 0.27 → 0.40), i.e. calibration is what the additional data
buys, and it is still not fully bought at n = 347.

**But fine-tuning does not overtake the baselines on either metric.** At full
data T2 reaches RMSE 0.662 against B1's 0.603 (paired Wilcoxon, Holm-corrected
p = 0.074 — inconclusive, not a demonstrated tie), and Spearman 0.614 against
B1's 0.670, **beating B1 on ranking in only 3 of 10 seeds**. Like T1, it never
reaches B1's full-data RMSE at any training size, so its **data-efficiency
ratio is also 0**.

**A recorded near-miss.** An interim read of this arm at 3 of 10 completed
seeds showed ρ = 0.680 and suggested T2 was the best-ranking arm, which would
have made the paper's conclusion metric-dependent. It was not: the completed
sweep gives ρ = 0.613, and the three seeds seen first happened to include T2's
single best (seed 0, ρ = 0.737). The interim value was labelled provisional
and withheld from the results tables by `make_report.py --require-seeds 10`,
which is the mechanism that prevented it from being reported. We note it
because the failure mode — reading a partial sweep in seed order and finding
an encouraging pattern — is common and self-confirming.

### 5.7 Summary of findings

Scoped to this dataset, this encoder, and these arms:

1. **Neither transfer arm improved on the baselines.** Both have DER = 0. The
   frozen probe is significantly worse (Holm p = 0.012); the fine-tune is worse
   but **inconclusive** (p = 0.074) — we do not claim a demonstrated difference
   for it, only the absence of a demonstrated benefit. Neither shows the
   low-data advantage that motivates pretraining: at n = 50 both trail both
   baselines.
2. **The two baselines are indistinguishable** (p = 0.56). RF on RDKit
   descriptors has the lower marginal median but wins in only 4 of 10 seeds, so
   we report no winner between them.
3. **The frozen probe degraded more than the baselines under Butina splitting**
   (§5.5): 13% R² retained vs 40–50%. Direction is consistent and the magnitude
   large, but this is one transfer arm on one target, and the fine-tune was not
   run on Butina, so it is suggestive rather than established.
4. **Scaffold splitting was not harder than random splitting here** (§5.5).
   Scaffold-split R² exceeds random-split R² for B1 (+0.014) and B2 (+0.052)
   and is indistinguishable for T1 (−0.002) — i.e. never harder, despite zero
   shared scaffolds, because 29% of scaffold-split test compounds still have a
   near neighbour in training. The B2 gap is the largest and we have no
   mechanism for it beyond split-to-split variance.
5. **The fine-tune learned ranking before calibration** (§5.6). At n = 50 its R²
   is −1.26 while its Spearman matches the baseline's, so in this regime RMSE
   and rank correlation support opposite conclusions about the same model. This
   is a caution about single-metric evaluation at small n, demonstrated for one
   arm here rather than shown to be general.

## 6. Limitations

1. **Target identity.** Affinities are CVA16 2A^pro (§3.1).
2. **Scaffold split is not conservative** (§5.5, measured). Butina clustering
   is now run; UMAP clustering, which Guo et al. (2024) rank as stricter still,
   is not.
3. **No decontamination.** `plan.md` §7.1 requires measuring overlap between the
   ChemBERTa pretraining corpus and our test sets, then re-pretraining without
   it. The PubChem 77M corpus was not retrieved, so **this ablation is not
   performed**, and any transfer advantage reported here is an upper bound.
4. **Narrow dynamic range** (§3.2) flatters RMSE; read Spearman ρ alongside.
5. **Single target, single assay.** No claim generalises beyond this dataset.
6. **10 seeds** power only large effects (§4.4). The T2 vs B1 RMSE comparison
   (Holm p = 0.074) is inconclusive rather than a demonstrated tie.
7. **T2 was run on the scaffold split only** (~160 s per run on CPU), so it has
   no random- or Butina-split numbers; §5.5's cross-split claims rest on the
   other four arms.
8. **Hyperparameters were not tuned per arm.** `plan.md` §5 specifies an equal
   32-trial budget for every arm; this was not run, so all arms use sensible
   fixed defaults. Equal-budget fairness is preserved in the sense that no arm
   received tuning, but a tuned transformer might close some of the gap.

## 7. Reproduction and verification

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
non-zero on any mismatch. It currently checks **85 claims** across sections 3.2
through 5.6. It has already caught one error: a count of compounds with
replicate measurements taken from the raw table (137) rather than the curated
one (133).

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

Listed with reading-depth labels in [`../docs/literature.md`](../docs/literature.md).
