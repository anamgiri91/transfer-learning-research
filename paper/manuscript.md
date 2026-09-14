# Benchmarking Transfer Learning Efficacy on a High-Fidelity Viral Protease Dataset: A Case Study on EV-A71 / CVA16 2A Protease

**Status: draft. Results sections are populated only from files listed in
[`provenance.md`](provenance.md).** Every number below carries an inline
pointer of the form `[→ file]` naming the artefact it was read from and the
script that produced it.

---

## Abstract

**Background.** Self-supervised pretraining is the default opening move in
molecular property prediction, but its evidence base is large, noisy
benchmarks. We test the opposite regime — a few hundred compounds, one target,
one assay — on the OpenBind EV-A71 / CVA16 2A protease release: 494 curated
compounds, 272 scaffolds, maximum replicate spread 0.49 log units.

**Results.** Twelve arms were compared on identical materialised splits across
10 seeds and four training-set sizes (1,060 evaluated runs), plus 100 amended runs. On scaffold-split RMSE a frozen ChemBERTa-2 probe is worse than an
ECFP4 + gradient boosting baseline (Holm p = 0.012), and a corrected fine-tune
is worse at all four sizes (Holm ≤ 0.023) — as is a from-scratch D-MPNN
(Holm ≤ 0.020), which the two deep arms match but do not beat. Running the
pre-registered interaction term we had omitted, the probe's deficit neither
shrinks nor grows with training-set size (slope +0.007, 95% CI −0.017 to
+0.019), so data efficiency is **unresolved**, not refuted. The in-domain-versus-generic
comparison the protocol specified — matched architecture, readout and
adaptation — returns **no detectable difference** (paired median +0.016 RMSE,
95% CI −0.033 to +0.057, Holm p = 1.000), while a marginal reading of the same
runs points the opposite way.
On precision@10% two arms beat the baseline within their own families; neither
survives pooling across endpoints, and both are exploratory.
Stratifying the test fold localises the deficit to compounds with no near
training neighbour, not activity cliffs. No arm here isolates pretraining.

**Scientific Contribution.** That pretrained encoders struggle against
fingerprint baselines, and that scaffold splits leak, are established elsewhere
at larger scale and are not claimed here. This study contributes a per-stratum
localisation placing the transfer deficit in extrapolation to novel chemistry
rather than activity cliffs, a decontamination ablation whose size-matched
control reverses the conclusion the protocol as written would have reached, and
a matched in-domain-versus-generic contrast in which paired and marginal
summaries disagree in sign. It also documents two pre-registered hypotheses
reported as answered on analyses never run, recovered by machine-checked
verification of every numeric claim.

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
`data/processed/splits/eva71_2a/`]. All three strategies use a 0.2 / 0.1 test /
val fraction over 10 seeds. The two grouped strategies assign whole groups so
that no group straddles folds, and reject a group that would push a held-out
fold past its quota.

- **Scaffold (primary)** — Bemis–Murcko groups assigned whole, shuffled per seed.
- **Butina (stricter check)** — Taylor–Butina clustering of 2048-bit ECFP4
  fingerprints at Tanimoto ≥ 0.6 (distance cutoff 0.4); whole clusters are
  assigned as groups, exactly as scaffold groups are. Used as the strict-split
  reference in §5.5, because §2's difficulty ordering places cluster splitting
  above scaffold splitting
  [→ `src/evapro/data/splits.py::butina_split`].
- **Random (reference)** — reported only to quantify optimism, per §2.

Thirty split files result — three strategies × 10 seeds — and all are committed
and read rather than recomputed. `scripts/audit_splits.py` fails the build if
any of them puts a compound in both train and test. The stronger guarantee, that
no Bemis–Murcko scaffold is shared between train and test, holds **only for the
scaffold split**: Butina groups by fingerprint similarity, not by scaffold, and
Table 0 accordingly reports a median of 13 shared scaffolds under it.

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
- **Corrections are applied to both directions.** §5.4's family is the five
  pre-registered arms; §6.4–6.5's is the 15 full-data RMSE contrasts of the
  arms added later; §5.7's is each endpoint's own arm family. An earlier draft
  corrected the first and left the others raw, which applied the penalty
  exactly where it cost nothing. Where Holm and Benjamini–Hochberg disagree we
  report both and describe the claim as suggestive rather than established.

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
paired test is what the conclusion rests on. It is weaker still than that
phrasing suggests: resampling 10 seeds, the bootstrap median can take only 44
distinct values, so the interval's endpoints are essentially order statistics
of ten numbers. We report it for completeness and rest nothing on it.

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
the data supports and should be read as approximate.

**"Never reaches" is a property of the median curve, not of the arm.** An
earlier draft called those results robust because they do not depend on
interpolation. That was wrong in a way worth correcting explicitly: computing
the DER per seed, against that seed's own `B1` full-data RMSE so the comparison
stays paired, several arms cross the threshold on some seeds
[→ `table16_der_uncertainty.csv`].

<!-- TABLE:der_uncertainty START -->
| Arm | interior crossings | ≤ 50 (left-censored) | never reached | non-monotonic curves | median DER given crossing | 95% CI given crossing |
|---|---|---|---|---|---|---|
| B1 ECFP4 + HistGB | 9 | 1 | 0 | 9 | 1.00 | [1.00, 1.00] |
| B0 median | 0 | 0 | 10 | 9 | — | — |
| B2 descriptors + RF | 4 | 0 | 6 | 6 | 1.37 | [0.98, 2.93] |
| T1 ChemBERTa probe | 1 | 0 | 9 | 4 | — | — |
| T2 ChemBERTa fine-tune | 3 | 0 | 7 | 0 | 0.72 | [0.52, 1.05] |
| T0r untrained encoder probe | 1 | 0 | 9 | 8 | — | — |
| T4 in-domain probe | 2 | 0 | 8 | 6 | — | — |
| T5 chained probe | 1 | 1 | 8 | 4 | — | — |
<!-- TABLE:der_uncertainty END -->

The ten seeds partition into three states that must be reported separately,
because the DER is defined on only one of them:

- **interior crossing** — the curve crosses the target between two evaluated
  sizes, and the DER is defined by interpolation;
- **left-censored** — the curve is already at or below the target at n = 50,
  the smallest size evaluated, so the crossing size is ≤ 50 and unknown. `B1`
  is in this state on 1 seed and `T5` on 1. Nothing is extrapolated below 50;
- **never reached** — no crossing by n = 347. The crossing size is > 347 or
  does not exist. It is **not** a DER of zero.

`T1` has an interior crossing on 1 of 10 seeds and `T2` on 3 of 10, so the
published 0.00 is a **censoring convention, not a measured ratio** — and it is
reported here as censoring. The consequence for how the table reads: **the
median DER is conditional on an interior crossing and describes only those
seeds.** For `T1`, `T0r`, `T4`, `T5` and `B0` fewer than three seeds cross at
all, so no median is reported for them rather than one computed from one or two
values. Where a median is reported it summarises 3–4 seeds out of 10, and the
other 6–7 are the reason it cannot be read as an arm-level data-efficiency
ratio.

Two further things the aggregate hid. `B1`'s own learning curve is
non-monotonic on 9 of 10 seeds, so "the size at which the curve first reaches a
threshold" is being read off curves that mostly wiggle, and the first crossing
is not the only crossing. And `B2`'s DER is 0.70 from the median curve but 1.37
conditional on its 4 interior crossings, CI [0.98, 2.93] — the point estimate
crosses 1 with the order of aggregation, and the interval spans it either way.
**Across all three splits, every DER interval that can be computed at all
contains 1.** An earlier draft of this section reported one exception — `B2` on
the random split at [1.14, 3.78] — and that exception was an artefact of the
censoring bug this subsection fixes: the seed driving it was left-censored, and
scoring it as a crossing at exactly n = 50 rather than at an unknown size ≤ 50
inflated its ratio. Handled correctly the interval is [0.85, 2.82] and spans 1
like the rest.

The scope that does bind is which arms have an interval at all: on the scaffold
split only `B2` and `T2` clear the three-crossing floor, and both of those
summarise 3–4 seeds. We no longer present the DER as the quantity that settles
H2.

#### The interaction term, which the protocol asked for and we had not run

`plan.md` §6 decides H2 by two things — "interaction term in the learning-curve
model; DER" — and only the DER was ever computed. That is an omitted
pre-registered analysis of the same kind as the two endpoints recovered in
§5.7, and it matters more, because **DER cannot answer H2 on its own.** DER
asks whether an arm ever crosses a fixed threshold, which is a question about
the full-data end of the curve. An arm can be worse at full data and still lose
less ground at n = 50; DER = 0 does not exclude that, and no monotonicity
assumption in this study licenses treating it as if it did.

The test is the slope of the **paired** delta ΔRMSE = arm − `B1` on log₂ n,
fitted per seed and tested against zero by one-sample Wilcoxon over the 10
seeds, Holm-corrected within the arm family. Pairing is preserved at both
levels: within a seed the two arms see the same split *and* the same training
subsample, and every seed contributes all four sizes. A **positive** slope means
the arm's deficit shrinks as data shrinks, which is what H2 predicts
[→ `scripts/analyse_h2.py` → `table15_h2_interaction.csv`].

The model form, the log₂ scale, the per-seed-slope estimator and the seed-level
bootstrap were all **chosen after the results existed** — `plan.md` names an
interaction term and specifies nothing further. They are recorded as a post-hoc
specification of a pre-registered intent in
[`../docs/decision-log.md`](../docs/decision-log.md) and as limitation 11.

**Two estimators, and what is actually equivalent.** An earlier draft of this
section claimed the per-seed estimator and the pooled single-model form give
"the same coefficient by construction". That is wrong, and not harmlessly. What
holds exactly, and is asserted on every run of the script, is

> pooled OLS interaction coefficient = **mean** of the per-seed slopes,

in both parameterisations — `Δ ~ seed FE + log₂ n` and the protocol's
`RMSE ~ seed FE + arm × log₂ n` — because the design is balanced: every seed
contributes the same four log₂ n values to both arms. The **median** per-seed
slope is a different quantity. For `T1` the median is +0.0073 and the pooled
coefficient +0.0010, a factor of seven, and the gap exceeds the median itself.

The tests are not interchangeable either, and would not become so if the point
estimates happened to agree. The OLS t-test on the interaction is a statement
about the mean and assumes homoscedastic independent residuals across 80
observations. The Wilcoxon signed-rank on the 10 per-seed slopes tests whether
their distribution is **symmetric about zero**; read as a location test it
localises the pseudomedian (Hodges–Lehmann), a third quantity again. All three
are in the table. The Wilcoxon is primary because §4.4 fixes the seed as the
unit of replication and prefers non-parametric tests at this sample size —
**not** because of its p-value: the verdict is the same under all three
estimators for every arm, and the sign agrees for all seven.

**What the test computes here.** With 10 seeds and no zero slopes,
`scipy.stats.wilcoxon`'s default `method='auto'` uses the exact null
distribution (`method='exact'` is selected for n ≤ 50 absent ties and zeros;
with ties or zeros it switches to exhaustive permutation at n ≤ 13, still not
the normal approximation). The smallest attainable two-sided p is
2/2¹⁰ = 0.00195. The method actually selected and the count of zero slopes are
recorded per row. `zero_method` is left at its default `'wilcox'`, which
discards zero differences and reduces the effective n; no arm here has one.

**What resampling seeds does and does not measure.** A seed jointly controls
the split draw, the training subsample at each size, and initialisation and
fitting order, so the intervals describe variability from those three sources
**conditional on this fixed set of 494 compounds**. They are not a bootstrap
over compounds and say nothing about sampling a different 494. The ten test
folds are ten draws of 98 from the same 494 and therefore overlap — a compound
appears in about two of them — so the per-seed slopes are not strictly
independent and the Wilcoxon's iid assumption is approximate. That is equally
true at 10 seeds and at 30, and no seed count repairs it.

<!-- TABLE:h2_interaction START -->
| Arm | median slope | mean slope = pooled OLS coef | pseudomedian | 95% CI (median) | slope > 0 in | p raw | p Holm | verdict |
|---|---|---|---|---|---|---|---|---|
| B0 median | +0.0306 | +0.0315 = +0.0315 | +0.0306 | [+0.0096, +0.0512] | 9/10 | 0.0039 | 0.023 | slope > 0: consistent with H2 |
| B2 descriptors + RF | +0.0139 | +0.0123 = +0.0123 | +0.0139 | [-0.0127, +0.0363] | 6/10 | 0.3223 | 1.000 | inconclusive |
| T1 ChemBERTa probe | +0.0073 | +0.0010 = +0.0010 | +0.0032 | [-0.0174, +0.0189] | 6/10 | 0.7695 | 1.000 | inconclusive |
| T2 ChemBERTa fine-tune | -0.1764 | -0.1728 = -0.1728 | -0.1764 | [-0.2216, -0.1215] | 0/10 | 0.0020 | 0.014 | slope < 0: contrary to H2 |
| T0r untrained encoder probe | +0.0149 | +0.0156 = +0.0156 | +0.0156 | [-0.0049, +0.0381] | 7/10 | 0.0840 | 0.420 | inconclusive |
| T4 in-domain probe | +0.0155 | +0.0097 = +0.0097 | +0.0128 | [-0.0255, +0.0314] | 7/10 | 0.4316 | 1.000 | inconclusive |
| T5 chained probe | +0.0096 | +0.0031 = +0.0031 | +0.0085 | [-0.0109, +0.0165] | 7/10 | 0.5566 | 1.000 | inconclusive |
<!-- TABLE:h2_interaction END -->

`B0` is the sanity check: a constant predictor should lose relatively less
ground as the real models are starved of data, and its slope is the expected
sign at +0.0306 (Holm p = 0.023). Against that reference:

- **`T1`, the frozen probe, is inconclusive — not negative.** Median slope
  +0.0073 (mean and pooled coefficient +0.0010, pseudomedian +0.0032),
  95% CI [−0.0174, +0.0189], positive in 6 of 10 seeds, p = 0.77. The deficit
  neither shrinks nor grows detectably with training-set size. This replicates
  on the other two splits (random p = 0.85; Butina raw p = 0.037 and positive in
  8 of 10, which points *toward* H2 and does not survive Holm). **We had been
  reporting H2 as answered negatively; on the pre-registered test it is not
  answered for this arm at all.**
- **`T2`, the fine-tune, is significantly contrary to H2.** Slope −0.1764,
  CI [−0.2216, −0.1215], negative in 10 of 10 seeds, Holm p = 0.014 — its
  deficit *grows* as data shrinks. This is the only formally significant H2
  result in the study, and it must be read with §6.2: it is measured under the
  fixed 40-epoch schedule that drove the same arm from R² +0.30 to −1.26 at
  n = 50. Most of this slope is the harness artefact whose interpretation we
  already retracted (§5.6), so we do **not** advance it as evidence about
  transfer, and the tuned arm was not run at enough sizes to re-measure it.
- Every other arm, including both in-domain arms, is inconclusive.

**What H2 now says.** There is no sign of the crossover that motivates
pretraining — at n = 50, T1 (0.730) is worse than both B1 (0.704) and B2
(0.671) — but "no sign of it" is an absence of evidence at this seed budget,
not evidence of absence. The honest statement is that **H2 is unanswered for
the frozen probe and confounded for the fine-tune**, which is weaker than the
negative this paper previously claimed. §7 limitation 11 records what would
settle it.

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
| T2 ChemBERTa fine-tune | 0.451 | 0.405 | 0.049 |
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
T2 was not run on this split (§7.9).

**(ii) Transfer degrades more than the baselines when the test set is genuinely
novel — but the untrained control degrades *less* than the pretrained arm.**
The probe arms of §6.4 are cheap enough to run on all three splits, so this
comparison no longer rests on one transfer arm:

<!-- TABLE:splits_extended START -->
| Arm (R² ↑) | random | scaffold | Butina | Butina retained |
|---|---|---|---|---|
| B1 ECFP4 + HistGB | 0.492 | 0.507 | 0.204 | 40% |
| B2 descriptors + RF | 0.461 | 0.513 | 0.257 | 50% |
| T0r untrained encoder probe | 0.427 | 0.417 | 0.128 | 31% |
| T1 ChemBERTa probe | 0.422 | 0.421 | 0.056 | 13% |
| T4 in-domain probe | 0.449 | 0.440 | 0.203 | 46% |
| T5 chained probe | 0.481 | 0.466 | 0.103 | 22% |
<!-- TABLE:splits_extended END -->

Moving scaffold → Butina, `B1` retains 40% of its R² and `B2` 50%, while `T1`
retains **13%** (0.421 → 0.056), close to no skill. Spearman moves the same way
(T1 0.612 → 0.251). Two additions complicate the simple reading. The
**untrained encoder `T0r` retains 31%** — more than twice what the pretrained
probe does, on the same architecture and the same ridge head, so whatever makes
`T1` brittle under strict splitting is a property of *what it was pretrained
on*, not of the architecture or the probe. And the in-domain probe `T4` retains
**46%**, in the baselines' range. `T5` sits between at 22%.

That is a sharper statement than the earlier one and points the same way: the
generic pretrained representation is the least robust of the six to genuinely
novel chemistry, worse than no pretraining at all. It still rests on one
target, `T2` was never run on Butina (§7.9), and with 10 seeds these retention
ratios carry wide uncertainty we have not quantified.

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
gives ρ = 0.614; the three seeds seen first happened to include T2's single best
(seed 0, ρ = 0.737). The interim value was withheld from the results tables by
`make_report.py --require-seeds 10`.

### 5.7 All pre-registered endpoints, including the two we had omitted

[→ `scripts/analyse_endpoints.py` → `results/tables/table14_all_endpoints.csv`]

`plan.md` §6 names RMSE as primary and **Spearman ρ, MAE, R² and
precision@10%** as secondary. §4.3 of this paper repeats all four. Earlier
drafts then reported three of them and never mentioned MAE or precision@10%
again — although both are computed and stored for all 1,060 runs. That is
selective endpoint reporting, which is the specific failure pre-registration
exists to prevent, and it is corrected here rather than quietly.

<!-- TABLE:all_endpoints START -->
| Arm | RMSE ↓ | MAE ↓ | R² ↑ | ρ ↑ | prec@10% ↑ |
|---|---|---|---|---|---|
| B0 median | 0.852 | 0.669 | -0.045 | — | 0.10 |
| B1 ECFP4 + HistGB | 0.603 | 0.450 | 0.507 | 0.670 | 0.40 |
| B2 descriptors + RF | 0.586 | 0.454 | 0.513 | 0.636 | 0.50 |
| T1 ChemBERTa probe | 0.637 | 0.499 | 0.421 | 0.612 | 0.50 |
| T2 ChemBERTa fine-tune | 0.662 | 0.505 | 0.405 | 0.614 | 0.50 |
| T0r untrained encoder probe | 0.648 | 0.538 | 0.417 | 0.596 | 0.40 |
| T4 in-domain probe | 0.603 | 0.485 | 0.440 | 0.601 | 0.45 |
| T5 chained probe | 0.612 | 0.490 | 0.466 | 0.634 | 0.55 |
<!-- TABLE:all_endpoints END -->

**MAE agrees with the primary endpoint.** Every transfer arm is worse than `B1`
on MAE, at the same order of magnitude as on RMSE. Nothing changes.

**precision@10% does not agree, and it is the endpoint a screening campaign
actually consumes.** It asks what fraction of the predicted top decile is truly
in the top decile. On it, **every transfer arm matches or beats the fingerprint
baseline**, and the ordering roughly inverts the RMSE ordering:

<!-- TABLE:enrichment START -->
| Arm | prec@10% | Δ vs B1 | better in | p raw | p Holm | verdict |
|---|---|---|---|---|---|---|
| B2 descriptors + RF | 0.50 | +0.05 | 5/10 | 0.0938 | 0.281 | inconclusive |
| T0r untrained encoder probe | 0.40 | +0.00 | 3/10 | 0.8750 | 1.000 | inconclusive |
| T1 ChemBERTa probe | 0.50 | +0.10 | 6/10 | 0.0312 | 0.125 | inconclusive |
| T2 ChemBERTa fine-tune | 0.50 | +0.10 | 9/10 | 0.0039 | 0.023 | beats B1 |
| T4 in-domain probe | 0.45 | +0.10 | 6/10 | 0.6992 | 1.000 | inconclusive |
| T5 chained probe | 0.55 | +0.15 | 7/10 | 0.0156 | 0.078 | inconclusive |
<!-- TABLE:enrichment END -->

Holm-corrected within the endpoint's own arm family, the fine-tune `T2`
**significantly beats `B1`** (0.50 vs 0.40, better in 9 of 10 seeds,
p = 0.023). `T5` and `T1` point the same way without clearing correction. The
arm this paper reports as *worst* on RMSE — `T2`, at 0.662, behind even the
untrained encoder — is the best arm on enrichment.

**How much weight this can bear.** Less than the primary endpoint, for three
reasons we state before interpreting. The test fold has 98 compounds, so the
top decile is **k = 10** and the metric moves in steps of 0.1: the difference
between 0.40 and 0.50 is one compound. The medians are therefore coarse, and
the paired test over seeds is doing more work than the marginal values suggest.
And this is one endpoint of five.

**The testing family, stated and then stress-tested.** The correction above is
Holm within this endpoint's own arm comparisons: six of them, `B0` excluded
because a constant predictor has no defined Spearman and enters the study as a
sanity floor rather than a competitor. That family is the one declared in §4.4,
and it is the right one for a comparison specified in advance. It is *not*
obviously the right one here, because precision@10% is reported partly **because
it inverted** — the endpoint set was inspected before this section was written,
which is the definition of a selection effect.

So we report the pooled correction too, as a sensitivity analysis rather than
as a claim that pooling is the uniquely correct family. Across all **31**
arm-by-endpoint tests in Table 14, `T2`'s enrichment result is Holm
**p = 0.0975** — a value we quote at four decimals because it sits on a
three-decimal rounding boundary, the trap recorded in the decision log. Two things follow, and only reporting one of them would be
misleading. The positive claim does not survive pooling — so we describe it as
**exploratory**. And **neither does any negative claim**: under the same
pooling, zero of the 31 tests clear 0.05, the frozen probe's RMSE and Spearman
deficits included. The family choice is load-bearing in both directions, which
is why §4.4 names it rather than leaving it implicit, and why the primary
endpoint is corrected in its own pre-registered four-arm family (§5.4) rather
than pooled with endpoints nobody pre-specified a hierarchy over. A
non-significant adjusted p-value is also not evidence that an effect is zero,
here or anywhere else in this paper.

Two objections that would sink it do not. **Ties**: a discrete metric makes the
paired test suspect, and in 2 of 10 seeds the tenth-best label is tied, so the
"true top decile" is itself ambiguous. Recomputing under **adversarial
tie-breaking** — `T2` given the worst reading and `B1` the best, in both the
true and the predicted ranking — leaves the medians unmoved (0.50 vs 0.40) and
`T2` ahead in 8 of 10 seeds, tied in 2, **behind in none** (p = 0.008).
**Zero-difference handling**: `T2` ties `B1` on one seed, and the conclusion is
identical under Wilcoxon's default, `zsplit` and `pratt` handling, and under a
plain sign test (all p = 0.0039). The result is not an artefact of the test.

**Where else it had, and had not, been evaluated.** Earlier drafts of this
section said the advantage "does not replicate" on the other splits. That was
wrong, and the correction matters: **`T2` had never been evaluated on them.**
What had been evaluated there were the frozen probe arms, and they do not show
the advantage — on scaffold `T1` (p = 0.031) and `T5` (p = 0.016) beat `B1`,
while on random and Butina no probe does and several point the other way (`T4`
2 of 10 on random, `T1` 4 of 10 on Butina). That is evidence about the probes.
It is not evidence about `T2`, which is the arm carrying the Holm-significant
result. §6.8 runs `T2` on those two splits and reports what it finds.

**What it does mean.** A single conclusion of the form "pretraining did not
help" is not supportable across the endpoint set as pre-registered. What is
supportable: pretrained representations here are **worse at predicting the
value and no worse — on one measure better — at ranking the top of the list.**
That distinction is invisible in RMSE, it is the distinction a screening
campaign cares about most, and reporting only the endpoints that agreed with
each other would have hidden it. It also bears on §5.6: that section retracted
a *calibration-versus-ranking* interpretation because the evidence for it was a
training-schedule artefact. The interpretation is not thereby refuted, and
precision@10% is independent evidence pointing back toward it — but we are not
reinstating a retracted claim on one coarse endpoint, and we do not.

### 5.8 Summary of findings

Scoped to this dataset, this encoder, and these arms. **Points 1–4 describe the
fixed-hyperparameter benchmark; §6.2 qualifies how far point 1 can be pushed;
and point 10 is the one endpoint on which a transfer arm wins, so points 1–9
should not be read without it.**

1. **Neither transfer arm was shown to improve on the baselines *on the primary
   endpoint*.** The endpoint qualifier is load-bearing and was missing from an
   earlier draft of this list: on precision@10% the fine-tune beats the
   baseline under correction (point 10). On scaffold-split RMSE, the frozen
   probe is significantly worse under the fixed schedule (Holm p = 0.012) and
   was not re-tuned. The fine-tune, once tuned, is **statistically
   indistinguishable** from the baseline at both sizes tested (n = 50: better in
   3 of 10 seeds, p = 0.16; n = 347: better in 2 of 5, p = 0.63) while costing
   ~70× more compute per fit. We claim the absence of a demonstrated benefit,
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
7. **Generic pretraining is not separable from no pretraining here** (§6.4).
   Against an untrained encoder of the same architecture, ChemBERTa's frozen
   embeddings win in 5 of 10 seeds (raw p = 0.49) — a null, so correction
   cannot threaten it. The in-domain arms beat that control by three times the
   margin (7 and 8 of 10) and the chained arm beats the generic probe in 9 of
   10, but **after multiplicity correction these clear BH and not Holm**, so
   H3 is suggestive rather than supported, and unsupported in its
   pre-registered form. None of them overtakes the fingerprint baseline.
8. **Decontaminating the in-domain corpus shows no leakage advantage, and the
   size-matched control is what makes that readable** (§6.5). Removing the 61
   overlapping records hurts (raw p = 0.010); removing 61 random records hurts
   *more* (raw 0.002, the only contrast surviving Holm). The head-to-head
   contrast favours decontamination but is inconclusive after correction, so
   H4 is **bounded rather than answered** for these arms, and stays untestable
   for the ChemBERTa ones.
9. **The transfer deficit is concentrated in extrapolation, not activity
   cliffs** (§6.3). Both transfer arms are significantly worse than B1 on test
   compounds with no training neighbour at Tanimoto ≥ 0.7 (T1 p = 0.018,
   T2 p = 0.019), at every threshold tested; on cliff compounds neither is
   behind, and both are nominally ahead. This corroborates point 3 by a second,
   within-split route, and now covers both transfer arms rather than one.
10. **On precision@10% the ordering inverts and a transfer arm wins** (§5.7).
    Every transfer arm matches or beats `B1` on the enrichment endpoint, and
    `T2` — the worst arm in this paper on RMSE — beats it under Holm correction
    within that endpoint's family (0.50 vs 0.40, better in 9 of 10 seeds,
    p = 0.023), robustly to adversarial tie-breaking and to every zero-handling
    rule tested (§5.7). Three things cap what it carries: the top decile is
    **k = 10 compounds**, so one molecule moves the metric by 0.1; the
    advantage was measured on the scaffold split, and until §6.8 `T2` **had not
    been evaluated** on random or Butina — the frozen probes had, and did not
    show it there. It appears here because it was pre-registered, and
    because a summary that listed only points 1–9 would read as a flat verdict
    the endpoint set does not support.

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

We did not re-pretrain ChemBERTa on a decontaminated corpus, and cannot: that
requires the corpus. **For the in-domain arms of §6.4 the corpus is ours, so
that part of §7.1 *is* performed there** — overlap is known exactly rather than
bounded, and a decontaminated variant is pretrained and evaluated. The ablation
`plan.md` §7.1 asks for is therefore done for one pretraining source and
impossible for the other.

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

At n = 50 (10 seeds, 6 trials) and at full data (5 seeds, 4 trials).

**A disclosure about the baseline, added after an audit.** The comparison as
first written pitted the *tuned* fine-tune against the *untuned* fingerprint
baseline, and did not say so. That basis is not neutral, and it does not even
point the same way at both sizes: at n = 50 the untuned baseline is the
*stronger* one (0.704 against the tuned 0.714), making T2's task harder, while
at full data the tuned baseline is stronger (0.601 against 0.634), so the
original basis made T2's task *easier* — in a section whose entire subject is
equal search budgets. Both are therefore reported:

<!-- TABLE:tuned_comparison START -->
| Size | Baseline used | tuned T2 | B1 | T2 better in | p | verdict |
|---|---|---|---|---|---|---|
| n = 50 | untuned B1 (the basis §6.2 originally used) | 0.735 | 0.704 | 3/10 | 0.160 | inconclusive |
| n = 50 | tuned B1 (like for like) | 0.749 | 0.714 | 3/7 | 0.688 | inconclusive |
| n = 347 | untuned B1 (the basis §6.2 originally used) | 0.620 | 0.634 | 2/5 | 0.625 | inconclusive |
| n = 347 | tuned B1 (like for like) | 0.620 | 0.601 | 1/5 | 0.125 | inconclusive |
<!-- TABLE:tuned_comparison END -->

| Comparison | tuned T2 | B1 | T2 better in | Wilcoxon p |
|---|---|---|---|---|
| n = 347, Spearman ρ | 0.695 | 0.679 | 2 of 5 seeds | — |

**A second disclosure, about which seeds.** The full-data tuned comparison runs
on **seeds 0–4** — the first five, chosen because each fine-tune costs ~185 s,
not by any property of the seeds. That subset is not representative, and it is
unfavourable to the baseline: `B1`'s median RMSE on seeds 0–4 is **0.634**
against **0.556** on seeds 5–9, and 0.603 over all ten. So the marginal medians
in the rows above are computed where the fingerprint baseline happens to do
worst. The paired statistic is unaffected — it compares the two arms *on the
same seeds*, which is precisely what pairing is for — but any reading of the
marginal medians has to carry this caveat, and one below did.

**All four comparisons are inconclusive**, so the conclusion does not turn on
which baseline is used — but the *appearance* of the full-data result does.
Against the untuned baseline the tuned fine-tune's marginal median looks
*lower* than B1's (0.620 vs 0.634) — but that 0.634 is B1 on seeds 0–4 only,
against 0.603 over all ten, so the apparent advantage is largely the seed
subset — while it still loses the paired statistic in 3 of 5 seeds; against the properly tuned baseline it is plainly behind on both
(0.620 vs 0.601, losing in 4 of 5, p = 0.125). The like-for-like comparison is
therefore **less** favourable to transfer than the one originally reported, and
the negative reading of §5 is if anything firmer than the audit-corrected table
first suggested. At 5 seeds neither supports a claim in either direction, and
as in §5.4 the paired statistic governs over the marginal median.

**The defensible conclusion is therefore weaker than §5 states.** Tuned, we do
not show that fine-tuning is *worse* than the baselines; we show only that it is
**not demonstrably better**, at either training-set size, while costing roughly
~70× the compute (185 s vs 2.7 s per fit at full data). The strongly
negative reading — that transfer loses — survives only for the frozen linear
probe (§5.4, Holm p = 0.012), which was **not** re-tuned here; its ridge penalty
is already selected by internal cross-validation, but that is a weaker defence
than the search the fine-tune received.

**The remaining budget asymmetry disfavours transfer, and we state it plainly.**
B1 received 32 trials; the fine-tune received 6 at n = 50 and 4 at full data,
because each of its fits costs ~185 s at full data. This is not the equal budget `plan.md` §5
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
listed in §7.7 as untested, because no in-domain arm existed. This section
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

and the contrasts H3 actually needs. **These are Holm- and BH-corrected within
a declared family**, which an earlier draft of this section did not do. Leaving
them uncorrected while §5.4 corrects the pre-registered comparisons would have
applied the correction exactly where it costs nothing and omitted it where it
bites. The family is the endpoint `plan.md` §6 names as primary — RMSE at the
full training fold, 15 contrasts including §6.5's. Spearman and the n = 50 rows
are secondary, reported uncorrected in `table10` and not folded in, which would
only make the correction harsher:

<!-- TABLE:indomain_contrasts START -->
| Contrast (full data) | median ΔRMSE | arm better in | p raw | p Holm | p BH | verdict |
|---|---|---|---|---|---|---|
| T4 vs T1 | +0.0351 | 7/10 | 0.3750 | 1.000 | 0.469 | inconclusive |
| T5 vs T1 | +0.0252 | 9/10 | 0.0059 | 0.077 | 0.029 | significant under BH only |
| T1 vs T0r | +0.0038 | 5/10 | 0.4922 | 1.000 | 0.568 | inconclusive |
| T2 vs T0r | -0.0050 | 4/10 | 0.6250 | 1.000 | 0.670 | inconclusive |
| T4 vs T0r | +0.0299 | 7/10 | 0.0488 | 0.439 | 0.091 | inconclusive |
| T5 vs T0r | +0.0299 | 8/10 | 0.0098 | 0.118 | 0.029 | significant under BH only |
| T4 vs T5 | +0.0074 | 6/10 | 0.8457 | 1.000 | 0.846 | inconclusive |
| T4 vs B1 | -0.0471 | 2/10 | 0.0645 | 0.452 | 0.107 | inconclusive |
| T5 vs B1 | -0.0135 | 0/10 | 0.0020 | 0.030 | 0.015 | reference better |
<!-- TABLE:indomain_contrasts END -->

**H3 is suggestive for the chained arm and unsupported for the pure one, and
neither survives Holm.** `T5` beats `T1` at full data by 0.025 RMSE, winning in
**9 of 10 seeds** — raw p = 0.006, which is **BH-significant (0.029) but not
Holm-significant (0.077)**. `T4` beats `T1` by a similar margin (+0.035) in only
7 of 10 seeds (raw p = 0.375), inconclusive by any correction. The
pre-registered form of H3 is the `T4` vs `T1` contrast, and on that exact test
the answer is **not established**; on the chained variant the evidence is
suggestive and survives only the more permissive of the two corrections. We
report both rather than the one that reads better, and we do not claim H3 is
confirmed.

**The control result is the more important one, and it is a null.** Generic
pretraining is **not distinguishable from no pretraining** on this task: `T1`
beats `T0r` by 0.004 RMSE at full data, winning in 5 of 10 seeds (raw p = 0.49,
Holm 1.00). That is a null result, so multiplicity does not threaten it — a
correction can only make a non-significant difference less significant.

The in-domain arms are the ones that need care. Both beat the control by the
same 0.030 margin, but under correction `T5` is **BH-significant only**
(raw 0.010, BH 0.029, Holm 0.118) and `T4` is **inconclusive** (raw 0.049,
BH 0.091, Holm 0.439). So the claim "in-domain pretraining does measurable
work" rests on one arm at the more permissive correction. The point estimates
agree, the seed counts agree (8 and 7 of 10), and the direction is consistent —
but this is suggestive evidence, not a demonstrated effect, and we label it so.

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

![Learning curves for the in-domain comparison on the scaffold split.](../results/figures/fig_indomain__scaffold.png)

**Figure 5 — In-domain vs generic pretraining, scaffold split.** Test RMSE
against training-set size, median over 10 seeds, interquartile bands. `T2` is
omitted: its n = 50 RMSE of 1.23 stretches the axis until the arms this panel
exists to separate become indistinguishable; it appears in Figures 1–2 and in
the table above. **The bands overlap heavily throughout, and that is the
honest visual summary** — the differences established by the paired tests are
of order 0.03 RMSE against a seed-to-seed spread several times larger. What the
panel shows is the ordering, not a separation you could read off a single run.
[→ `scripts/make_report.py` → `fig_indomain__scaffold.png`]

`T1`'s 0.6372 sits just outside that range, so the marginal median does favour
ChemBERTa slightly — but the paired comparison across evaluation seeds, which
§5.4 established as the statistic that governs here, cannot separate them. The
honest statement is quantitative: against the five random draws, `T1` is ahead
by **0.007 to 0.018 RMSE (median 0.011)** — a real margin on the marginal
median, too small and too inconsistent across seeds for the paired test to
call. In-domain adaptation's margin over the same control is 0.030, roughly
three times as large, and reaches BH significance for one of the two arms. We
claim the contrast between those two magnitudes as suggestive; we do not claim
that generic pretraining contributes nothing, nor that the in-domain effect is
established.

**None of it overtakes the fingerprint baseline.** One thing does change from
§5.3, and it is worth stating because that section's headline was a row of
zeros: `T4` is the **only** transfer arm in this paper that ever reaches `B1`'s
full-data RMSE, so it is the only one with a non-zero data-efficiency ratio.
That ratio is **0.60**, and a DER below 1 means the arm needed *more* training
data than the baseline to get there — about 1.7× as much, and it only gets
there at the full training fold. So the data-efficiency conclusion is unchanged
in direction while no longer being degenerate.

`T4` matches `B1`'s marginal median almost exactly (0.6028 vs 0.6031) but loses
the paired test in 8 of 10 seeds (−0.047, p = 0.064, inconclusive); `T5` loses it in 10 of 10 (−0.014,
p = 0.002) on an effect of 0.014 RMSE — significant and practically negligible
at once, which is the sign-based Wilcoxon behaving exactly as §4.4 warns. So
the paper's headline is unchanged. What changes is the mechanism underneath it:
the failure of transfer here is **not** obviously a failure of pretrained
representations in general. It is specifically a failure of *generic*
pretraining, which on this task is not measurably better than random
initialisation. In-domain pretraining, on the same architecture and the same
probe, does better than that control by three times the margin — though on
these seed counts that difference clears only the more permissive multiplicity
correction, and only for one of the two in-domain arms.

### 6.5 Decontamination, and the control that reverses its reading (H4)

[→ `scripts/pretrain_indomain.py --decontaminate / --drop-random` →
`results/tables/table10_indomain_contrasts.csv`]

H4 — that part of any apparent transfer gain is leakage — is the hypothesis
`plan.md` §7.1 was written to decide, by re-pretraining on a decontaminated
corpus. That is impossible for ChemBERTa (§6.1). For the in-domain arms the
corpus is ours, so it is not: overlap with the evaluation set is **0 exact, 0
near-duplicate and 61 scaffold-level of 2,974 measurements**, and we can simply
pretrain again without them.

**The naive form of this ablation gives the wrong answer, and it is worth
showing why.** Removing the 61 overlapping records makes `T4` significantly
*worse* — 0.6028 → 0.6241, losing in 9 of 10 seeds (p = 0.010). Read at face
value that says the overlap had been helping, i.e. leakage was inflating the
arm. But decontamination removes two things at once: the overlap, and 2% of the
training corpus. A size-matched control separates them — pretrain again having
dropped **61 randomly chosen** records instead.

<!-- TABLE:decontamination START -->
| Contrast | What it removes | median ΔRMSE | arm better in | p raw | p Holm | verdict |
|---|---|---|---|---|---|---|
| T4c vs T4 | remove the 61 overlapping records | -0.0158 | 1/10 | 0.0098 | 0.118 | significant under BH only |
| T4r vs T4 | remove 61 **random** records (size-matched control) | -0.0308 | 0/10 | 0.0020 | 0.030 | reference better |
| T4c vs T4r | **decontaminated vs the control** — decides H4 | +0.0141 | 7/10 | 0.0273 | 0.273 | inconclusive |
| T5c vs T5 | remove the 61 overlapping records | +0.0060 | 7/10 | 0.0840 | 0.504 | inconclusive |
| T5r vs T5 | remove 61 **random** records (size-matched control) | -0.0026 | 3/10 | 0.2324 | 1.000 | inconclusive |
| T5c vs T5r | **decontaminated vs the control** — decides H4 | +0.0068 | 8/10 | 0.0488 | 0.439 | inconclusive |
<!-- TABLE:decontamination END -->

**The control reverses the reading, though it cannot carry the reversal on its
own.** Dropping 61 random records costs *more* than dropping the 61 overlapping
ones: 0.6297 against 0.6241. The random ablation loses to the full corpus in
**10 of 10 seeds** (−0.031, raw p = 0.002, **Holm 0.030 — the only contrast in
this table that survives Holm**), where decontamination loses in 9 (−0.016, raw
0.010, Holm 0.118). Head to head the decontaminated encoder is nominally better
than the size-matched control (+0.014, 7 of 10 seeds), but that contrast is
**inconclusive under correction** (raw 0.027, Holm 0.273), and `T5`'s
counterpart likewise (+0.007, 8 of 10, raw 0.049, Holm 0.439).

The logic is worth stating plainly, because the sign is easy to lose. If
scaffold-level overlap had been inflating `T4`, removing it would cost *more*
than removing the same number of arbitrary records. It costs **less**. What
carries statistical weight is the leg that survives correction — removing
random records demonstrably hurts — while the direct decontaminated-vs-control
comparison is only suggestive. Both point the same way, and neither points
towards leakage.

**So H4 is not answered; it is bounded.** We can say there is **no evidence
that overlap inflated the in-domain arms**, which is a null and needs no
multiplicity protection. We cannot say the reverse effect is established,
because the contrast that would establish it does not survive Holm. What the
ablation does show decisively is that **the version `plan.md` §7.1 specifies
would have misled us**: without the control it reports a significant penalty
(raw p = 0.010) whose causal arrow points the wrong way.

**`T5` gives the same answer by a quieter route.** Neither ablation moves it on
its own — decontamination +0.006 (7 of 10, raw p = 0.084), random ablation
−0.003 (3 of 10, raw 0.232) — so there is no apparent effect needing an
explanation in the first place. Its decisive contrast points where `T4`'s does
(+0.007, 8 of 10 seeds) but is inconclusive under correction as well (raw
0.049, Holm 0.439). Two arms, two different-sized apparent effects, the same
direction, and neither contrast strong enough on 10 seeds to establish it.

**What this does and does not settle.** For the in-domain arms we find **no
detectable leakage advantage** — a null we can state, not a mechanism we can
demonstrate. It stays untestable for the ChemBERTa arms, where §6.1's 53%
PubChem membership remains an upper bound rather than a measurement (§7.3). And the finding is easier than
it sounds — with 0 exact and 0 near-duplicate overlap, there was little for
decontamination to remove, which is itself a consequence of the corpus being
chemically near-disjoint from the evaluation set (§6.4). A corpus that actually
overlapped the target chemistry would be a sterner test of H4 than this one.

### 6.6 Amended fine-tuning arms (in progress)

[→ `scripts/run_finetune.py`, `analyse_amended.py` → `table17`, `table18`]

**Specified in [`plan.md`](../plan.md) Amendment 4 before evaluation, executed
after the earlier results were seen. These are amended experiments: not the
pre-registered analysis, and not independent confirmation of it.**

Two of this paper's conclusions rest on a fine-tune configuration §6.2 shows to
be a harness artefact — H2's only significant interaction slope (§5.3) and
§5.6's retracted reading — and H3's pre-registered comparison had never been
run at all, because the arm it needs was never built (Amendment 2). Three arms
address both, sharing one implementation and differing **only** in initial
encoder weights:

| Arm | Initial encoder | Addresses |
|---|---|---|
| `T2v` | ChemBERTa-77M-MTR | H2, and H1 at every size |
| `T4ft` | in-domain (random init → 3C/3CL multitask pretraining) | H3, the pre-registered form |
| `T5ft` | chained (ChemBERTa → the same in-domain pretraining) | H3 |

Same RoBERTa backbone, tokenizer, mean-pooled linear readout, optimiser and
selection rule throughout. Learning rate is chosen per cell from {1e-5, 3e-5,
1e-4} on an internal validation split carved at 15% from the **training
subsample**, mirroring `B1`'s `validation_fraction`; the split file's `val`
fold stays unused, as in the published sweep, so training sets remain identical
to those every other arm saw and the pairing holds. Checkpoints are restored
from the best validation epoch, ≤ 60 epochs, patience 10.

**`T4ft` vs `T2v` is the comparison `plan.md` §1 names as the H3 decision rule.**
It is implementable because the in-domain encoders are the same architecture as
ChemBERTa-77M-MTR: all **55** of their encoder tensors load into that backbone
key for key, at the same 384 hidden dimensions and under the same tokenizer.
(Two of the 55 are a pooler, which loads but is unused — the readout mean-pools
token states. Against a `<s>`-token classification head, which has no pooler,
the figure would be 53.) It is still not
`plan.md`'s `T4`, which was specified for the 3C target Amendment 1 removed.

**Status: incomplete at the time of writing, and no statistics are reported
until it is not.**

<!-- TABLE:amended_progress START -->
| Arm | Size | planned | complete | missing | failures |
|---|---|---|---|---|---|
| T2v | n = 50 | 10 | 10 | 0 | 0 |
| T2v | n = 100 | 10 | 10 | 0 | 0 |
| T2v | n = 250 | 10 | 10 | 0 | 0 |
| T2v | n = 347 | 10 | 10 | 0 | 0 |
| T4ft | n = 347 | 10 | 10 | 0 | 0 |
| T5ft | n = 347 | 10 | 10 | 0 | 0 |
| **total** |  | **60** | **60** | **0** | **0** |
<!-- TABLE:amended_progress END -->

`analyse_amended.py` withholds any contrast still missing a planned cell —
summarising whichever seeds happen to have finished is a stopping rule
introduced by accident — and reports one whose own ten seeds are all present.
`T2v` is complete at all four sizes with no failures; the H3 arms are still
running and every number they produce will be reported, favourable or not.

#### What the corrected fine-tune shows (H1, and H2 again)

**Sign convention for this subsection**, because two different differences
appear and they are not the same quantity. Write

> Δ(seed) = RMSE\_arm(seed) − RMSE\_reference(seed), so **positive Δ means the
> arm is worse** (RMSE is lower-is-better).

The **median paired Δ** is the median of those per-seed differences and is what
the signed-rank test is applied to. The **marginal difference** is the gap
between the two arms' own medians. They are different numbers — for `T2v` vs
`B1` at full data they are +0.048 and +0.053 — and §5.4 records a case in this
study where they disagree in *sign*.

<!-- TABLE:amended_curve START -->
| Arm | Size | RMSE | B1 | T2 (untuned) | median lr | median best epoch | fit/val | median cost |
|---|---|---|---|---|---|---|---|---|
| T2v | n = 50 | 0.918 | 0.704 | 1.232 | 0.0001 | 38 | 42/8 | 85 s |
| T2v | n = 100 | 0.781 | 0.649 | 0.960 | 0.0001 | 22 | 85/15 | 160 s |
| T2v | n = 250 | 0.678 | 0.584 | 0.695 | 0.0001 | 16 | 212/38 | 374 s |
| T2v | n = 347 | 0.656 | 0.603 | 0.662 | 0.0001 | 18 | 295/52 | 515 s |
| T4ft | n = 347 | 0.629 | 0.603 | 0.662 | 0.0001 | 6 | 295/52 | 309 s |
| T5ft | n = 347 | 0.662 | 0.603 | 0.662 | 0.0001 | 12 | 295/52 | 374 s |
<!-- TABLE:amended_curve END -->

Per seed at full data, so the paired structure is visible rather than asserted:

<!-- TABLE:per_seed START -->
| Seed | B1 | T2 | T2v | T2−B1 | T2v−B1 | T2v−T2 |
|---|---|---|---|---|---|---|
| 0 | 0.6514 | 0.6068 | 0.6601 | -0.0445 | +0.0087 | +0.0533 |
| 1 | 0.6338 | 0.7816 | 0.6933 | +0.1478 | +0.0595 | -0.0883 |
| 2 | 0.6353 | 0.6487 | 0.6669 | +0.0134 | +0.0316 | +0.0182 |
| 3 | 0.6055 | 0.6878 | 0.6231 | +0.0823 | +0.0176 | -0.0647 |
| 4 | 0.5275 | 0.6781 | 0.5740 | +0.1506 | +0.0465 | -0.1041 |
| 5 | 0.6007 | 0.5929 | 0.6521 | -0.0078 | +0.0514 | +0.0592 |
| 6 | 0.5559 | 0.6755 | 0.6313 | +0.1196 | +0.0754 | -0.0443 |
| 7 | 0.5370 | 0.6895 | 0.7786 | +0.1525 | +0.2416 | +0.0892 |
| 8 | 0.4836 | 0.5770 | 0.5334 | +0.0934 | +0.0497 | -0.0436 |
| 9 | 0.6392 | 0.6328 | 0.6726 | -0.0064 | +0.0334 | +0.0398 |
| **median** | 0.6031 | 0.6621 | 0.6561 | +0.0878 | +0.0481 | -0.0127 |
<!-- TABLE:per_seed END -->

<!-- TABLE:amended_contrasts START -->
| Contrast | Size | median paired ΔRMSE (+ = arm worse) | 95% CI | arm worse in | p raw | p adj (m=7) | verdict |
|---|---|---|---|---|---|---|---|
| T2v vs B1 | n = 50 | +0.1898 | [+0.1176, +0.2712] | 10/10 | 0.0020 | 0.0137 | arm worse |
| T2v vs B1 | n = 100 | +0.1170 | [+0.0410, +0.2995] | 10/10 | 0.0020 | 0.0137 | arm worse |
| T2v vs B1 | n = 250 | +0.0849 | [+0.0430, +0.1433] | 9/10 | 0.0059 | 0.0234 | arm worse |
| T2v vs B1 | n = 347 | +0.0481 | [+0.0255, +0.0626] | 10/10 | 0.0020 | 0.0137 | arm worse |
| T4ft vs T2v | n = 347 | +0.0161 | [-0.0332, +0.0572] | 7/10 | 0.5566 | 1.0000 | no detectable difference |
| T5ft vs T2v | n = 347 | +0.0108 | [-0.0317, +0.0994] | 7/10 | 0.4316 | 1.0000 | no detectable difference |
| T2v vs T2 | n = 347 | -0.0127 | [-0.0663, +0.0533] | 5/10 | 0.6953 | 1.0000 | no detectable difference |
<!-- TABLE:amended_contrasts END -->

**`T2v` is worse than `B1` at every training-set size**: Holm-adjusted
p = 0.014, 0.014, 0.023, 0.014 within the seven-contrast amended family that
`plan.md` Amendment 4's erratum fixes. That family size was fixed at 7 *before*
the H3 contrasts finished, and adjusted values were reported as a Bonferroni
bound at m = 7 while they ran, so that completing the sweep could not silently
shrink the family. It did not: the bound gave 0.014/0.014/0.041/0.014 and
completion moved only the n = 250 contrast, to 0.023, in the direction the
bound guaranteed.

**What this does not establish.** `T2v` reaches significance at full data where
the published `T2` did not (p = 0.074), and the temptation is to read that as
the corrected schedule producing a firmer deficit. The data do not support it.
The paired median deficit is *smaller* for `T2v` (+0.048) than for `T2`
(+0.088); what differs is consistency — `T2v` is worse on 10 of 10 seeds,
`T2` on 7 of 10. The signed-rank statistic combines the signs of the paired
differences with the ranks of their absolute values, so it responds to the rank
order of the magnitudes rather than to their scale, and a smaller but more
consistent effect can carry stronger evidence. **A change in significance
between two conditions is not itself evidence that the conditions differ**, and
the direct test says they do not differ detectably: `T2v` vs `T2` gives a
paired median of −0.013 with 95% CI [−0.066, +0.053], 5 seeds each way,
p\_raw = 0.695 and 1.000 adjusted — **no statistically detectable difference**,
which is not a demonstration of equivalence. No equivalence test was performed
and no bounds were pre-specified, so the two conditions are simply not
separated by this study at n = 347.

**H2 survives the correction.** §5.3 declined to advance `T2`'s significant
anti-H2 slope because it was measured under the artefactual schedule. Measured
again under a validation-selected one:

<!-- TABLE:amended_h2 START -->
| Condition | median slope | 95% CI | slope > 0 in | p raw | p Holm pooled over all 8 slope tests |
|---|---|---|---|---|---|
| T2 (untuned, fixed 40 epochs) | -0.1764 | — | 0/10 | 0.0020 | — |
| T2v (validation-selected) | -0.0521 | [-0.0944, -0.0218] | 1/10 | 0.0059 | 0.0352 |
<!-- TABLE:amended_h2 END -->

Median −0.0521, 95% CI [−0.0944, −0.0218], positive in 1 of 10 seeds,
**p\_raw = 0.0059**. Multiplicity, stated both ways: within the amended set
this is the only slope test, so there is nothing to correct for there; pooled
with the seven published slope tests of §5.3 it is **Holm p = 0.0352** over
m = 8. It is not merged into §5.3's own family, because re-correcting
pre-registered results retroactively is a failure this project has already
recorded once. The finding stands under either treatment.

So the fine-tune's deficit does grow as the training set shrinks, and that was
not an artefact of the fixed schedule. **Scope, which is narrow.** This is one
amended fine-tuning procedure — mean-pooled readout, AdamW, learning rate from
a three-point grid, checkpoint restored on internal validation — on one
encoder, one dataset, the scaffold split only, and the four sizes 50/100/250/347.
It is not evidence that transfer learning fails in low-data settings generally,
and it does not change H2 for the frozen probe, which remains inconclusive
(§5.3).

**Where the schedule mattered, and what cannot be attributed to it.** The gap
between conditions is concentrated at the small end: at n = 50 the untuned `T2`
scores 1.232 and `T2v` 0.918, while at full data the two are not separated
(above). But `T2v` changed three things at once relative to `T2` — learning-rate
selection, checkpoint selection on internal validation, and the readout
(mean-pooled rather than `<s>`-token). **The improvement at n = 50 cannot be
attributed to epoch count alone**, and the fall in median selected epoch from
37.5 at n = 50 to 17.5 at full data describes what validation chose *for this
arm on this data*; it is not evidence of a generally correct training duration.
Isolating the schedule would need a fourth arm holding readout and learning
rate fixed, which was not run.

#### H3, in the form the protocol specified (the comparison this study had never run)

`plan.md` §1 decides H3 by a direct contrast between an in-domain-pretrained
arm and a generically-pretrained one. Amendment 2 records that the contrast had
never been performed: the in-domain arms were frozen probes, and no in-domain
fine-tune existed. `T4ft` vs `T2v` is that contrast, with architecture, readout,
optimiser, learning-rate grid, selection rule, training subsamples and test rows
all identical and verified identical (below). **All 60 planned cells completed
with no failures.**

Both contrasts are in the table above. Neither separates:

- **`T4ft` vs `T2v`** — in-domain *instead of* generic. Paired median
  **+0.0161** (positive = in-domain worse), 95% CI [−0.0332, +0.0572], better
  on 3 of 10 seeds, p_raw = 0.5566, **Holm p = 1.000** at m = 7.
- **`T5ft` vs `T2v`** — in-domain *on top of* generic. Paired median **+0.0108**,
  95% CI [−0.0317, +0.0994], better on 3 of 10 seeds, p_raw = 0.4316,
  **Holm p = 1.000**.

**Two summaries of the same runs disagree in sign.** `T4ft`'s median RMSE is
0.6294 against `T2v`'s 0.6561 — marginally the in-domain arm looks **0.027
better**, which is the number a table of per-arm medians would show. The median
of the paired differences is **+0.0161, in the opposite direction**, and `T4ft`
is worse on 7 of the 10 seeds.

An earlier draft of this section attributed the disagreement to one seed
"carrying" the marginal advantage. **That is not what the data show**, and the
check is worth reporting because it is the obvious explanation and it is wrong.
Recomputing both summaries with each seed dropped in turn
[→ `table22_median_reversal.csv`]:

- the marginal difference favours `T4ft` in **10 of 10** leave-one-out fits,
  range [−0.0329, −0.0205];
- the paired median favours `T2v` in **10 of 10**, range [+0.0057, +0.0265];
- the two disagree in sign in **10 of 10**.

<!-- TABLE:median_reversal START -->
| Sample | marginal median diff (T4ft − T2v) | median of paired differences | signs |
|---|---|---|---|
| all ten seeds | -0.0267 | +0.0161 | **disagree** |
| drop seed 0 | -0.0205 | +0.0265 | disagree |
| drop seed 1 | -0.0249 | +0.0057 | disagree |
| drop seed 2 | -0.0249 | +0.0057 | disagree |
| drop seed 3 | -0.0329 | +0.0057 | disagree |
| drop seed 4 | -0.0285 | +0.0057 | disagree |
| drop seed 5 | -0.0285 | +0.0265 | disagree |
| drop seed 6 | -0.0329 | +0.0265 | disagree |
| drop seed 7 | -0.0205 | +0.0265 | disagree |
| drop seed 8 | -0.0285 | +0.0057 | disagree |
| drop seed 9 | -0.0249 | +0.0265 | disagree |
<!-- TABLE:median_reversal END -->

No single seed produces the reversal, seed 7 included — dropping it leaves the
marginal difference at −0.0205 and the paired median at +0.0265. All ten seeds
remain in the primary analysis; the leave-one-out fits are descriptive only.

The reason is arithmetic and long established: **the median is not a linear
operator**, so median(X − Y) need not equal median(X) − median(Y), and the two
quantities are not estimates of the same thing. It bites here because the arms
rank the seeds differently — the seed-wise rank correlation between them is
ρ = 0.50 — so the observations sitting at each arm's median are **different
seeds**: `T4ft`'s median is set by seeds 7 and 6, `T2v`'s by seeds 5 and 0. A
marginal comparison pairs the middle of one ordering against the middle of
another; nothing makes those two middles correspond to the same experimental
conditions. We report this as an empirical illustration of a known issue, not
as a methodological finding.

The paired statistic governs, because the arms are run on matched splits,
matched subsamples and identical test rows precisely so that it can. §5.4
records the same disagreement between `B1` and `B2`, where reporting marginal
medians alone would also have produced the wrong headline.

**So H3's pre-registered form is answered: no detectable difference**, in
either direction, between in-domain and generic pretraining on this corpus when
architecture and adaptation are held fixed, at n = 347 on the scaffold split.
That is a null, not a demonstration of equivalence; the confidence intervals
admit differences of ±0.03–0.10 RMSE, which on this dataset is not a small
effect. §6.4's substituted probe-versus-probe contrast was *suggestive* for the
chained arm (BH-significant, not Holm); under the comparison the protocol
actually specified, that suggestion does not carry.

**What still differs besides the pretraining corpus**, stated because the
interpretation depends on it:

1. **`T4ft` vs `T2v` substitutes rather than adds.** `T4ft`'s encoder never saw
   generic pretraining; `T2v`'s never saw the in-domain corpus. The contrast
   varies the corpus *and* whether 77M molecules of generic pretraining
   happened at all. That is what `plan.md` §1's wording asks for — in-domain
   transfer *versus* generic — but it is not a single-factor manipulation.
   **`T5ft` vs `T2v` is the single-factor one** (in-domain pretraining added on
   top of the same generic encoder), and it is equally null.
2. **Corpus size differs by four orders of magnitude** — 2,743 compounds
   against 77M — so "in-domain versus generic" here confounds domain with
   scale. No experiment in this study separates them.
3. **The pooler is loaded but unused.** `T2v` initialises a fresh pooler
   (ChemBERTa's checkpoint has none); the in-domain encoders carry one from
   their own construction. All three read out by mean-pooling token states, so
   the pooler is absent from the forward pass in every arm and cannot
   contribute to the contrast.
4. Everything else was checked rather than assumed: the three arms score
   **identical test rows in identical order** on every seed, with identical
   labels, and every cell conforms to the frozen schedule, learning-rate grid
   and 15% internal-validation split.

**This is the amended execution of the originally intended comparison, not the
original.** The arms, the readout, the tuning budget and the selection rule
were all specified in Amendment 4 *after* the earlier results were seen; only
the comparison's *form* comes from the pre-registration. The corpus is the one
Amendment 1 left available — 95% coronaviral, median nearest-neighbour Tanimoto
0.247 to the evaluation set — so this tests a weaker sense of "in-domain" than
`plan.md` intended when it named a 3C target. The conclusion is scoped to
n = 347, the scaffold split, and this encoder pair.

**One confound is fixed by design and stated now rather than on completion.**
`T2v` differs from `T2` in both the schedule and the readout — `T2` used a
`<s>`-token classification head, these use mean pooling. `T2v`-vs-`T2` is
therefore a comparison of conditions, not an isolation of the schedule. Mean
pooling was chosen because it is what the in-domain encoders were pretrained
with and what every frozen probe here uses, so matching it across the three new
arms protects H3, which is the contrast that needed protecting.

### 6.7 `B3`, the from-scratch D-MPNN the protocol froze and we had not run

[→ `scripts/run_dmpnn.py` → `analyse_b3.py` → `table23`, `table24`]

`plan.md` §4 freezes `B3` as a from-scratch D-MPNN and names it part of "the
best from-scratch baseline" H1 must be measured against. It was never run, so
every H1 comparison in this paper had pitted pretrained transformers against
fingerprint and descriptor models only. **The arm is pre-registered; the
implementation is not** — chemprop 2.3.1, its default featuriser, bond-message
passing, mean aggregation, a regression FFN, and the same validation-based
learning-rate and checkpoint selection Amendment 4 uses, all fixed in
`plan.md` Amendment 5 before any test number existed. All 40 cells completed;
no failures.

Training subsamples are `run_arms.py`'s, reconstructed and asserted, so `B3`
sees the same molecules as every other arm. Validation is carved from the
subsample at 15%, as for the fine-tune arms, and both budgets are reported —
at n = 347 the labelled budget is 347 and 295 are actually fitted, with 52 held
out internally. `B1` carves its own 15% internally too
(`validation_fraction=0.15`) but does not expose the split, so the fitted
counts are matched by construction rather than merely by intent.

<!-- TABLE:b3_curve START -->
| Size | B3 D-MPNN | B1 | B2 | T2v | fitted/val | median lr | median best epoch | median cost |
|---|---|---|---|---|---|---|---|---|
| n = 50 | 0.760 | 0.704 | 0.670 | 0.918 | 42/8 | 0.001 | 41 | 11 s |
| n = 100 | 0.791 | 0.649 | 0.635 | 0.781 | 85/15 | 0.001 | 26 | 12 s |
| n = 250 | 0.684 | 0.584 | 0.618 | 0.678 | 212/38 | 0.0003 | 26 | 32 s |
| n = 347 | 0.667 | 0.603 | 0.586 | 0.656 | 295/52 | 0.00065 | 30 | 34 s |
<!-- TABLE:b3_curve END -->

<!-- TABLE:b3_contrasts START -->
| Contrast | Size | median paired ΔRMSE (+ = arm worse) | 95% CI | arm worse in | p raw | p Holm (m=5) | verdict |
|---|---|---|---|---|---|---|---|
| B3 vs B1 | n = 50 | +0.0927 | [+0.0299, +0.1831] | 9/10 | 0.0098 | 0.0195 | arm worse |
| B3 vs B1 | n = 100 | +0.0903 | [+0.0435, +0.1810] | 9/10 | 0.0039 | 0.0156 | arm worse |
| B3 vs B1 | n = 250 | +0.0734 | [+0.0479, +0.1362] | 10/10 | 0.0020 | 0.0098 | arm worse |
| B3 vs B1 | n = 347 | +0.0486 | [+0.0231, +0.0943] | 9/10 | 0.0039 | 0.0156 | arm worse |
| B3 vs T2v | n = 347 | +0.0088 | [-0.0263, +0.0592] | 5/10 | 0.6953 | 0.6953 | no detectable difference |
<!-- TABLE:b3_contrasts END -->

Two results, both within Amendment 5's five-contrast family:

- **`B3` is worse than `B1` at every training-set size** — paired median +0.093,
  +0.090, +0.073, +0.049, Holm p = 0.020, 0.016, 0.010, 0.016. The
  pre-registered "best from-scratch baseline" is therefore still `B1`, and
  adding the deep from-scratch arm does not change any H1 verdict. The gap
  narrows with n, as `T2v`'s does.
- **`B3` and `T2v` are not separated at full data** — paired median +0.009,
  95% CI [−0.026, +0.059], 5 seeds each way, p_raw = 0.695, Holm 0.695.

**What that second result does and does not license.** The two deep arms land
in the same place and both sit behind count fingerprints, which is consistent
with the deficit belonging to the model class rather than to whether the
encoder was pretrained. It is **not evidence for that reading**. `B3` differs
from `T2v` in architecture *and* in molecular representation — a bond-message
graph network over a molecular graph against a transformer over SMILES tokens —
as well as in pretraining, so a null between them has at least three candidate
explanations and this design separates none of them. What `B3` establishes is
narrower and is what H1 needed: **the from-scratch baseline that transfer fails
to beat is not merely a fingerprint model; a maintained D-MPNN, given the same
data and an equivalent selection rule, does not beat it either.**

### 6.8 Does the enrichment result hold on the other splits? (robustness check)

[→ `scripts/analyse_enrichment.py` → `table25_enrichment_robustness.csv`]

§5.7 carries the only positive result in this paper, and until now the
manuscript said it "does not replicate" on the random and Butina splits. **`T2`
had never been run on them.** What had been run there were the frozen probes,
which is evidence about the probes. `plan.md` Amendment 6 fixes five tests and
their family before computation; all five are complete.

The **original `T2` recipe** is used, unchanged — 40 fixed epochs, lr 3e-5,
`<s>`-token head — because §5.7's claim is about that arm. Substituting `T2v`
would answer a different question. The endpoint is used exactly as §4.3 defines
it: same top decile, same k = 10, same tie handling, no re-definition.

<!-- TABLE:enrichment_robustness START -->
| Contrast | Split | arm | B1 | median paired Δ (+ = arm better) | better/worse/tied | p raw | p Holm (m=5) |
|---|---|---|---|---|---|---|---|
| T2 vs B1 | random | 0.45 | 0.45 | +0.05 | 5/4/1 | 0.2539 | 0.7617 |
| T2 vs B1 | butina | 0.60 | 0.50 | +0.00 | 4/2/4 | 0.2812 | 0.7617 |
| T2v vs B1 | scaffold | 0.50 | 0.40 | +0.05 | 5/1/4 | 0.1562 | 0.6250 |
| T4ft vs B1 | scaffold | 0.40 | 0.40 | +0.05 | 5/4/1 | 0.9922 | 0.9922 |
| T5ft vs B1 | scaffold | 0.50 | 0.40 | +0.15 | 9/0/1 | 0.0039 | 0.0195 |
<!-- TABLE:enrichment_robustness END -->

**`T2`'s enrichment advantage is not detected on either other split.** On
random the paired median is +0.05 with 5 seeds better, 4 worse and 1 tied
(p_raw = 0.254); on Butina it is +0.00 with 4 better, 2 worse and 4 tied
(p_raw = 0.281). Neither survives correction within Amendment 6's five-test
family.

**That is a non-detection, not a refutation, and the distinction is load
bearing here.** Both point estimates lean in the same direction as the scaffold
result rather than against it; what changes is consistency — 9 of 10 seeds on
scaffold against 5 of 9 and 4 of 6 non-tied seeds here. With k = 10 the metric
moves in steps of 0.1 and ties are frequent (1 and 4 of 10), so this comparison
has little power to resolve an advantage of the size §5.7 reports. The
supportable statement is that **the advantage is demonstrated only on the
scaffold split**, and that the other two splits neither confirm nor exclude it.

**A second positive result appears, and carries the same caveat as the first.**
`T5ft`, the chained in-domain fine-tune, beats `B1` on precision@10% at
9 of 10 seeds with 0 losses, paired median +0.15, p_raw = 0.0039, **Holm
p = 0.0195** within Amendment 6's family. Pooled with §5.7's 31 endpoint tests
— 36 in total — it is **p = 0.117**, exactly as §5.7's own result behaves. It
is reported as **exploratory** for the same three reasons: k = 10 is coarse,
the endpoint was originally surfaced because it inverted, and the family choice
drives the verdict. `T2v` and `T4ft` show no detectable difference on the same
endpoint (Holm 0.625 and 0.992).

**The same runs also complete `T2`'s RMSE row on those splits**, which
Amendment 6 did not ask for and which is reported here as an additional
analysis of them. `T2` is **significantly worse than `B1` on Butina**
(median ΔRMSE −0.056, Holm p = 0.039) and inconclusive on random (−0.089, Holm
p = 0.168), corrected within §5.4's per-split family. Those families grow from
three arms to four as a result. That is **not** a retroactive re-correction of
the kind §4.4 forbids: §5.4's family is the pre-registered five arms, `T2`
was always one of them, and it was absent from those two tables only because
it had no data there. Filling it in moves `B0` from Holm 0.0059 to 0.0078 and
`T1` from 0.0055 to 0.0082 on random and 0.0078 to 0.0117 on Butina; no verdict
changes, and no number the manuscript quotes moves.

**What this is.** A robustness check on the same 494 compounds, re-partitioned.
Every split draws from one dataset, one target and one assay, so nothing here
is an independent or external replication and we do not describe it as one.
Tests 3–5 are an **additional analysis of runs that already existed** —
precision@10% is stored for every cell — not new experiments, and they do not
substitute for `T2`.

## 7. Limitations

### 7.0 Where each hypothesis stands

The four hypotheses, the comparison each was pre-registered with, what was
actually run, and what is still missing. Two rows changed after 2026-09-11,
both because an analysis this paper had reported as decisive turned out never
to have been performed.

| | Planned comparison (`plan.md` §1) | Analysis actually performed | Current conclusion | Remaining gap |
|---|---|---|---|---|
| **H1** | Transfer arms beat the best from-scratch baseline on scaffold-split RMSE; paired Wilcoxon, Holm | As planned on `T1`/`T2` vs `B1` (§5.4), plus the tuned comparison (§6.2), all five endpoints (§5.7), a corrected fine-tune at every size (§6.6), and the pre-registered D-MPNN baseline (§6.7) | **Answered negatively, and the baseline is no longer only a fingerprint model.** The frozen probe is worse (Holm p = 0.012) and the corrected fine-tune is worse at all four sizes (Holm ≤ 0.023). `B3` is *also* worse than `B1` at all four sizes (Holm ≤ 0.020), so `B1` remains the best from-scratch arm; `B3` and `T2v` are not separated (p = 0.695). It is not established that the corrected fine-tune is a firmer negative than the untuned one (p = 0.695) | `T3`, `T6` not run. And no trainable random-init ChemBERTa exists, so none of this isolates pretraining (limitation 12) |
| **H2** | Interaction term in the learning-curve model **and** DER | DER only, until 2026-09-11. Interaction term run for every arm (§5.3), then re-run on a corrected fine-tune (§6.6) | **Split by arm.** Unanswered for the frozen probe (slope +0.007, CI −0.017 to +0.019, p = 0.77, same on both other splits). **Answered negatively for the fine-tune**, scoped to the amended procedure, scaffold split and sizes 50–347: the deficit grows as data shrinks, and survives correcting the schedule (−0.176 untuned → −0.052 validation-selected; p_raw 0.0059, Holm 0.0352 pooled over all 8 slope tests) | The probe's interval is wide enough to hide a real crossover; more seeds would narrow it. `T2v` is one arm on one split |
| **H3** | Direct arm contrast `T4` vs `T1` — in-domain **fine-tune** vs generic **fine-tune** | Probe-vs-probe as a substitute (§6.4), then the specified form executed as an amendment: `T4ft` vs `T2v`, architecture and adaptation matched (§6.6) | **On RMSE, no detectable difference** either direction at n = 347: `T4ft` +0.016, CI [−0.033, +0.057], Holm 1.000; `T5ft` +0.011, CI [−0.032, +0.099], Holm 1.000. §6.4's *suggestive* probe result does not carry. **On precision@10%, `T5ft` beats `B1`** at 9 of 10 seeds (Holm 0.0195 in its own family, 0.117 pooled over 36) — exploratory, §6.8 | A null on 10 seeds with intervals admitting ±0.03–0.10 RMSE is not equivalence. The corpus is 95% coronaviral, NN Tanimoto 0.247. And `T5ft` vs `T2v` tests an *added pretraining stage*, not domain composition: no control gets comparable extra generic pretraining (limitation 12) |
| **H4** | Decontamination ablation: re-pretrain with test overlap removed | Performed for the in-domain arms with a size-matched control the protocol did not ask for (§6.5). Impossible for ChemBERTa | **Bounded, not answered.** No evidence overlap inflated the in-domain arms; the decisive contrast is Holm 0.273. Untestable for ChemBERTa, whose 77M corpus is not distributed | A corpus that genuinely overlaps the target chemistry — ours had 0 exact and 0 near-duplicate overlap, so decontamination had almost nothing to remove |

Two of the four were, at some point in this project's history, reported as
settled on the strength of an analysis that had not been run. H2 rested on the
DER alone, which cannot decide it; H3 was reported as tested when the arm its
decision rule names did not exist. Both were found by auditing this manuscript
against its own protocol rather than by any reviewer, and §10 describes the
apparatus that now makes that audit routine. **Both have since been run** — the
interaction term in §5.3, the matched fine-tune contrast in §6.6 — and in each
case the executed analysis returned something different from what the paper had
been claiming: H2 unanswered for the probe rather than negative, and H3 null
rather than suggestive.



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
"EV-A71" result, including the title, is a CVA16 result extrapolated across a
handful of non-catalytic substitutions — 7 or 8 depending on strain, which our
own re-derivation puts at odds with the count of five the source paper reports
for its constructs (§3.1).

**3. Pretraining decontamination is answered for one pretraining source and
unanswerable for the other.** `plan.md` §7.1 makes re-pretraining on a
decontaminated corpus the load-bearing ablation, and H4 — that part of any
apparent transfer gain is leakage — is what it was to decide.

For **ChemBERTa** it cannot be done: the 77M corpus is not redistributed, so it
can be neither diffed nor rebuilt. §6.1 supplies a PubChem-membership **upper
bound** instead — up to 53% of test compounds could have been seen — so **H4
remains untested for those arms.** The bound's direction is the saving grace:
contamination can only flatter a transfer arm, so it cannot explain a transfer
arm losing, and the negative result survives. The asymmetry does not run the
other way — had transfer won, this design could not have told you whether the
win was real.

For the **in-domain arms** the corpus is ours, so overlap is measured rather
than bounded (0 exact, 0 near-duplicate, 61 scaffold-level of 2,974
measurements) and a decontaminated variant is pretrained and evaluated (§6.4).
That is the ablation the protocol asks for, on the one pretraining source where
it is available.

**4. The search budget is unequal, and unequal in the direction that
disfavours transfer.** `plan.md` §5 specifies an identical fixed budget for
every arm. §6.2 delivers 32 trials to B1 but only 6 to the fine-tune at n = 50
and 4 at full data, because each fine-tune fit costs ~185 s against the
baseline's ~2.7 s. B2 and the frozen probe T1 were not re-tuned at all — and T1
is the arm carrying the one significant negative result (§5.4). Its ridge
penalty is chosen by internal cross-validation, so it is not untuned, but that
is weaker than the search its comparator received. This is why §5.8 claims **no
demonstrated benefit** rather than a demonstrated deficit.

**5. Multiplicity is the binding constraint on every positive claim.** The arms
added after pre-registration required 15 full-data contrasts, and the endpoint
set adds more. Under Holm within those families, **every surviving result is a
negative one** — the frozen probe below the baseline, the chained probe below
it, the random ablation below the full corpus. The positive claims — in-domain
pretraining beating the untrained control, H3's chained contrast, the
decontamination reversal — clear Benjamini–Hochberg at best and Holm not at
all. We report them as suggestive. A reader who insists on Holm throughout
should read this paper as: generic pretraining fails on the primary endpoint,
one transfer arm beats the baseline on enrichment, and nothing else is
established.

**6. The study is powered only for large effects.** Ten seeds for the main
sweep, and five for the tuned full-data comparison. Several of the comparisons
that matter most — the fine-tune against B1 untuned (p = 0.074) and tuned
(p = 0.16 at n = 50, p = 0.63 at full data) — are inconclusive rather than null,
and are reported that way throughout (§4.4). A larger seed budget could move any
of them in either direction.

**7. Most of the pre-registered arm list was not run.** `plan.md` §4 freezes
four baselines and six transfer arms; this paper reports three baselines
(B0–B2) and two transfer arms (T1, T2). B3 (D-MPNN from scratch), T3
(alternative encoder) and T6 (ligand + ESM-2 target embedding) were not run.
T4 and T5 **were** run (§6.4), so H3 is tested — but on a corpus that Amendment
1 made harder than the protocol intended: the in-domain set was specified for
an EV-A71 3C target and is being transferred to 2A, and it is chemically
near-disjoint from the evaluation set (median nearest-neighbour Tanimoto 0.247).
H3 is *suggestive* for the chained arm T5 — Benjamini–Hochberg-significant but
not Holm-significant across the 15 full-data contrasts — and inconclusive for
T4, which is the contrast `plan.md` names. A stronger in-domain corpus —
picornaviral rather than 95% coronaviral, and overlapping the fragment
chemistry — could still change the answer.

**8. The dataset is a fragment screen, and its regime is narrow.** pK_D spans
3.44–7.94 with SD 0.86, mostly compressed between 4 and 6, and the compounds
are small fragments rather than an optimised lead series. Two consequences.
Predicting near the mean scores deceptively well on RMSE, which is why
Spearman ρ is carried alongside throughout (§3.2). And the chemistry is not
where transfer is usually deployed: whether these findings hold on a
lead-optimisation series with a wide potency range is an open question this
dataset cannot answer.

**9. Split coverage is incomplete.** The temporal split of `plan.md` §3.3 was
dropped — the OpenBind release carries no per-compound year — so the
deployment-realism endpoint is absent. The fine-tune T2 was run on the scaffold
split only, at ~185 s per fit, so the Butina degradation result of §5.5(ii)
rests on the frozen probe alone; whether full fine-tuning degrades the same way
under a stricter split is untested.

**10. Four of the five §7.2 ablations are missing, for three different
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

**11. H2 rests on a test specified after the results existed, and it is
underpowered.** `plan.md` §6 names an interaction term and says nothing about
its form; the model, the log₂ scale, the per-seed-slope estimator and the
bootstrap were all chosen in 2026-09-11 with the learning curves already in
front of us (§5.3, and the decision log). We report the choice rather than
present it as pre-specified. What would settle H2 is not a better model but
more seeds: the frozen probe's slope CI is [−0.017, +0.019], and a real
crossover effect of the size transfer is supposed to deliver would sit inside
that interval. The one significant slope, the fine-tune's, is confounded by
the fixed training schedule (§6.2) and cannot be cleaned up without re-running
that arm tuned at every training size — the most expensive experiment this
study has left.

**12. No experiment here isolates the effect of pretraining, and three
comparisons that look as though they might, do not.** The study has one
randomly-initialised ChemBERTa, `T0r`, and it is a **frozen** random-feature
probe — the encoder is never trained. The control that would isolate
pretraining is a *fully trainable* randomly-initialised ChemBERTa matched to
`T2v` in architecture, readout, optimiser and selection rule, differing only in
whether the weights began pretrained. **That arm does not exist in this study,
and we did not add it in this pass.** The consequence is a scoping rule applied
throughout §6.6, §6.7 and §9:

- **`T2v` vs `B1`, `B3` and `B2`** establishes performance against the
  evaluated baselines. It does not show that pretraining confers no benefit —
  a randomly-initialised trainable transformer might do worse still, and
  nothing here measures that.
- **`B3` vs `T2v`** (§6.7) is null, but `B3` differs in architecture *and*
  molecular representation as well as pretraining. A null between them has at
  least three candidate explanations and this design separates none.
- **`T5ft` vs `T2v`** (§6.6) tests the effect of *adding an in-domain
  pretraining stage*. It does not isolate the domain composition of that
  corpus, because there is no control receiving a comparable amount of
  **additional generic** pretraining. Extra training on any corpus is
  confounded with extra training on *this* corpus.

Wherever this paper reports a transfer arm failing to beat a baseline, the
supportable claim is about that arm against those baselines. "Pretraining does
not help" is a stronger statement, and this design cannot make it.

**13. The cliff strata are small.** §6.3 rests on a median of 9 cliff compounds
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
fingerprints with gradient boosting at any training-set size tested.** It is
*not* the claim that transfer gave no data-efficiency advantage: §5.3 now runs
the pre-registered interaction test and finds the frozen probe's deficit
neither shrinks nor grows detectably with n (slope +0.007, CI [−0.017, +0.019],
p = 0.77). H2 is unanswered for that arm. The frozen probe was significantly
worse
(Holm p = 0.012); the fine-tune, once given a search budget, was
indistinguishable rather than worse. Neither reached the baseline's full-data
RMSE on the median curve — but that is a statement about threshold-crossing at
the full-data end, and treating it as the verdict on H2 was an error this draft
corrects. An arm can be worse at full data and lose no ground at n = 50, and
nothing in this design excludes it. Per seed the DER is censored on most seeds
rather than zero on all, and its interval spans 1 wherever it is defined
(§5.3).

**That claim is scoped to the primary endpoint, and one secondary endpoint
refuses it.** On precision@10% the fine-tune beats the baseline under Holm
correction within that endpoint's family (§5.7) — the same arm that is worst on
RMSE. It is a coarse metric on a 10-compound decile, it is specific to the
scaffold split, and `T2`'s replication elsewhere is untested; but "pretraining
did not help" is a statement about predicting the value, not about ranking the
top of the list.

What the paper does **not** support is the sentence it would be easiest to
extract from it. It is not evidence that molecular pretraining does not work —
and §6.4 makes that sharper rather than softer. **Generic** pretraining is what
fails here, and it fails in a specific and measurable way. Two margins against
the same untrained control, measured the same way: ChemBERTa's frozen
representation is ahead of it by a median of 0.011 RMSE across five draws of
that control, and by 0.004 against the single draw the paired test was run on,
where it wins 5 of 10 seeds (raw p = 0.49) and cannot be separated from zero.
In-domain pretraining on 2,743 related-protease compounds is ahead of the same
control by 0.030 — roughly three times the larger of those two margins — and
beats the generic probe in 9 of 10 seeds. **That second contrast is suggestive
and not established**: corrected across the 15 full-data contrasts it clears
Benjamini–Hochberg and not Holm, and the pre-registered `T4`-vs-`T1` form is
inconclusive (§6.4). It still does not overtake count fingerprints. A null
result is not proof of no effect, and we do not read it as one — what we claim
is the ratio between two margins, and that neither is large.

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
   here (p = 0.56), neither transfer arm was shown to beat either of them on
   RMSE at any training-set size, and they fit in ~2.7 s against the
   fine-tune's ~185 s. On this evidence a pretrained encoder is not the first
   thing to reach for — with one exception: if what you need is an enriched top
   decile rather than an accurate value, §5.7 points the other way.
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

### 8.5 What is actually new here, and what is not

Written after checking the claim against the literature rather than before.
Three of the four things this paper might be read as contributing are already
established, two of them at far larger scale, and saying so is the only honest
way to state the fourth.

| Prior work | What it established | What this study adds | Scope of ours |
|---|---|---|---|
| Praski, Adamczyk & Czech, arXiv:2508.06199 (**preprint**) | 25 pretrained embedding models × 25 datasets: nearly all show negligible or no improvement over an ECFP baseline | **Nothing on the direction of the result.** Ours is the same finding at 1/25th the scale | One target, one assay, 494 compounds |
| Li & Fourches, *J. Cheminform.* 12, 27 (2020) (**journal**) | Self-supervised pretraining matches or beats RF-on-Morgan across 642–41,127 compounds; transfer always beats from-scratch *within* one architecture | A regime where it does not: single-target, single-assay, fragment chemistry. Our `T0r` control separates "beats from-scratch" from "beats fingerprints" — the two claims their result and ours are about | Does not contradict them; a different comparison |
| §2's domain-adaptation literature (Sultan et al. 2025, **preprint**) | Transformer gains concentrate in domain-adapted rather than generic pretraining, at corpus scales a single project can assemble (≤ 4K molecules) | A direct test of that prediction on this dataset, with architecture and adaptation matched: **no detectable difference** (§6.6). Our corpus is 2,743 compounds, the scale they identify | One corpus, 95% coronaviral, n = 347, one split. A null, not a refutation |
| Guo, Hernandez-Hernandez & Ballester, *J. Cheminform.* 17, 94 (2025) (**journal**) | random < scaffold < Butina < UMAP difficulty over 2,100 models on 60 NCI-60 datasets; scaffold splits still leak | **Nothing on the ordering.** One further dataset consistent with theirs, not a replication | 98-compound test folds |
| van Tilborg, Alenicheva & Grisoni, *JCIM* 62, 5938 (2022) (**journal**) | Fingerprint models frequently beat deep models on activity cliffs (MoleculeACE) | A three-way stratification (cliff / smooth / distant) that locates the transfer deficit on **distant** compounds and finds transfer *not* behind on cliffs | Median 9 cliff compounds per fold |

**So what is new.** Not that fingerprints are hard to beat; not that scaffold
splits leak; not that pretraining underperforms on a small set. Those are the
field's current position, and this study is one more data point consistent with
it. What we have not found elsewhere, and offer as the contribution:

1. **The deficit is localised rather than global.** Two independent cuts —
   across splits (§5.5) and within one split (§6.3) — agree that the pretrained
   representation loses on compounds with no near training neighbour and not on
   activity cliffs. That is the opposite of where the cliff literature would
   place a deep model's weakness, and it is a per-stratum result rather than an
   aggregate score.
2. **A decontamination ablation whose control reverses it.** Removing the 61
   overlapping records makes the in-domain arm worse, which reads as leakage
   having helped; removing 61 *random* records costs more (§6.5). The protocol's
   own specification — contaminated versus decontaminated, no size-matched
   control — would have reported a significant effect with the causal arrow
   backwards. We have not seen that control reported in this literature, and it
   is cheap.
3. **A matched in-domain-versus-generic contrast, and a worked example of a
   reporting hazard.** §6.6: the in-domain arm's median RMSE is 0.027 *better*
   than the generic arm's, while the paired median is 0.016 *worse* and it
   loses on 7 of 10 seeds. That median(X − Y) ≠ median(X) − median(Y) is
   elementary and not our finding; what we contribute is a case where the two
   disagree in **sign** on a real benchmark, survive every leave-one-out
   perturbation, and would have produced an in-domain gain in the per-arm
   median table most benchmark papers report. The disagreement is visible only
   because the arms share splits, subsamples and test rows by construction, and
   §5.4 records a second instance between two baselines.
4. **An endpoint disagreement inside one study.** The arm worst on RMSE is best
   on precision@10% (§5.7). Reported because it was pre-registered, discounted
   because it is exploratory and family-sensitive, and relevant because a
   screening campaign consumes the ranking rather than the value.
5. **The verification apparatus, and what it caught.** 414 machine-checked
   prose claims, tables generated from artefacts, fault injection against every
   checker, and a reconstruction check run against an immutable baseline in an
   isolated tree. This is method, not science, and we present it as such — but
   four audits of our own manuscript found defects that changed reported
   conclusions, including two hypotheses answered on analyses that had never
   been run. The apparatus is the contribution most likely to transfer to
   another group's benchmark.

**What remains limited to this dataset.** Every numeric result in §5 and §6.
The affinities are CVA16 2A^pro measured by one assay on 494 fragments; the
pretrained encoder is one model family; the in-domain corpus is 95%
coronaviral. Nothing here licenses a general claim about molecular pretraining,
and §7 lists eleven reasons why.

### 8.4 What we would need to change our minds

Both pre-registered experiments have now been run (§6.4, §6.5) and neither is
settled. What would settle them:

- **H3, in-domain pretraining (arms T4 and T5).** Run, and suggestive rather
  than established: the chained arm beats the generic probe in 9 of 10 seeds,
  which clears Benjamini–Hochberg and not Holm across the 15 full-data
  contrasts, and the pre-registered `T4`-vs-`T1` form is inconclusive. Two
  things would decide it. **More seeds** — at 10 the study is powered only for
  large effects and the effect here is 0.025 RMSE. And **a corpus that is
  in-domain chemically as well as by protein family**: ours is 95% coronaviral,
  its median nearest-neighbour Tanimoto to the evaluation set is 0.247, and it
  was specified for the 3C target Amendment 1 removed. A picornaviral corpus
  overlapping the fragment chemistry would test the mechanism §2 identifies as
  the actual source of transfer gains, and the scale required is modest:
  [Sultan et al. (2025)](https://arxiv.org/abs/2503.03360) report domain
  adaptation working on sets of **≤ 4K molecules**, and report pretraining
  corpus size saturating at **400K–800K** — so the relevant scale for this
  intervention is one a single-target project can actually assemble, and is the
  same order as the corpus §6.4 builds.
- **H4, decontamination.** Still not answerable for ChemBERTa, whose corpus is
  not distributed; §6.1 could only bound test-set overlap at 53%. Changing the
  design does make it answerable, and §6.5 does so for the in-domain arms —
  negatively, and only because a size-matched control was run alongside. What
  remains open is the harder version: a corpus that genuinely overlaps the
  target chemistry, where decontamination would have something substantial to
  remove. Ours had zero exact and zero near-duplicate overlap to begin with.

Both would also address the asymmetry that currently limits this study's
positive claims: because contamination can only flatter a transfer arm, this
design can refute transfer but could never have credited it.

## 9. Conclusion

On a small, high-fidelity, single-target protease dataset, a generically
pretrained chemical language model did not beat count fingerprints with
gradient boosting — not at full data and not at 50 compounds. On data
efficiency the honest verdict is weaker than the one we previously printed
here: the pre-registered interaction test leaves H2 **unanswered** for the
frozen probe, and the DER, computed per seed with its censoring made explicit,
is nowhere distinguishable from 1 (§5.3). The frozen probe was significantly worse; the
fine-tune was worse at every size once given a documented validation-based
schedule (§6.6), at ~70× the compute per fit. For projects in this regime the classical
baseline remains the right default, and the burden of proof sits with the
pretrained model.

**The deep arms land together, behind fingerprints.** A from-scratch D-MPNN —
the baseline the protocol froze and this study had not run — is also worse than
the fingerprint baseline at every training-set size (Holm ≤ 0.020), and is not
separated from the corrected fine-tune (§6.7). So the arm transfer fails to
beat is not merely a fingerprint model; a maintained graph network given the
same data does not beat it either.

**And in-domain pretraining does not rescue it, in the form the protocol
specified.** §6.4's probe-versus-probe contrast was suggestive; run as
`plan.md` names it — in-domain against generic pretraining with architecture,
readout and adaptation matched — the difference is not detectable in either
direction (paired median +0.016 RMSE, 95% CI −0.033 to +0.057, Holm 1.000;
§6.6). On the enrichment endpoint the chained in-domain arm *does* beat the
baseline at 9 of 10 seeds, and carries the same caveats as the one other
positive in this paper: coarse metric, family-sensitive, exploratory (§6.8).

**What none of this establishes is that pretraining confers no benefit.** This
study has no fully trainable randomly-initialised encoder matched to its
fine-tuned one — `T0r` is a frozen random-feature probe — so every result here
is a statement about particular arms against particular baselines (§7,
limitation 12). Where the generic arms' deficit concentrated was not on
activity cliffs but on compounds unlike anything in the training fold — a
distinction a single aggregate score hides.

Three methodological points generalise further than the headline. Transfer arms
are far more sensitive to their training schedule than the baselines are, so
equalising hyperparameters across arms — the intuitive fairness move —
systematically disadvantages transfer, and cost us a claim we had to retract.
Raw RMSE is not comparable across splitting strategies, because stricter splits
change the variance of the target. And zero scaffold overlap is not chemical
novelty: our scaffold split was no harder than a random one.

Finally, the scope, which is narrow. One encoder family, one assay, one protein
— itself a surrogate differing from the one in the title at a handful of
non-catalytic residues (§3.1). H1 is answered negatively. **H2 is not
answered**: the DER on which we had rested it cannot decide it, and the
interaction term the protocol also specified is inconclusive for the frozen
probe and confounded by the training schedule for the fine-tune. H3 is
*suggestive* in its chained form — clearing Benjamini–Hochberg but not Holm —
and inconclusive in the form this study pre-registered; H4 remains
unanswerable for the ChemBERTa arms, whose corpus is not distributed. The in-domain corpus that supports H3 is 95% coronaviral and
chemically near-disjoint from the evaluation set, so it tests a weaker version
of "in-domain" than the phrase suggests. These results constrain what to expect
in this regime. They are not a verdict on molecular pretraining.

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
non-zero on any mismatch. It currently checks **514 claims** across sections 3.1
through 6.5 and the summary sections §5.8, §8.1 and §8.3 — the last three added
after the 2026-09-06 audit found that every statement it caught drifting lived
in a section with no claims at all. That count is itself one of the claims: the
script parses this
sentence and fails if the stated total disagrees with the number of checks it
actually ran, so the one hand-typed number in a section arguing that no number
is hand-typed cannot go stale either. It has already caught two errors: a count
of compounds with replicate measurements taken from the raw table (137) rather
than the curated one (133), and this sentence itself, left reading 85 after the
ablations of §6 added checks.

**44** of those checks belong to §6.3, and six of them are **cross-checks
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

**Summaries are checked against the artefacts too, because that is where these
errors came from.** Every one of the eleven above lived in a *summary* — the
abstract, a README, a limitations item, the conclusion, the protocol checklist
— not in a computed number. `verify_manuscript.py` re-derives values and so
cannot see a sentence that merely restates a conclusion; the eleven were found
by reading. `scripts/verify_consistency.py` now closes the part of that gap a
machine can close, across seven documents: a count stated in prose (evaluated
runs, arms, machine-checked claims, tables, figures) must equal the value
derived from the artefacts **everywhere it is stated**, and a sentinel fires if
an artefact exists while a summary still claims the work it represents was not
done. The append-only histories — the decision log and the literature review —
are exempt by design, since recording what was true at the time is their job.

It earned its place immediately. On its first run it found **two further stale
claims the manual audit had missed**, both because a patch script had composed
the edit in memory and written only a different file: §6.1 still said
decontamination "remains the one part of §7.1 not performed", and limitation 2
still described a "five-residue extrapolation". Both had been reported as
fixed. `tests/test_consistency.py` pins the catchable cases, including a
negative test that a retrospective mention of a superseded count is *not*
flagged.

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

**The pipeline is reconstructed from its inputs and diffed, in an isolated
tree against an immutable baseline.** `make verify-repro` creates a throwaway
`git worktree` at a committed ref, links in the supplied inputs, runs every
offline stage there and diffs the result. The working tree is never written to.
The previous in-place design could — and once did — let a failed run's output
become the baseline that the next run compared against and passed.

Three questions are answered separately, because an earlier version ran them
together and so reported 1,104 artefacts "regenerated" when the number was 424:

- **Supplied inputs — 21 files, verified by checksum, never rebuilt.** The
  `data/raw/` payloads and the six pretrained encoders in `models/` are not in
  git (186 MB), and the pipeline consumes rather than produces them.
  `docs/input-checksums.json` records a SHA-256 for each; the check confirms
  the bytes here are the bytes the results were built from, and can do no more
  than that. Obtaining them is a precondition of reproducing the study.
- **Reconstructed — the artefacts a stage rewrites.** Watching **2,036**
  artefacts in total, a full-tier run **reconstructs 1,584** of them: the
  curated datasets, all 30 split files, the metric files and predictions of
  every arm except the ChemBERTa fine-tune, all 32 tables and 9 figures.
- **Compared only — 452 files no executed stage rewrote.** Inputs to the check,
  not outputs of it. Counted separately so the headline cannot overstate.

A further **329 artefacts are regenerated with no committed baseline to diff
against**: the 9 figures, gitignored as regenerable, and 320 prediction files,
because predictions were only ever committed for the scaffold split. The 120
Amendment 4 artefacts — 60 metric files and 60 prediction bundles — are
**compared, not regenerated**: re-running those fine-tunes costs ~7 h, so the
training is declared unexecuted while the five tables derived from them are
reconstructed by the `analyse_amended` stage. The analysis is verified; the
training behind it is verified only as committed bytes. Until
2026-09-12 this section claimed all nine figures reproduce byte-identically;
that was checked against uncommitted local copies by an in-place run, and
against a committed baseline there is nothing to check. It is now reported as
unverified rather than as verified.

**Adding this accounting is what revealed that the stage which re-fits models
was fitting nothing.** `run_arms.py` skips any cell whose metric file already
exists, so against a populated `results/metrics/` the stage exited in 8 seconds
having fitted zero models and reported success — the vacuous pass this
repository has now hit five times, this time inside the checker written to
prevent it. Stages declare the artefacts they own, those are cleared first, and
a stage that fails to rewrite what it cleared fails as `DISAPPEARED`.

**How the 1,584 reconstructions compare** falls into three categories that are
not interchangeable. Every run reports the split; the counts themselves are
**not stable between runs** and are deliberately not quoted as a fixed number
here, because which files land in the second and third categories depends on
thread scheduling. Two consecutive passing runs gave 869 and 1,006
byte-identical. What is stable, and is the claim:

| Category | Basis | Bound |
|---|---|---|
| byte-identical after canonicalisation | provenance timestamps stripped, nothing else | — |
| bytes differ, every number identical | wall-clock `seconds` in metric JSONs; identity fields and metric names checked exactly | drift exactly 0 |
| within tolerance, non-zero drift | float reduction order | **< 1e-15** observed against a 1e-9 limit, across every run |

**The amended arms' training path is verified on a predetermined subset.**
Placing 200 artefacts in *compared only* leaves the training itself unshown:
committed bytes are self-consistent, but that is not evidence they can be
produced again. `make verify-retrain` retrains five cells fixed in the source
rather than chosen afterwards — `T2v` at n = 50 and n = 347, `T4ft` and `T5ft`
at n = 347, `B3` at n = 347, all seed 0 — in a throwaway worktree with the
supplied inputs symlinked read-only, and diffs them against the worktree's own
pristine checkout. The committed outputs are the reference and are never
written to.

**All five reproduce bit-identically**: metric drift 0.00e+00, prediction drift
0.00e+00, with test-row identities, labels, selected learning rate and
train/validation sizes compared **exactly** rather than within tolerance —
numbers agreeing while the selection landed elsewhere would be coincidence, not
a rerun. The torch fine-tuning path on this hardware is fully deterministic
under the study's seeding, so the 1e-6 tolerance declared for it was never
approached. Results in `docs/retrain-verification.json`; the remaining 195
cells stay *compared only*, with commands and costs in
[`../docs/reproduction-coverage.md`](../docs/reproduction-coverage.md).

**Tolerances are declared, justified and tested from both sides.** 1e-9
relative for metrics and derived tables — four orders above the worst observed
drift (~1e-15 from `n_jobs=-1` reduction order, amplified to ~1e-13 by
`table2`'s learning-curve interpolation) and six below the third decimal the
paper reports. 1e-6 for torch fine-tune predictions. **Exact** for structural
invariants: row identities, split membership, array shapes, missingness, and
the identity fields of every metric file — a result attached to a different
`n_train` is a different experiment, not floating-point drift.
`tests/test_reproducibility.py` plants a perturbation just over each tolerance
and asserts rejection, and one just under and asserts it passes *and is
reported*. A tolerance nothing can violate is not a tolerance.

**What is not covered, with reasons rather than silence.** The ChemBERTa
fine-tune's 40 cells (~2.1 h at ~185 s per fit), the six in-domain encoders
(796–934 s each), the 41 tuned cells (101 min), and `table5`, whose
`measure_contamination.py` needs PubChem. The encoders are covered as supplied
inputs by checksum, which verifies their identity and not their derivation.
The stage table below was the original hand check:

| Stage | Result |
|---|---|
| `prepare_openbind.py` → `eva71_2a.csv` | byte-identical |
| `prepare_indomain.py` → `indomain_3c.csv` | byte-identical |
| `build_splits.py` → 30 split files | byte-identical (all 30) |
| `run_arms.py` B1 / B2 / T1 re-runs | metrics identical to < 1e-12 |
| `run_arms.py` T2 (torch fine-tune) | metrics identical to < 1e-9 |
| `run_arms.py` T0r / T4 / T5 re-runs (120 cells) | metrics identical to < 1e-9 |
| all 32 tables in `results/tables/` | data rows byte-identical |
| all 9 figures in `results/figures/` | regenerated; **no committed baseline** (gitignored as regenerable), so not verified |

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

## Declarations

### Availability of data and materials

All data and code required to reach the conclusions of this paper are openly
available without registration, login, or licence terms other than those below.

| What | Where | Licence |
|---|---|---|
| Source code, protocol, decision log, manuscript | this repository | MIT (OSI-approved) |
| Curated evaluation set (`data/processed/eva71_2a.csv`, 494 compounds) | this repository | CC0, inherited from the source release |
| In-domain pretraining corpus (`data/processed/indomain_3c.csv`, 2,974 measurements) | this repository | CC BY-SA 3.0, inherited from ChEMBL |
| Materialised split files (30 JSON, InChIKey → fold) | this repository | CC0 |
| All 1,040 per-run metric files and 690 prediction bundles | this repository | CC0 |
| All 23 result tables | this repository | CC0 |
| Primary structure–affinity data | OpenBind Zenodo record 20026661 | CC0 |
| In-domain source records | ChEMBL, 8 targets, manifests in `data/raw/` | CC BY-SA 3.0 |
| Pretrained encoder | `DeepChem/ChemBERTa-77M-MTR`, HuggingFace | as published |

**Two categories are not redistributed here, and both are stated rather than
implied.** The 186 MB of raw payloads and the six in-domain encoder checkpoints
in `models/` are not in version control; `docs/input-checksums.json` gives a
SHA-256 for each of the 21 files so that a reproduction can confirm it has the
same bytes, and `docs/reproduction-coverage.md` states which pipeline stages
that does and does not cover. The ChemBERTa-2 pretraining corpus (77M molecules)
is not distributed by its authors, which is why H4 is untestable for those arms
(§6.1, §7 limitation 3).

`docs/data-licences.md` records the per-source terms. Derived files inherit the
most restrictive licence among their inputs.

### Competing interests

The authors declare no competing interests.

### Funding

No external funding was received for this work.

### Authors' contributions

**Requires the corresponding author's input before submission — deliberately
left blank rather than drafted.** Journal of Cheminformatics requires a
statement of what each author contributed; that is a factual matter about
people, and inventing it would be a fabrication of exactly the kind the rest of
this manuscript's verification apparatus exists to prevent. The same applies to
the author list, affiliations, ORCIDs, corresponding-author designation, and to
the Funding and Competing-interests declarations above if either is inaccurate
as stated.

### Ethics approval and consent to participate

Not applicable. This study uses only published small-molecule and protein
sequence data; no human or animal subjects were involved.

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

10. Guo, Q., Hernandez-Hernandez, S., Ballester, P. J. (2025). *UMAP-based
    clustering split for rigorous evaluation of AI models for virtual screening
    on cancer cell lines.* Journal of Cheminformatics 17, 94.
    DOI 10.1186/s13321-025-01039-8 —
    **[full text of abstract and landing page]**. Peer-reviewed article; it
    supersedes the arXiv:2406.00873 / ICANN 2024 preprint form we cited until
    2026-09-12. Source of the random < scaffold < Butina < UMAP difficulty
    ordering across 2,100 models (700 per splitting algorithm) on 60 NCI-60
    datasets, and the reason Butina clustering is reported here as the stricter
    check. Our §5.5 result is consistent with their argument on one further
    dataset; a single target is not a replication of their 60-dataset study and
    is not claimed as one.

10b. Praski, M., Adamczyk, J., Czech, W. (2025, rev. 2026). *Benchmarking
    pretrained molecular embedding models for molecular representation
    learning.* arXiv:2508.06199 — **[abstract]**. **Preprint, not peer
    reviewed.** 25 pretrained embedding models over 25 datasets; reports that
    nearly all show negligible or no improvement over an ECFP baseline. Cited
    in §8.5 for where our headline sits in the literature, not for any number
    we rely on.

10c. Li, X., Fourches, D. (2020). *Inductive transfer learning for molecular
    activity prediction: next-gen QSAR models with MolPMoFiT.* Journal of
    Cheminformatics 12, 27. DOI 10.1186/s13321-020-00430-x — **[full text via
    PMC]**. Self-supervised pretraining on ~1M ChEMBL molecules; reports
    transfer matching or beating baselines including random forest on Morgan
    fingerprints, on datasets from 642 to 41,127 compounds. The positive result
    §8.5 reconciles ours against.

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
