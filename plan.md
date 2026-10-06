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


> ## ⚠ Amendment 2 — 2026-09-11, arm labels and what H3 actually tested
>
> **The transfer-arm labels in §4 below are not the labels used in the
> executed study, and for two arms the executed procedure is not the one §4
> specifies either.** The protocol text is preserved as written; the mapping
> is stated here. Nothing in the manuscript is renamed — the manuscript's
> descriptions of what each arm does have always been correct — but any
> reading of §4, §7.2 or the H3 decision rule against the paper must go
> through this table.
>
> | §4 label | §4 procedure | Executed label | Executed procedure | Status |
> |---|---|---|---|---|
> | `T1` | ChemBERTa-class, **full fine-tune** | `T2_chemberta_full_finetune` | full fine-tune, 40 epochs, AdamW | run, scaffold split only |
> | `T2` | same encoder, **frozen linear probe** | `T1_chemberta_linear_probe` | frozen mean-pooled encoder + ridge | run, all three splits |
> | `T3` | alternative pretrained encoder | — | — | not run |
> | `T4` | in-domain multitask pretrain → **fine-tune** | — | — | **not run** |
> | `T5` | chained SSL → in-domain multitask → **fine-tune** | — | — | **not run** |
> | `T6` | ligand encoder + ESM-2 target embedding | — | — | not run |
> | — | — | `T4_indomain_probe` | in-domain pretrain → **frozen probe** | run, all three splits |
> | — | — | `T5_chained_probe` | chained pretrain → **frozen probe** | run, all three splits |
> | — | — | `T0r_untrained_encoder_probe` | untrained encoder → frozen probe | run, all three splits |
>
> `T1` and `T2` are a straight transposition: the executed `T1` is this
> document's `T2`. It happened when `scripts/run_arms.py` was written and
> nothing followed it; `config/arm_t1_chemlm_ft.yaml` and
> `config/arm_t2_chemlm_probe.yaml` still carry the protocol's assignment,
> which is one of the reasons that harness is retired (Amendment 3).
>
> **The consequence is not cosmetic, and one earlier statement of it was
> wrong.** Read in this document's own labels, the §10 checklist said "the
> fine-tune ran on all three splits and the probe only on scaffold" — the
> exact opposite of the truth. That is corrected below.
>
> ### H3 was not tested in the form this protocol specifies
>
> §1 decides H3 by "Direct arm contrast `T4` vs `T1`" — in this document's
> labels, **in-domain fine-tune versus generic fine-tune**. Neither the
> saved runs nor `results/metrics/` contain an in-domain fine-tune: the only
> fine-tuned arm in the study is the generic one. `scripts/run_arms.py`
> routes `T4`/`T5` through `indomain_embeddings` and a ridge probe; nothing
> calls `finetune_chemberta` for any in-domain encoder.
>
> **So the pre-registered H3 comparison is unperformed, not inconclusive.**
> What was run is a different contrast: in-domain **probe** versus generic
> **probe**, which holds adaptation fixed and varies only the pretraining
> corpus. That is a legitimate — arguably cleaner — operationalisation of
> H3's *intent*, and it is what `paper/manuscript.md` §6.4 reports. It is
> not a substitute for the planned test, because it is silent on whether
> in-domain pretraining pays off when the encoder is allowed to adapt, which
> is where §2's literature locates domain-adaptation gains. H3 is therefore
> recorded as **tested in a substituted form, with the planned form not
> run.**
>
> ## ⚠ Amendment 3 — 2026-09-11, the `config/` harness is retired
>
> §9 names `make bench` as deliverable 2 and the §10 checklist credits
> `make splits`. Neither command runs: both default to `TARGET=eva71_3c`, a
> target Amendment 1 removed, and `config/` describes a study that was never
> executed (LightGBM, D-MPNN, 500-compound budgets, the pre-Amendment-2 arm
> assignment). The executed pipeline is the one in `README.md`. The Makefile
> targets now point at it and `config/` is marked superseded rather than
> deleted.

> ## ⚠ Amendment 4 — 2026-09-11, the amended fine-tuning experiments
>
> **These are amended experiments, designed after the earlier results were
> observed.** They are not the pre-registered analysis, they are not
> independent confirmation of it, and no text in the manuscript may describe
> them as either. This section is written and dated *before* the runs are
> evaluated; the only results seen beforehand are the two timing cells noted
> at the end.
>
> ### Why
>
> Two conclusions rest on numbers produced by a fine-tune configuration §6.2
> already shows to be a harness artefact — the fixed 40-epoch, single-learning-
> rate schedule that never reads a validation fold. Those are H2's only
> significant interaction slope (§5.3) and §5.6's retracted calibration
> reading. And H3's pre-registered comparison form — in-domain pretraining
> versus generic pretraining, fine-tuned — has never been run at all
> (Amendment 2), although the encoders needed for it exist.
>
> ### Arms (`scripts/run_finetune.py`)
>
> | Arm | Initial encoder weights | Purpose |
> |---|---|---|
> | `T2v` | ChemBERTa-77M-MTR | corrected generic fine-tune (H2) |
> | `T4ft` | `models/indomain_T4.pt` — random init, then 3C/3CL multitask pretraining | H3, in-domain |
> | `T5ft` | `models/indomain_T5.pt` — ChemBERTa, then the same in-domain pretraining | H3, chained |
>
> All three share one implementation and differ **only** in initial encoder
> weights: same RoBERTa backbone (384 hidden), same tokenizer, same
> mean-pooled linear readout, same optimiser, same schedule, same selection
> rule. `T4ft` vs `T2v` is the H3 decision rule of §1 with architecture and
> downstream adaptation held fixed.
>
> ### Fixed before running
>
> | | |
> |---|---|
> | **Split** | scaffold only (the primary endpoint; keeps this iteration bounded) |
> | **Seeds** | the existing ten, ids 0–9 — no new seeds |
> | **Sizes** | `T2v`: 50, 100, 250, 347 (H2 needs the curve). `T4ft`, `T5ft`: 347 only (H3's contrast is defined at full data, and this is what the compute allows) |
> | **Training subsample** | reconstructed with `run_arms.py`'s exact RNG sequence, so every comparison is paired at the subsample level; asserted, not assumed |
> | **Validation** | carved from the training subsample at 15%, mirroring `B1`'s `validation_fraction=0.15`. The split file's `val` fold is **not** used, exactly as in the published sweep — using it would hand these arms 49 compounds `B1` never sees |
> | **Tuning budget** | learning rate from {1e-5, 3e-5, 1e-4}, selected per (seed, size) on internal-validation RMSE. Fixed at three points because the measured cost is 128 s per cell at n = 50 and 566 s at n = 347; a wider grid was not affordable locally, and the budget is therefore disclosed as unequal to `B1`'s 32 trials in the same direction §7 limitation 4 already records |
> | **Schedule** | ≤ 60 epochs, AdamW, batch 16, early stopping at patience 10 on internal-validation RMSE, checkpoint restored from the best validation epoch |
> | **Recorded per cell** | selected lr, selected epoch, optimizer steps, full validation history, internal train/val sizes, wall-clock, and any failure with its traceback |
> | **Endpoints** | the five of §4.3, unchanged. Primary RMSE |
> | **Comparisons** | `T2v` vs `B1` (paired Wilcoxon over seeds, at each size); `T2v` interaction slope vs log₂n for H2; `T4ft` vs `T2v` and `T5ft` vs `T2v` at n = 347 for H3, Holm-corrected within the amended family (see the erratum below) and reported beside the pre-registered families rather than merged into them |
>
> ### Erratum, 2026-09-12 — the size of the amended family
>
> Recorded **before any `T4ft` or `T5ft` result was analysed**, because a
> family resolved after seeing the results it governs is not a family.
>
> The row above said "the three-contrast amended family" while enumerating
> more than three contrasts: `T2v` vs `B1` at four sizes, plus `T4ft` vs `T2v`
> and `T5ft` vs `T2v`. "Three" was an error — it counted arms, not contrasts.
>
> The amended RMSE family is **seven contrasts**: those six, plus `T2v` vs
> `T2`, which the row did not enumerate but which the implementation runs.
> Including it is the conservative reading, and a test that was run and then
> left out of its own family is the exact failure this erratum exists to
> prevent.
>
> Two consequences follow and neither is optional:
>
> - **The family size is 7 whether or not all seven have finished.** While the
>   sweep is incomplete, adjusted values are reported as a Bonferroni bound at
>   m = 7 — a valid upper bound on the Holm adjustment that cannot be beaten by
>   whatever the missing tests turn out to be. Correcting over however many
>   contrasts happen to be complete would shrink the family as a side effect of
>   scheduling.
> - **The `T2v` interaction-slope test is not in this family.** It is a slope
>   against zero, not an arm-vs-arm contrast, and it is the only such test in
>   the amended set, so within that set there is nothing to correct for. §6.6
>   also reports it pooled with the seven published slope tests of §5.3 as a
>   sensitivity analysis, and states both. It is **not** merged into §5.3's
>   family, because that would retroactively re-correct pre-registered results
>   — the failure recorded on 2026-09-02.
> | **Stopping rule** | none. Every planned cell runs; nothing is stopped or extended on the basis of a p-value |
> | **Reporting rule** | every planned cell is reported, including failures and unfavourable results |
>
> ### What these experiments cannot settle
>
> `T2v` differs from `T2` in **both** the schedule and the readout — `T2` used
> the `<s>`-token classification head, these use mean pooling, which is what
> the in-domain encoders were pretrained with and what every frozen probe here
> uses. `T2v`-vs-`T2` is therefore a comparison of two conditions, not an
> isolation of the schedule. The readout was matched across the three new arms
> because H3 is the contrast that needed protecting.
>
> `T4ft` is **not** `plan.md`'s `T4`: that arm was specified for the EV-A71 3C
> target Amendment 1 removed, and the corpus behind these encoders is 95%
> coronaviral with median nearest-neighbour Tanimoto 0.247 to the evaluation
> set. This is the pre-registered comparison *form* on the corpus the study
> actually has.
>
> ### Disclosure
>
> Two cells — `T2v` seed 0 at n = 50 and at n = 347 — were run before this
> amendment was finalised, to measure wall-clock and fix the budget. Their
> RMSE values were visible when the budget was chosen. The budget was set by
> measured cost, not by those values, and both cells are retained in the run
> rather than discarded.

> ## ⚠ Amendment 5 — 2026-09-12, the D-MPNN baseline `B3`
>
> **Written and dated before any `B3` test-set result was computed.** `B3` is
> a **pre-registered arm** (§4) that was never run; the *implementation*
> choices below were made in 2026-09-12 with the rest of the study already in
> front of us, and the two are distinguished throughout.
>
> ### From the pre-registration, unchanged
>
> `plan.md` §4 specifies `B3` as "D-MPNN (Chemprop-style) trained from
> scratch", a from-scratch baseline, capacity matched where architecture
> permits. Its role in H1 is to be part of "the best from-scratch baseline"
> that transfer must beat. Without it, this study has compared pretrained
> transformers against fingerprint and descriptor models only.
>
> ### Decided now (post-hoc implementation choices)
>
> | | |
> |---|---|
> | **Implementation** | `chemprop` 2.3.1 from PyPI — the reference D-MPNN implementation (Yang et al. 2019), maintained, rather than a re-implementation |
> | **Environment note** | installing it upgraded `rdkit` 2026.3.5 → 2026.3.6. The full reconstruction was re-run and every committed artefact still matches, so the upgrade does not disturb the existing results |
> | **Representation** | chemprop's default `SimpleMoleculeMolGraphFeaturizer` on the same `canonical_smiles` every other arm reads |
> | **Architecture** | `BondMessagePassing` (the D-MPNN proper) + mean aggregation + `RegressionFFN`; chemprop 2.x defaults for depth and hidden size, reported with the run |
> | **Split** | scaffold only — the primary endpoint, matching Amendment 4's scope |
> | **Seeds / sizes** | the existing ten (0–9) and all four sizes (50, 100, 250, 347), using the committed split files and the same training subsamples every other arm received, reconstructed with `run_arms.py`'s RNG sequence and asserted |
> | **Validation** | 15% carved from the **training subsample**, exactly as Amendment 4's fine-tune arms do and mirroring `B1`'s `validation_fraction=0.15`. The split file's `val` fold stays unused. Both the labelled budget (`n_train`) and the number actually fitted (`n_train_fitted`) are recorded per cell |
> | **Tuning budget** | learning rate from {1e-4, 3e-4, 1e-3} on internal-validation RMSE — three trials, matching Amendment 4's grid size so the two amended arms receive equal search. This is **less** than `B1`'s 32 trials, in the direction §7 limitation 4 already records |
> | **Schedule** | ≤ 60 epochs, batch 16, early stopping patience 10 on internal-validation RMSE, checkpoint restored from the best validation epoch — identical rule to Amendment 4 |
> | **Endpoints** | the five of §4.3, unchanged. Primary RMSE |
> | **Stopping rule** | none. All 40 cells run |
> | **Reporting rule** | every planned cell reported, including failures and unfavourable results |
>
> ### Comparisons, and the complete family, fixed now
>
> Four contrasts, all paired over the ten seeds:
>
> 1. `B3` vs `B1` at n = 50 2. at n = 100 3. at n = 250 4. at n = 347
>
> plus **`B3` vs `T2v` at n = 347**, the one that bears on H1: whether a
> from-scratch deep model changes what "the best from-scratch baseline" is.
>
> **The `B3` family is these five contrasts, m = 5**, Holm-corrected within
> itself and reported separately from the pre-registered families and from
> Amendment 4's seven-contrast family. As in Amendment 4, the family size is 5
> whether or not all five have finished; while incomplete, adjusted values are
> a Bonferroni bound at m = 5.
>
> ### What `B3` does and does not establish
>
> It broadens architecture coverage: H1 currently compares pretrained
> transformers against fingerprint and descriptor baselines only. It does
> **not** isolate the effect of pretraining, because `B3` differs from every
> transfer arm in architecture *and* molecular representation simultaneously.
> The control that would isolate pretraining is a fully trainable,
> randomly-initialised ChemBERTa matched to `T2v`; §7 records that this study
> does not have one.

> ## ⚠ Amendment 6 — 2026-09-12, the enrichment robustness check
>
> **Written and dated before any of these cells was run.** The tests and their
> correction family are enumerated below, in advance.
>
> ### Why
>
> §5.7 reports the paper's only positive result: on precision@10% the original
> fine-tune `T2` beats `B1` within that endpoint's six-arm family (Holm
> p = 0.023). The manuscript has been saying this "does not replicate" on the
> random and Butina splits. **That wording is wrong and is corrected here**:
> `T2` was never run on those splits, so the result has **not been evaluated**
> there. The probe arms were, and did not show the advantage — which is
> evidence about the probes, not about `T2`.
>
> ### What is run
>
> The **original `T2` recipe, unchanged**: `run_arms.py`'s
> `finetune_chemberta` — 40 fixed epochs, lr 3e-5, batch 16, `<s>`-token
> classification head. Not `T2v`. Substituting the corrected arm here would
> answer a different question, and §5.7's claim is about the arm that produced
> it.
>
> | | |
> |---|---|
> | **Arm** | `T2`, the original recipe, byte-identical code path |
> | **Splits** | random and Butina — the two never evaluated |
> | **Size** | n = 347 (full training fold) only; §5.7's claim is at full data |
> | **Seeds** | the existing ten, 0–9 |
> | **Endpoint** | precision@10% **exactly as defined in §4.3 and implemented in `evapro.evaluation.metrics.precision_at_k_frac`** — same top-decile definition, same k = round(0.10 × n_test) = 10, same `np.argsort(-y)` tie handling. No re-definition |
> | **Comparator** | the committed `B1` results on the same split, seed and test rows |
> | **Stopping rule** | none; all 20 cells run regardless of intermediate significance |
>
> ### Additional analysis of existing results
>
> precision@10% is **already computed and stored** for every completed `T2v`,
> `T4ft` and `T5ft` cell — it is in `METRIC_NAMES`. Reporting it is an
> additional analysis of runs that already exist, not a new experiment, and is
> labelled as such. It does **not** substitute for `T2`.
>
> ### The tests, enumerated before computation
>
> | # | Test | Split |
> |---|---|---|
> | 1 | `T2` vs `B1`, precision@10% | random |
> | 2 | `T2` vs `B1`, precision@10% | Butina |
> | 3 | `T2v` vs `B1`, precision@10% | scaffold (existing runs) |
> | 4 | `T4ft` vs `B1`, precision@10% | scaffold (existing runs) |
> | 5 | `T5ft` vs `B1`, precision@10% | scaffold (existing runs) |
>
> **Family: these five, m = 5**, Holm-corrected within themselves, reported
> separately from §5.7's six-arm scaffold family and from the amended families
> of Amendments 4 and 5. Bonferroni bound at m = 5 while incomplete. §5.7's
> pooled 31-test sensitivity analysis is retained, and a pooled figure over
> those 31 plus these 5 is reported alongside.
>
> ### What this is not
>
> A **robustness check on the same 494 compounds**, re-partitioned. Every split
> draws from one dataset, one target and one assay, so no result here is an
> independent or external replication and none may be described as one.
> §5.7's original caveat stands unchanged: precision@10% was reported partly
> because it inverted, k = 10 makes it coarse, and it is one endpoint of five.



This document is pre-registration-shaped on purpose: the analysis plan below is
fixed *before* results exist, and any deviation gets an entry in
`docs/decision-log.md`.

---

## Amendment 7 — 2026-10-04, label provenance and publication corrections

The original protocol below used “high-fidelity” and a replicate-agreement
gate. The available OpenBind table has rows for crystal complexes, not identified
assay replicates. After structural filtering, 361 compounds have one row; of
133 multi-row compounds, 128 repeat one label and only 5 have distinct labels.
The ≤1-log gate rejects nothing in this release. This checks consistency of
released labels; it does not establish low experimental noise. The manuscript
and current title now describe a curated CVA16 2A dataset. These wording and
provenance corrections do not change compounds, splits, model fits or scores.

Completed Amendments 4–6 supersede the older missing-arm statements. The live
checklist below is updated; earlier amendments remain historical records.
Wilcoxon is described as a signed-rank test, and seed intervals are scoped to
repeated splits of the same dataset. The source ZIP and cached ChemBERTa
revision are pinned, and reproduction includes all six in-domain encoders.

## Amendment 8 — 2026-10-05, numerical-sensitivity analyses for the retraining failure

Written before either analysis below was computed.

**What prompted it.** On 2026-10-04 two of the predetermined retraining cells
stopped matching their saved outputs: `B3` seed 0 at n = 347 (maximum prediction
drift 0.46) and `T2v` seed 0 at n = 50 (drift 1.43e-6, which moved
precision@10% from 0.2 to 0.1). Diagnosis on 2026-10-05, recorded in
`docs/decision-log.md`:

- Neither script fixes the number of CPU threads. Re-running each cell with
  `OMP_NUM_THREADS` set to different values gives a different result per value,
  and the same result every time for a given value. Parallel floating-point
  reductions are summed in a thread-count-dependent order.
- For `T2v` this stays at the last float32 digit. For `B3` it does not: about
  400 optimiser steps with batch normalisation and validation-based early
  stopping amplify it into a different trajectory and a different stopping epoch.
- No thread count from 1 to 15 regenerates the saved bytes of either cell today,
  although both regenerated bit-identically on 2026-09-13. The Python
  environment, the system libraries and the source are unchanged since then. A
  second, unidentified machine-state factor therefore exists. It is not
  explained and is not claimed to be.
- `precision_at_10pct` breaks ties at the top-decile cutoff by `np.argsort`
  order, so two predictions that differ in the last digit, or not at all, decide
  the metric arbitrarily.

**What has already been seen.** `B3` seed 0, n = 347 under twelve thread
settings (RMSE 0.640–0.691 against the saved 0.680), and `T2v` seed 0, n = 50
under five. No other cell has been re-run and no contrast has been recomputed.

**Analysis A — thread-pinned `B3` replicate.** All 40 `B3` cells are re-fitted
with `scripts/run_dmpnn.py --threads 1 --out-root results/sensitivity/b3_threads1`,
which is the Amendment 5 recipe unchanged except that torch is limited to one
intra-op thread. Amendment 5's five contrasts are recomputed against the same
saved `B1` and `T2v` cells, with the same test and the same Holm family (m = 5).
A contrast is called *numerically robust* if its direction and its Holm verdict
at 0.05 agree between the saved and the pinned set. Both sets are reported
whatever the outcome.

**Analysis B — tie sensitivity of precision@10%.** No retraining. For every saved
prediction file, predictions within ε = 1e-5 pKD of the k-th largest prediction
are treated as tied at the cutoff, and the lowest and highest precision@10% any
tie-breaking order could give are computed. ε is about seven times the largest
drift observed and two orders below reported precision. Every reported
precision@10% contrast against `B1` — the six scaffold tests of §5.7 and
Amendment 6's five — is recomputed twice: with the arm at its lower bound and
`B1` at its upper, and the reverse. Where `B1` predictions were not saved (the
random and Butina splits) only the arm side is bounded, and that is stated. A
contrast is *tie-robust* if its Holm verdict is the same in both scenarios as in
the saved analysis. Spearman is reported with ε-tied predictions given equal
rank, as the size of the shift only.

**What does not change.** The saved outputs remain the primary results and are
not overwritten. No tolerance is altered and `make verify-retrain` keeps its
existing criterion, which the two cells continue to fail against the saved
outputs. These analyses bound how much the reported conclusions depend on the
arithmetic; they do not restore bitwise regeneration of the saved files.

### Amendment 8 validation addendum — 2026-10-05

Recorded after the two analyses above, before the additional checks below.
Their observed results are not independent confirmation. The original two
endpoint scenarios are retained, but they are not assumed to bound a
two-sided signed-rank p-value: changing differences also changes their ranks.

For each of the same eleven tests, enumerate every attainable precision hit
count between the fixed epsilon bounds at each seed. Keep the historical
SciPy Wilcoxon calculation and the two original Holm families unchanged.
Compute the minimum and maximum raw p-value over all configurations; applying
Holm to the vectors of minima and maxima gives conservative adjusted bounds
(shared B1 outcomes may make those joint extrema unattainable). Claim robustness
over all configurations only if that entire bound and the possible directions
retain the saved verdict. Report any failure without changing epsilon, tests,
families or primary metrics. B1 remains fixed on random and Butina because its
historical predictions were not saved. Equal-ranking Spearman remains a
descriptive scenario, not a worst-case bound.

Independently rerun all 40 one-thread B3 cells in the previously installed
fresh environment, against copies of the existing pinned artifacts taken
before retraining. Repeat the originally failing T2v seed-0 n=50 cell twice
in separate one-thread processes. Require exact predictions, labels, row
identities and metric/selection records (excluding elapsed seconds). Report
the environment and input/source hashes. This checks present repeatability,
not regeneration of historical results or determinism on other machines.
The historical `full_reproduction_passed` field remains false.

Interpretation correction: the thread-count experiments establish a numerical
sensitivity mechanism, but do not isolate the cause of the September-to-October
historical mismatch or attribute it to Accelerate specifically. The historical
records lack sufficient thread/backend/machine-state metadata to rule out all
environment differences. The earlier assertion of a second machine-state
factor should be read as an unresolved possibility, not an identified cause.

## Amendment 9 — 2026-10-05, replicate encoders for the H4 ablation

Written and dated before any replicate encoder was trained.

**What prompted it.** Every encoder-level contrast in §6.4–6.5 compares two
*single* pretraining runs. `models/` holds exactly one checkpoint per condition,
and the ten downstream seeds resample only the probe's split and subsample, so
no reported interval or p-value contains any pretraining variance. The audited
consequence: `T4c` vs `T4r`, the contrast §6.5 calls "H4 decided", is a
difference between one decontaminated encoder and one size-matched random-ablated
encoder (+0.0141 median RMSE, Holm 0.273). The study already demonstrates that
this class of variation is non-trivial — `table11` re-draws `T0r`'s random
encoder five times and spans 0.6438 to 0.6549, a range of 0.011, which is
comparable to the effect being attributed to the corpus intervention.

A second entanglement found in the same audit: one `--seed` controlled the
weight initialisation, the corpus validation split and the random-ablation draw
simultaneously. The three historical `T4` conditions therefore differ in corpus
content *and* in which molecules formed the pretraining validation set
(`n_val` 447 / 438 / 437) *and* in the epoch early stopping selected
(24 / 19 / 21). `pretrain_indomain.py` now takes `--init-seed`, `--split-seed`
and `--draw-seed` separately; each defaults to `--seed`, so every historical
command reproduces unchanged.

**Scope.** The `T4` family only — `T4`, `T4_clean`, `T4_rand61`. H4's decisive
contrast lives there, and the `T5` family's equivalent is inconclusive by a wide
margin (Holm 0.439). This is a variance-scale check on an existing claim, not a
new hypothesis.

**Design.** Two replicate encoders per condition, at `--init-seed 1` and
`--init-seed 2`, joining the existing seed-0 encoder for R = 3. `--split-seed 0`
and `--draw-seed 0` are held fixed, so the only factor varying within a
condition is the encoder initialisation and batch order; the corpus validation
split and the identity of the 61 removed records are the historical ones.
Every other pretraining hyperparameter is the published one. Downstream
evaluation is the published frozen probe — `StandardScaler` then `RidgeCV` over
the same alpha grid — on the committed scaffold split files, seeds 0–9,
n = 347. No published arm, metric file or table is modified.

**Estimand and decision rule, fixed before execution.** For each encoder,
the median test RMSE over the ten downstream seeds. A condition's value is the
mean of its three encoder medians. The contrast of interest is
`T4_clean` − `T4_rand61`.

The contrast is called **encoder-robust** only if its magnitude exceeds the
largest within-condition range of encoder medians across the three conditions.
Otherwise it is called **within encoder noise**, and §6.5's H4 reading plus the
corresponding contribution claim are rescoped to single-encoder evidence. Both
outcomes are reported, with all nine encoder medians and the within-condition
spreads, whatever the direction.

**What this cannot establish.** R = 3 supports no usable encoder-level p-value;
this compares the size of an effect against the size of a nuisance, and is not
a test. Holding `--draw-seed` fixed means draw-to-draw uncertainty in *which*
61 records are removed remains unestimated, and no claim is made about it. The
historical per-seed contrasts stay as published and are not recomputed under
this amendment.

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

Updated 2026-10-04. This is the live status; the dated amendments above retain
what was known when each decision was made. Arm labels below use the executed
study's names (see Amendment 2 for the original mapping).

- [x] Data: `make data` downloads/checks the versioned OpenBind ZIP, reconstructs
  `master.csv` (925 → 649 labelled complexes), and curates 494 compounds.
- [x] Label provenance audit: repeated crystal rows do not identify independent
  assay replicates. The original high-fidelity claim is withdrawn; the
  historical spread fields remain compatibility names, not precision estimates.
- [x] N < 300 decision rule: N = 494, so the deep arms remain in scope.
- [x] Compound/scaffold split integrity: 30 committed split files.
- [x] B0–B2: all three splits, four budgets, ten seeds. B3: all four budgets
  and ten seeds on scaffold splits (Amendment 5).
- [x] T1, T0r, T4/T5 probes: all three splits and four budgets; T4c/T4r/T5c/T5r
  ablations: scaffold splits. All six in-domain encoders have build commands,
  including the two `--drop-random 61` controls.
- [x] T2: scaffold learning curve plus full-data random/Butina runs.
- [x] T2v: all four scaffold budgets; T4ft/T5ft: full-data scaffold runs.
  All 60 Amendment 4 cells completed without failures.
- [x] H2 interaction: original sweep plus corrected T2v learning curve.
- [x] H3 matched fine-tune form: performed under Amendment 4, with no detectable
  RMSE difference; the earlier probe contrast remains suggestive only.
- [~] H4: in-domain decontamination and size-matched controls completed; the
  mechanistic contrast does not survive Holm. ChemBERTa's corpus is unavailable.
- [~] T3/T6, LoRA and layer-wise adaptation were not evaluated. Temporal splitting
  lacks dates; assay-fidelity sensitivity lacks identifiable measurement replicates.
- [~] Fully trainable random-init ChemBERTa: design prepared in
  `docs/random-init-plan.md`; no evaluation has been performed.
- [x] Retraining failure of 2026-10-04: cause identified as unpinned CPU
  threading; Amendment 8's two sensitivity analyses completed. Reported verdicts
  hold, except that the sign of the B3-versus-T2v null is not stable. Bitwise
  regeneration of the saved T2v and B3 files is not restored.
- [~] Submission: results and verification are available; author metadata,
  final manuscript formatting, and public archival deposit remain outstanding.

**Current interpretation.** H1 has no demonstrated improvement on primary RMSE
against B1. H2 is unresolved for the frozen probe and contrary for T2v within
its scaffold procedure. H3 detects no difference for the matched fine-tune;
this is not equivalence. H4 remains bounded for the in-domain source and
untestable for ChemBERTa. The current results compare recipes and do not isolate
the causal contribution of generic pretraining.
