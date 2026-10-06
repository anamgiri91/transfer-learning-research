# Reproduction coverage

Historical coverage below refers to the commit named in `reproduction-coverage.json`.
Fresh-environment checks of the publication corrections are recorded separately
in `docs/publication-validation.json`; they do not replace the historical run.

What a reproduction of this study can and cannot verify, stated as three
distinct guarantees rather than one number. Generated numbers come from
`docs/reproduction-coverage.json`, which is written by the passing full-tier
run that achieved them — not re-derived here, because a count re-derived in a
second place is a count that drifts.

Run it with `make verify-repro` (fast tier, ~3 min) or
`make verify-repro-full` (~25 min).

## The three guarantees, which are not interchangeable

### 1. Supplied inputs — verified by checksum, never rebuilt

21 files, 186 MB, none in version control: the `data/raw/` payloads and the six
in-domain encoder checkpoints in `models/`. `docs/input-checksums.json` records
a SHA-256 for each.

This confirms **identity, not derivation.** It proves the bytes present are the
bytes the committed results were built from. It does not prove the encoders
were trained the way `pretrain_indomain.py` says, and re-deriving them costs
796–934 s each. The local review archive built by `scripts/package_release.py` includes these
files. A public archival deposit and DOI are still pending; a git clone alone
does not contain them.

### 2. Reconstructed — rebuilt by a stage and diffed against an immutable baseline

A full-tier run watches **2,036** artefacts and **reconstructs 1,584**: the
curated datasets, all 30 split files, the metric files and prediction bundles
of every arm except the ChemBERTa fine-tune, and all 32 tables.

Stages run inside a throwaway `git worktree` at a committed ref. The working
tree is never written to, so a failed run cannot become the next run's
baseline — which is how an earlier in-place design once laundered a real
difference into a pass.

Comparison falls into three categories, reported on every run. **The counts are
not stable between runs** (two consecutive passes gave 869 and 1,006
byte-identical) because which file lands in which category depends on thread
scheduling; the stable claim is the bound in the third row.

| Category | Basis |
|---|---|
| byte-identical after canonicalisation | provenance timestamps stripped, nothing else |
| bytes differ, every number identical | wall-clock `seconds` in metric JSONs |
| within tolerance, non-zero drift | float reduction order; **approximately 1e-15 observed**, limit 1e-9 |

### 3. Compared only, and no baseline

**452** watched files that no executed stage rewrote — inputs to the check, not
outputs of it.

**329** artefacts are regenerated with **no committed baseline to diff
against**: the 9 figures (gitignored as regenerable, and matplotlib PNGs are
not byte-stable across environments) and 320 prediction bundles (only the
scaffold split's predictions were ever committed, 710 of 1,060 cells). These
are reported as reconstructed-without-verification, which is weaker than
verified and says so.

## Tolerances

Declared and justified, then tested from both sides. Every tolerance is
**relative**, because the compared columns are not on one scale.

| Scope | Tolerance | Why this size |
|---|---|---|
| metric JSONs, derived CSV tables | 1e-9 | worst observed drift ~1e-15 (`n_jobs=-1` reduction order), amplified to ~1e-13 by `table2`'s learning-curve interpolation. Four orders above the worst drift, six below the third decimal the paper reports |
| torch fine-tune predictions | 1e-6 | threaded CPU BLAS non-determinism accumulating over ~1e3 optimiser steps |
| row identities, split membership, array shapes, missingness, metric-file identity fields (`arm`, `split`, `seed`, `n_train`, `n_test`), metric names | **exact** | these are integers and strings. A result attached to a different `n_train` is a different experiment, not drift |

`tests/test_reproducibility.py` plants a perturbation just over each tolerance
and asserts rejection, and one just under and asserts it passes *and is
reported*. It also asserts a changed `n_train`, a changed split label, a
flipped verdict string, changed row identities and a dropped metric all fail
regardless of the numeric outcome. A tolerance nothing can violate is not a
tolerance.

## Not executed, with reasons

| Stage | Cost | Status |
|---|---|---|
| `run_arms.py --arms T2`, 40 cells | ~185 s per fit, ~2.1 h | declared unexecuted; the arm's committed metrics are compared-only |
| `pretrain_indomain.py`, six encoders | 796–934 s each, ~1.4 h | encoders covered as supplied inputs by checksum (identity, not derivation) |
| `tune_arms.py`, 41 tuned cells | 101 min | declared unexecuted |
| `measure_contamination.py` → `table5` | needs PubChem | the one committed table with no offline reconstruction path |
| `run_finetune.py`, 60 Amendment 4 cells | ~7 h total | training **not** re-executed; the 60 metric files and 60 prediction bundles are committed and compared, and tables 17–22 derived from them **are** reconstructed by the `analyse_amended` stage |

## Retraining the amended arms

The amended fine-tunes (60 cells, 120 artifacts) are **compared only** in the
full reconstruction. B3 (40 cells, 80 artifacts) is re-fitted in the full tier
and compared only in the fast tier. The separate retraining check below also
executes selected fine-tuning paths.

`make verify-retrain` closes that for a **predetermined** subset, fixed in the
source rather than chosen after seeing which cells reproduce:

| Cell | Why this one |
|---|---|
| `T2v` seed 0, n = 50 | cheapest fine-tune cell |
| `T2v` seed 0, n = 347 | most expensive |
| `T4ft` seed 0, n = 347 | in-domain encoder path |
| `T5ft` seed 0, n = 347 | chained encoder path |
| `B3` seed 0, n = 347 | chemprop path |

It runs in a throwaway `git worktree` at an immutable ref with the supplied
inputs symlinked read-only, clears only those five cells, refits them, and
diffs against the worktree's own pristine checkout. **The committed outputs are
the reference and are never written to.** Compared: every metric, every
prediction, and — exactly, not within tolerance — test-row identities, labels,
the selected learning rate and the train/validation sizes, because numbers
agreeing while the selection landed elsewhere would be a coincidence rather
than a rerun. Results are written to `docs/retrain-verification.json`.

## Reproduction environment and inputs

Use CPython **3.14.7** (the tested platform is macOS arm64):

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

`requirements-repro.txt` pins the installed transitive dependency closure,
including torch, transformers, chemprop and RDKit. The `[repro]` package extra
also declares the direct reproduction dependencies. `[dev]` alone is not the
reproduction environment. These pins are tested on the platform above, not a
promise of binary identity on every operating system or processor.

Every executed ChemBERTa loader pins checkpoint and tokenizer to revision
`66b895cab8adebea0cb59a8effa66b2020f204ca` in
`DeepChem/ChemBERTa-77M-MTR`; see `evapro.models.pretrained`. The value comes
from the original study's cached snapshot, not today's moving default branch.
The unused legacy 10M model in the prototype code is not part of this study.

The review archive contains the 21 supplied inputs with their original hashes.
To prepare that archive locally:

```bash
python scripts/package_release.py
```

The bundle includes code, manuscript, committed datasets/results, generated
figures, raw exports and six encoder checkpoints. It excludes `private/`,
credentials and the local Python environment. Its manifest identifies the base
commit, uncommitted-change status and every file hash. It is **not yet a public
archival release**; upload/deposition and a persistent identifier remain pending.

To acquire inputs without the bundle:

```bash
python scripts/fetch_openbind.py   # exact Zenodo 20026661 archive + SHA-256
python scripts/prepare_openbind.py # source ZIP -> master.csv -> compound table
python scripts/fetch_indomain.py   # live ChEMBL queries; may differ from historical exports
make indomain                    # all six encoders, including both --drop-random 61
```

Live ChEMBL results are not guaranteed to recreate the historical CSV bytes.
Use the bundled historical exports for exact reproduction. Rebuilt encoder
sidecars contain new timestamps and timings; their hashes need not equal the
supplied files. Do **not** overwrite `docs/input-checksums.json` to turn an input
mismatch into a pass. That manifest describes the supplied historical inputs.

## Rebuilding results

`paper/provenance.md` lists the detailed order. `make bench` includes the original
sweep, amended fine-tunes, B3, cross-split T2 cells and the exact historical
41-cell tuning sweep. `make analysis` rebuilds the offline tables; the two
network analyses have explicit commands below.

```bash
make splits
make bench
make analysis
python scripts/measure_contamination.py  # table5; PubChem required
python scripts/verify_surrogate.py       # table9; UniProt required
make report
python scripts/render_manuscript_tables.py
make verify
```

For comparison with a committed baseline, use `make verify-repro-full`; use
`make verify-retrain` for the predetermined five-cell training subset. Those
commands require a git checkout and compare the selected commit, not uncommitted
edits. The published bundle can be unpacked and run without git for ordinary
preparation/training/analysis; the git-worktree comparison commands are separate.

## Total cost of the study

~5.3 CPU-hours on one laptop: 125 min of evaluation runs (of which 118 min is
the ChemBERTa fine-tune), ~86 min pretraining six in-domain encoders, 101 min
of hyperparameter search, and a few minutes of data preparation and analysis.
The Amendment 4 sweep adds roughly 7 h in total: T2v runs at four budgets,
while T4ft and T5ft run only at full data. B3 and cross-split T2 add further cost;
these are historical timings, not a measured duration for the repaired full recipe.

## Publication validation status

The pinned dependency closure installs successfully in a fresh environment.
The verification suite passes there, and 68 data/table artifacts match the
immutable review archive: 67 after canonicalization and one with an explanatory
header change but identical numeric values.

**Fresh T2v retraining does not pass the existing metric tolerance.** Seed 0 at
n = 50 has small prediction drift but changed precision@10% and Spearman. The
exact CLI reproduces the discrepancy seen through the direct training API.
The prediction drift itself is within the declared relative tolerance; the
ranking metrics are not. The original result files and tolerance thresholds
remain unchanged. See `publication-validation.json` and the paired prediction
CSV for evidence. The fresh B3 seed-0 full-data check also fails, with approximately 0.46
maximum absolute prediction drift despite matching row identities and selected
learning rate. This is larger than a ranking tie issue; the investigation and sensitivity
analysis below address it separately. A diagnostic rerun in the original `.venv`
produces the same discrepancy, so it is not specific to the fresh pip install.

**Established sensitivity mechanism (2026-10-05).** Neither `run_dmpnn.py` nor `run_finetune.py` fixes the
CPU thread count. Each thread count gives a different result and repeats
exactly, because parallel floating-point reductions sum in a different order.
For T2v the difference stays in the last float32 digit: two test rows predicted
6.4049935 and 6.4049931 in the saved run come out exactly equal in the rerun,
they straddle rank 10 of 98, and `np.argsort` breaks the tie. For B3, about 400
optimiser steps with batch normalisation and early stopping turn the same
last-digit difference into another stopping epoch: seed 0 at n = 347 gives RMSE
between 0.640 and 0.691 across thread counts 1–15, against the saved 0.680.

**Not explained.** None of the tested thread counts regenerates the saved
predictions today, although both cells passed the historical 2026-09-13 check.
Thread count is a demonstrated sensitivity, not a complete explanation of that
historical mismatch. Historical records lack enough thread, backend and
machine-state metadata to isolate the remaining difference. Attribution to
Accelerate specifically has not been established.

**Resolution: `plan.md` Amendment 8, fixed before either analysis ran.** Results
are in `numerical-sensitivity.json`, rebuilt by `make sensitivity`.

| Analysis | Result |
|---|---|
| All 40 B3 cells re-fitted on one thread (`make sensitivity`); Amendment 5's five contrasts recomputed unchanged | B3 remains worse than B1 at all four sizes (Holm 0.020 at each). B3 vs T2v remains undetected (p = 0.32), but its median changes sign, +0.009 to −0.009, so that contrast fails the pre-specified direction criterion |
| Size of the effect on single B3 cells | median absolute RMSE shift 0.014–0.043 by training size, largest 0.100 |
| precision@10% bounded over every tie-break within 1e-5 pKD, 810 prediction files | 101 files can change, 40 of them the constant predictor B0. All eleven reported contrasts keep their Holm verdict under exhaustive hit-count enumeration and conservative Holm bounds; the scaffold T2 result moves from 0.023 to at most 0.047 |
| Spearman with tied predictions given equal rank | largest shift 0.0008 |
| The failing T2v cell | its rerun value, 0.1, is the lower tie bound of the saved cell, [0.1, 0.2] |

The original two endpoint scenarios are retained. The dated validation addendum
in Amendment 8 additionally enumerates every attainable per-seed hit count;
extreme precision values alone need not bound a two-sided signed-rank p-value.
All eleven verdicts are certified over those configurations, with a conservative
Holm upper bound of 0.046875 for scaffold T2. B1 is held fixed on random and
Butina because historical predictions were not saved there. Spearman's 0.0008
is the largest observed shift under equal-ranking, not a worst-case bound.
These are post-failure sensitivity analyses, not independent confirmation.

Independent retraining evidence is recorded separately in
`docs/numerical-repeatability.json`. The check copies the existing pinned B3
baseline before refitting all 40 cells, and fits the failing T2v seed-0 n=50
cell twice in separate one-thread processes. It compares predictions, labels,
row identities and the full metric/selection records exactly, excluding elapsed
seconds. The environment and source/input hashes accompany the results. All 40 B3 cells match the existing pinned baseline exactly, and the two T2v fits match each other exactly (maximum prediction drift 0.0). These runs used the fresh dependency environment; every protected historical and pinned artifact retained its original hash.

```bash
make sensitivity PY=.venv/bin/python
make verify-numerical-repeatability PY=/path/to/fresh/environment/bin/python
make verify PY=.venv/bin/python
```

Both training CLIs accept `--threads 1 --out-root <separate-directory>`; a
thread-pinned run requires a separate output root. `make verify` re-derives the
sensitivity report and rejects missing, incomplete or stale evidence. The review
archive includes `plan.md`, the sensitivity artifacts and both JSON reports.

**Status.** The saved outputs are unchanged and remain the primary results. No
tolerance was changed, and the two cells still fail `make verify-retrain`
against the saved outputs: this is not a passing full reconstruction. What has
changed is that numerical sensitivity is measured and the reported verdicts
survive the specified checks. The B3-versus-T2v direction fails the declared
robustness criterion and is explicitly withdrawn as a stable finding. This
resolves the numerical issue through sensitivity analysis and limited claims;
it does not certify full historical reconstruction or cross-machine determinism.
