# Reproduction coverage

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
796–934 s each. Obtaining these files is a precondition of reproducing the
study; they are not redistributed here because of size.

### 2. Reconstructed — rebuilt by a stage and diffed against an immutable baseline

A full-tier run watches **2,036** artefacts and **reconstructs 1,584**: the
curated datasets, all 30 split files, the metric files and prediction bundles
of every arm except the ChemBERTa fine-tune, and all 23 tables.

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
| within tolerance, non-zero drift | float reduction order; **< 1e-15 observed**, limit 1e-9 |

### 3. Compared only, and no baseline

**452** watched files that no executed stage rewrote — inputs to the check, not
outputs of it.

**329** artefacts are regenerated with **no committed baseline to diff
against**: the 9 figures (gitignored as regenerable, and matplotlib PNGs are
not byte-stable across environments) and 320 prediction bundles (only the
scaffold split's predictions were ever committed, 690 of 1,040 cells). These
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
| `run_finetune.py`, 60 Amendment 4 cells | ~7 h total | training **not** re-executed; the 60 metric files and 60 prediction bundles are committed and compared, and tables 17–21 derived from them **are** reconstructed by the `analyse_amended` stage |

## Retraining the amended arms

`verify_reproducibility.py` places the 120 Amendment 4 artefacts and the 80
Amendment 5 ones in **compared only** — committed and diffed, but their
training not re-executed, because all 100 cells cost ~7.3 h. That leaves the
training *path* unverified: the bytes are self-consistent, but nothing had
shown they can be produced again.

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

## Reproducing the rest

```bash
# 1. Obtain the supplied inputs (not redistributed here; 186 MB)
#    - data/raw/: OpenBind Zenodo record 20026661, plus the 8 ChEMBL exports
#      whose manifests are in data/raw/*.manifest.json
#    - models/: six in-domain encoders, rebuilt by `make indomain` (~1.4 h)
python scripts/verify_reproducibility.py --write-input-manifest   # then diff

# 2. Everything cheap and offline (~25 min)
make verify-repro-full

# 3. The amended arms, in full (~7.3 h)
make finetune            # Amendment 4: T2v, T4ft, T5ft  (~7 h)
python scripts/run_dmpnn.py --splits scaffold          # Amendment 5: B3 (~17 min)
python scripts/run_arms.py --arms T2 --splits random butina --sizes 347 --save-preds

# 4. The original ChemBERTa fine-tune and the tuning sweep (~3.8 h)
python scripts/run_arms.py --arms T2 --splits scaffold --save-preds
python scripts/tune_arms.py
```

**Dependencies and access.** Python 3.14, `pip install -e ".[dev]"`, plus
`chemprop==2.3.1` for `B3` (installing it upgrades `rdkit` to 2026.3.6; the
full reconstruction was re-run under that version and every artefact still
matches). `DeepChem/ChemBERTa-77M-MTR` is pulled from HuggingFace on first use
and needs network access once. `measure_contamination.py` needs PubChem and
`verify_surrogate.py` needs UniProt; both are marked network stages and skip
cleanly when unreachable. No credentials, licences or registrations are
required for any of it.

## Total cost of the study

~5.3 CPU-hours on one laptop: 125 min of evaluation runs (of which 118 min is
the ChemBERTa fine-tune), ~86 min pretraining six in-domain encoders, 101 min
of hyperparameter search, and a few minutes of data preparation and analysis.
The amended fine-tuning arms of Amendment 4 add ~3.7 h per arm on top.
