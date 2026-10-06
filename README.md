# Benchmarking Molecular Transfer Learning on a Curated CVA16 2A Protease Dataset

How do pretrained molecular models compare with classical baselines on a small,
single-target affinity dataset?

This study uses the **OpenBind EV-A71 / CVA16 2A protease** release
([Zenodo, CC0](https://doi.org/10.5281/zenodo.20026661)): 494 curated compounds
and 272 scaffolds. Depositors report affinities measured with the Creoptix
WAVEsystem. Multiple crystal structures often repeat the same affinity label;
the release does not identify independent assay replicates. Its within-compound
label consistency is not evidence of low experimental noise.

**Note on the target.** Affinities are measured on *Coxsackievirus A16* 2A
protease as a surrogate for EV-A71. The source paper reports a five-residue
difference for its own constructs; re-deriving it from the UniProt-annotated 2A
chains gives **7 or 8 residues depending on the CVA16 reference strain**, so we
do not rely on the count. What is independently confirmed is the claim that
matters: the catalytic triad is identical and no differing residue is
catalytic. Note that this is weaker than "none near the active site" — in the
G-10 strain the N57D substitution sits between two structural zinc ligands. See
`paper/manuscript.md` §3.1 and `results/tables/table9_surrogate_divergence.csv`.

**Scope.** Compare evaluated pretrained recipes with from-scratch baselines
on identical splits and budgets. A fully trainable random-initialization
transformer control has not been evaluated, so the existing results do not
isolate the effect of pretraining.

---

## Layout

```
config/      Retired prototype configs; executed settings live in the scripts
data/        Directory structure + manifests are public; payloads are not
src/evapro/  Importable package: data, features, models, evaluation
scripts/     Executed training, analysis and verification entrypoints
notebooks/   Exploration only; anything load-bearing gets promoted to src/
tests/       pytest — split integrity and leakage checks are the important ones
results/     metrics/, predictions/ and tables/ are versioned; figures/ are regenerated
paper/       Manuscript and provenance map
docs/        Methods notes, data licences, decision log
private/     NOT version controlled — see private/README.md
```

Release inputs and their current availability are documented in
`docs/reproduction-coverage.md`. `private/` holds embargoed inputs, notes, and drafts only.

## Setup

The reproduction environment is CPython 3.14.7 on macOS arm64. Exact package
versions are recorded in `requirements-repro.txt`; other platforms are unverified.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps --no-build-isolation -e .
```

## Reproducing the benchmark

```bash
make data          # fetch/check source ZIP, build master.csv and compound dataset
make splits
make indomain      # corpus and all SIX encoders, including --drop-random 61
make bench         # original sweep, amendments, B3, enrichment and tuning
make analysis
make report
python scripts/render_manuscript_tables.py
```

This is a multi-hour rebuild. Live ChEMBL queries can change; the historical
exports/checkpoints and their publication status are documented in
[`docs/reproduction-coverage.md`](docs/reproduction-coverage.md).
The command list and exact tuning budgets are in [`paper/provenance.md`](paper/provenance.md).

Numerical validation preserves the historical outputs and tolerances. The
historical retraining check still fails for two cells; Amendment 8 reports a
separate pinned B3 replicate and ranking sensitivity analysis. The B3-versus-T2v
direction is unstable, while the specified B3-versus-B1 and enrichment verdicts
survive. See the coverage document for the limits of this evidence.

```bash
make sensitivity
make verify-numerical-repeatability  # fresh fits in a temporary source snapshot
```

Verify the stored evidence and manuscript:

```bash
make verify        # tests + table freshness + every machine-checked claim + citations
make verify-repro  # re-run every offline stage and diff it against what is committed
```

Full artefact map: [`paper/provenance.md`](paper/provenance.md).

Original-sweep runs write `results/metrics/<arm>__<split>__seed<N>__n<size>.json`;
amended fine-tunes and B3 use `metrics_ft/` and `metrics_b3/`. Those files are
committed, so any change in a headline number shows up as a reviewable diff.

## Benchmark design in one paragraph

Each *arm* is a (representation, pretraining source, adaptation strategy)
triple, evaluated against from-scratch baselines on identical splits and seeds.
Learning curves run at {50, 100, 250, 347} training compounds so that "transfer
helps" can be stated as a data-efficiency ratio rather than a single delta. The
primary endpoint is scaffold-split test performance; random-split numbers are
reported only as an optimism reference — and Table 0 quantifies exactly how much
optimism (23–31 shared scaffolds between train and test).

See `plan.md` for the full protocol and `docs/decision-log.md` for changes to it.

## Documents

| File | What it is |
|---|---|
| `plan.md` | Pre-registered protocol + Amendment 1 (data differed from assumptions) |
| `paper/manuscript.md` | The paper; tables are generated, prose numbers machine-checked |
| `paper/provenance.md` | Table/figure → script → output-file map |
| `docs/literature.md` | Annotated review, with honest reading-depth labels |
| `docs/decision-log.md` | Every deviation from the protocol, dated |
