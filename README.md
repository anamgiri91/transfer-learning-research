# Benchmarking Transfer Learning Efficacy on High-Fidelity Viral Protease Datasets

### A Case Study on EV-A71

Does pretraining actually help when the target dataset is *small but clean*?

This study uses the **OpenBind EV-A71 / CVA16 2A protease** structure–affinity
release ([Zenodo, CC0](https://doi.org/10.5281/zenodo.20026661)): 494 curated
compounds across 272 scaffolds, with pK_D from a single Creoptix WAVEsystem
assay. Maximum replicate spread across the whole set is 0.49 log units — this
really is a high-fidelity dataset, which is what makes the question clean.

**Note on the target.** Affinities are measured on *Coxsackievirus A16* 2A
protease as a surrogate for EV-A71; the two differ at five residues, none near
the active site. See `paper/manuscript.md` §3.1.

**Claim under test.** Transfer learning improves data efficiency on
high-fidelity, low-volume protease datasets — and the improvement survives
scaffold-based splitting and is not an artefact of test-set leakage.

---

## Layout

```
config/      YAML experiment configs (public, versioned — one per benchmark arm)
data/        Directory structure + manifests are public; payloads are not
src/evapro/  Importable package: data, features, models, evaluation
scripts/     Thin CLI entrypoints that call src/ (no logic lives here)
notebooks/   Exploration only; anything load-bearing gets promoted to src/
tests/       pytest — split integrity and leakage checks are the important ones
results/     metrics/*.json is versioned; figures/ and tables/ are regenerated
reports/     Public write-ups, preprint-ready figures
docs/        Methods notes, data licences, decision log
private/     NOT version controlled — see private/README.md
```

The public/private line: **anything needed to reproduce a published number is
public.** `private/` holds embargoed inputs, notes, and drafts only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # fill in optional API keys
nbstripout --install        # keep notebook outputs out of history
```

## Reproducing the benchmark

```bash
python scripts/prepare_openbind.py                # master.csv -> 494 compounds
python scripts/build_splits.py --target eva71_2a  # splits + leakage assertions
python scripts/audit_splits.py                    # Table 0: leakage audit
python scripts/run_arms.py --arms B0 B1 B2 T1     # baselines + frozen-encoder probe
python scripts/run_arms.py --arms T2              # ChemBERTa fine-tune (slow)
python scripts/make_report.py                     # tables + figures
```

Full artefact map: [`paper/provenance.md`](paper/provenance.md).

Every run writes `results/metrics/<arm>__<split>__seed<N>.json`. Those files are
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
| `paper/manuscript.md` | The paper; every number points at a file |
| `paper/provenance.md` | Table/figure → script → output-file map |
| `docs/literature.md` | Annotated review, with honest reading-depth labels |
| `docs/decision-log.md` | Every deviation from the protocol, dated |
