# Prototype pipeline (superseded, currently broken)

The original scripts for this project, written before the refactor into
`src/evapro/`. Preserved because `fetch_and_prepare_data.py` documents how the
raw data was obtained, and because the current pipeline should not silently
erase what came before it.

**These do not run against the current package.** Two breakages, both caused by
the refactor rather than by anything wrong with them:

| Script | Breakage |
|---|---|
| `run_benchmarks.py` | imports `get_morgan_fingerprint` and `RandomForestBaseline`; the refactor renamed these to `ecfp()` and `build_random_forest()` |
| `generate_reports.py` | reads `{rf_ecfp4,chemberta,schnet}__scaffold__seed42.json`; the first two moved to `results/legacy_prior_run/`, and the SchNet file was never produced |
| `generate_splits.py` | splits at **crystal-complex** level (649 rows); superseded by `scripts/build_splits.py`, which splits at compound level (494) because complex-level random splits leak 22 compounds between train and test |
| `fetch_and_prepare_data.py` | historical acquisition path; now superseded by `scripts/fetch_openbind.py` and `scripts/prepare_openbind.py` (the original path calculation no longer applies after moving into `prototype/`) |

Current equivalents: `prepare_openbind.py`, `build_splits.py`, `run_arms.py`,
`make_report.py`.

Either repair these against the new API or delete them; leaving them here
indefinitely is the one option that misleads.
