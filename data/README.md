# data/

Payloads are gitignored. What *is* tracked: this file, the per-stage READMEs,
`*.manifest.json` (source URL, query, retrieval date, row count) and `*.sha256`.

Flow: `raw/` (immutable, as-downloaded) → `interim/` (standardised units,
deduplicated) → `processed/` (model-ready, split-assigned). `external/` holds
third-party reference sets used for pretraining or negative controls.

Nothing downstream ever writes back into `raw/`.
