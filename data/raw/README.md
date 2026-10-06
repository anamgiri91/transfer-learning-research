# Raw inputs

The publication workflow uses two sources:

| Source | Acquisition | Historical identity |
|---|---|---|
| OpenBind CVA16/EV-A71 2A release | `python scripts/fetch_openbind.py` | Zenodo record 20026661, `OpenBind_EV-A71_2A.zip`; SHA-256 in `docs/input-checksums.json` |
| Eight ChEMBL 3C/3CL exports | `python scripts/fetch_indomain.py` | `indomain_*.csv`; query/target manifests here and SHA-256 in `docs/input-checksums.json` |

The OpenBind script downloads the versioned 98.4 MB ZIP and refuses a checksum
mismatch. `prepare_openbind.py` reads its metadata directly, renames three
columns and drops missing pKD labels to reconstruct `data/processed/master.csv`
(925 → 649 rows), then applies the compound-level curation.

The metadata identifies crystal complexes. It supplies no measurement IDs,
replicate counts or uncertainty estimates. Repeated structure rows are not
independent assay-replication evidence. `eva71_2a.label_audit.json` records this
limitation and the observed label reuse.

Live ChEMBL queries can change. The local review bundle built by
`scripts/package_release.py` includes the exact historical exports; public
deposition remains pending. Do not replace the checksum baseline with a newly
fetched dataset and call it an exact reproduction.

Data terms are recorded in `docs/data-licences.md`: OpenBind CC0; ChEMBL exports
CC BY-SA 3.0. Raw payloads remain excluded from git and are packaged separately.
