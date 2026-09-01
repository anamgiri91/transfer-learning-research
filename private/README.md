# private/ — not version controlled

Everything in this directory except this file is excluded by `.gitignore`.
Nothing here should ever be needed to reproduce a published result: if a
reviewer cannot rerun the pipeline without it, it is in the wrong place.

| Subdirectory | Holds | Notes |
|---|---|---|
| `data/` | Licence-restricted or unpublished assay data, collaborator spreadsheets, embargoed structures | Record provenance + licence in `notes/data-provenance.md` |
| `notes/` | Lab notebook, meeting notes, reviewer correspondence, dead ends | Free-form; date every entry |
| `manuscript/` | Drafts, figures with unpublished data, cover letters, rebuttals | Move to `reports/` only after clearance |
| `credentials/` | Cluster keys, service-account JSON, VPN configs | Prefer a real secret manager; this is a fallback |
| `scratch/` | Throwaway scripts and one-off exports | Assume it will be deleted without warning |

## Rules

1. **No code here.** Anything that produces a number belongs in `src/` where it
   can be tested and reviewed. Private *inputs* are fine; private *methods* make
   the paper unreproducible.
2. **Every private dataset gets a public stub.** When a file lands in
   `private/data/`, add a row to `data/raw/README.md` naming it, its source, its
   licence, and why it cannot be shared. The absence should be documented, not
   silent.
3. **Promotion is one-way and deliberate.** Moving a file out of `private/`
   publishes it. Check licence and co-author consent first.
4. **Before any push**, run `make check-private`.
