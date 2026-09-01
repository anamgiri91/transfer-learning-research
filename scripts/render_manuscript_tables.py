#!/usr/bin/env python
"""Render the manuscript's results tables directly from results/tables/*.csv.

Each table in paper/manuscript.md lives between marker comments:

    <!-- TABLE:name START -->   ... generated ...   <!-- TABLE:name END -->

Everything between the markers is overwritten from the source CSV, so a
hand-typed number cannot drift from the artefact it claims to come from.
This is what makes "the tables are not hallucinated" a mechanical property
rather than a promise.

Run: python scripts/render_manuscript_tables.py [--check]
  --check exits non-zero if the manuscript is out of date (for CI).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

TABLES = Path("results/tables")
MANUSCRIPT = Path("paper/manuscript.md")

ARM_LABEL = {
    "B0_median": "B0 median",
    "B1_ecfp_histgb": "B1 ECFP4 + HistGB",
    "B2_descriptors_rf": "B2 descriptors + RF",
    "T1_chemberta_linear_probe": "T1 ChemBERTa probe",
    "T2_chemberta_full_finetune": "T2 ChemBERTa fine-tune",
}
ORDER = list(ARM_LABEL)


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, comment="#")


def _fmt(v, nd=3, dash_if_nan=True):
    if pd.isna(v):
        return "—" if dash_if_nan else ""
    return f"{v:.{nd}f}"


def _md(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def table_full_data(split="scaffold") -> str:
    df = _read(f"table1_learning_curves__{split}.csv")
    d = df[df.n_train == df.n_train.max()].set_index("arm")
    rows = []
    for arm in ORDER:
        if arm not in d.index:
            continue
        r = d.loc[arm]
        rows.append([ARM_LABEL[arm], _fmt(r["median"]),
                     f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]",
                     _fmt(r.get("spearman_median")), _fmt(r.get("r2_median"))])
    return _md(["Arm", "RMSE", "95% CI", "Spearman ρ", "R²"], rows)


def table_paired(split="scaffold") -> str:
    df = _read(f"table3_paired_tests__{split}.csv").sort_values("p_holm")
    rows = [[ARM_LABEL.get(r.arm, r.arm),
             f"{r.median_rmse_delta_vs_baseline:+.4f}",
             f"{r.p_holm:.4f}",
             "significantly worse" if r.verdict == "significant" else "inconclusive"]
            for r in df.itertuples()]
    return _md(["Arm", "median ΔRMSE vs B1", "p (Holm)", "Verdict"], rows)


def table_curve(split="scaffold") -> str:
    df = _read(f"table1_learning_curves__{split}.csv")
    sizes = sorted(df.n_train.unique())
    rows = []
    for arm in ORDER:
        d = df[df.arm == arm].set_index("n_train")
        if d.empty:
            continue
        rows.append([ARM_LABEL[arm]] + [_fmt(d.loc[n, "median"]) if n in d.index else "—"
                                        for n in sizes])
    return _md(["Arm (RMSE ↓)"] + [f"n={n}" for n in sizes], rows)


def table_splits() -> str:
    df = _read("table4_split_difficulty.csv")
    rows = []
    for arm in ORDER:
        d = df[df.arm == arm].set_index("split")
        if d.empty:
            continue
        rows.append([ARM_LABEL[arm]] + [_fmt(d.loc[s, "r2_median"]) if s in d.index else "—"
                                        for s in ("random", "scaffold", "butina")])
    return _md(["Arm (R² ↑)", "random", "scaffold", "Butina"], rows)


def table_audit() -> str:
    df = _read("table0_split_audit.csv")
    g = df.groupby("split").median(numeric_only=True)
    rows = []
    for s in ("random", "scaffold", "butina"):
        if s not in g.index:
            continue
        r = g.loc[s]
        rows.append([s, f"{r['scaffolds_in_train_and_test']:.0f}",
                     _fmt(r["nn_tanimoto_median"]),
                     f"{100 * r['frac_test_with_nn_ge_0.7']:.1f}%"])
    return _md(["Split", "scaffolds shared train/test", "median NN Tanimoto",
                "test cmpds with NN ≥ 0.7"], rows)


def table_der(split="scaffold") -> str:
    df = _read(f"table2_der__{split}.csv")
    rows = []
    for arm in ORDER:
        d = df[df.arm == arm]
        if d.empty:
            continue
        n = d.n_to_reach_target.iloc[0]
        der = d[f"DER_vs_B1_ecfp_histgb"].iloc[0]
        rows.append([ARM_LABEL[arm],
                     "never" if not pd.notna(n) or n == float("inf") else f"{n:.0f}",
                     _fmt(der, 2)])
    return _md(["Arm", "n to reach B1's full-data RMSE", "DER vs B1"], rows)


RENDERERS = {
    "full_data": table_full_data,
    "paired": table_paired,
    "curve": table_curve,
    "splits": table_splits,
    "audit": table_audit,
    "der": table_der,
}


def render(text: str) -> str:
    for name, fn in RENDERERS.items():
        pat = re.compile(
            rf"<!-- TABLE:{name} START -->.*?<!-- TABLE:{name} END -->", re.DOTALL)
        if not pat.search(text):
            continue
        block = (f"<!-- TABLE:{name} START -->\n{fn()}\n"
                 f"<!-- TABLE:{name} END -->")
        text = pat.sub(lambda _m: block, text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    original = MANUSCRIPT.read_text()
    updated = render(original)
    found = [n for n in RENDERERS if f"<!-- TABLE:{n} START -->" in original]
    print(f"rendered tables: {', '.join(found) if found else '(none found)'}")

    if args.check:
        if updated != original:
            print("OUT OF DATE: manuscript tables differ from results/tables/. "
                  "Run scripts/render_manuscript_tables.py")
            return 1
        print("manuscript tables are current.")
        return 0

    if updated != original:
        MANUSCRIPT.write_text(updated)
        print("manuscript tables updated from source CSVs.")
    else:
        print("manuscript tables already current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
