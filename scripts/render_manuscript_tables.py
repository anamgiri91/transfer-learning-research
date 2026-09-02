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
    "T0r_untrained_encoder_probe": "T0r untrained encoder probe",
    "T4_indomain_probe": "T4 in-domain probe",
    "T5_chained_probe": "T5 chained probe",
    "T4c_indomain_probe_decontaminated": "T4 in-domain probe (decontaminated)",
    "T5c_chained_probe_decontaminated": "T5 chained probe (decontaminated)",
}
ORDER = list(ARM_LABEL)

# Tables in §5 describe the pre-registered five-arm sweep across three splits.
# The control and in-domain arms were run on the scaffold split only and are
# reported in §6.4, so including them here would add a row of em-dashes to a
# table whose entire point is the cross-split comparison.
PREREGISTERED = ["B0_median", "B1_ecfp_histgb", "B2_descriptors_rf",
                 "T1_chemberta_linear_probe", "T2_chemberta_full_finetune"]


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
    for arm in PREREGISTERED:
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
    for arm in PREREGISTERED:
        d = df[df.arm == arm].set_index("n_train")
        if d.empty:
            continue
        rows.append([ARM_LABEL[arm]] + [_fmt(d.loc[n, "median"]) if n in d.index else "—"
                                        for n in sizes])
    return _md(["Arm (RMSE ↓)"] + [f"n={n}" for n in sizes], rows)


def table_splits() -> str:
    df = _read("table4_split_difficulty.csv")
    rows = []
    for arm in PREREGISTERED:
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
    for arm in PREREGISTERED:
        d = df[df.arm == arm]
        if d.empty:
            continue
        n = d.n_to_reach_target.iloc[0]
        der = d[f"DER_vs_B1_ecfp_histgb"].iloc[0]
        rows.append([ARM_LABEL[arm],
                     "never" if not pd.notna(n) or n == float("inf") else f"{n:.0f}",
                     _fmt(der, 2)])
    return _md(["Arm", "n to reach B1's full-data RMSE", "DER vs B1"], rows)


def table_contamination() -> str:
    df = _read("table5_contamination.csv")
    rows = [[r.scope, r.split, str(int(r.n)), str(int(r.n_in_pubchem)),
             f"{100 * r.frac_in_pubchem:.1f}%"] for r in df.itertuples()]
    return _md(["Scope", "Split", "n", "in PubChem", "fraction"], rows)


def table_tuning() -> str:
    """Tuned vs untuned, matched on (arm, seed, n). Only cells present in both."""
    import glob, json
    tuned, untuned = {}, {}
    for f in glob.glob("results/tuned_metrics/*.json"):
        r = json.load(open(f))
        tuned[(r["arm"], r["split"], r["seed"], r["n_train"])] = r["metrics"]
    for f in glob.glob("results/metrics/*.json"):
        r = json.load(open(f))
        untuned[(r["arm"], r["split"], r["seed"], r["n_train"])] = r["metrics"]

    keys = sorted(set(tuned) & set(untuned))
    if not keys:
        return "_(no matched tuned/untuned cells yet)_"
    recs = [{"arm": k[0], "n": k[3],
             "untuned": untuned[k]["rmse"], "tuned": tuned[k]["rmse"],
             "sp_untuned": untuned[k]["spearman"], "sp_tuned": tuned[k]["spearman"]}
            for k in keys]
    d = pd.DataFrame(recs)
    rows = []
    for arm in ORDER:
        g = d[d.arm == arm]
        if g.empty:
            continue
        for n in sorted(g.n.unique()):
            gg = g[g.n == n]
            rows.append([ARM_LABEL[arm], str(int(n)), str(len(gg)),
                         _fmt(gg.untuned.median()), _fmt(gg.tuned.median()),
                         _fmt(gg.sp_untuned.median()), _fmt(gg.sp_tuned.median())])
    return _md(["Arm", "n", "seeds", "RMSE untuned", "RMSE tuned",
                "ρ untuned", "ρ tuned"], rows)


CLIFF_T = 0.7          # primary threshold, matching the Table 0 near-neighbour cut
STRATA = ("cliff", "smooth", "distant", "all")


INDOMAIN_ORDER = ["B1_ecfp_histgb", "T0r_untrained_encoder_probe",
                  "T1_chemberta_linear_probe", "T2_chemberta_full_finetune",
                  "T4_indomain_probe", "T5_chained_probe"]


def table_indomain_curve(split="scaffold") -> str:
    df = _read(f"table1_learning_curves__{split}.csv")
    sizes = sorted(df.n_train.unique())
    rows = []
    for arm in INDOMAIN_ORDER:
        d = df[df.arm == arm].set_index("n_train")
        if d.empty:
            continue
        rows.append([ARM_LABEL[arm]]
                    + [_fmt(d.loc[n, "median"]) if n in d.index else "—" for n in sizes]
                    + [_fmt(d.loc[sizes[-1], "spearman_median"]),
                       _fmt(d.loc[sizes[-1], "r2_median"])])
    return _md(["Arm (RMSE ↓)"] + [f"n={n}" for n in sizes] + ["ρ", "R²"], rows)


def table_indomain_contrasts() -> str:
    df = _read("table10_indomain_contrasts.csv")
    df = df[(df.metric == "rmse") & (df.n_train == df.n_train.max())]
    rows = [[f"{r.arm} vs {r.reference}", f"{r.median_delta:+.4f}",
             f"{r.arm_better_in_seeds}/{r.n_seeds}", f"{r.p_raw:.4f}", r.verdict]
            for r in df.itertuples()]
    return _md(["Contrast (full data)", "median ΔRMSE", "arm better in",
                "p", "verdict"], rows)


def table_random_draws() -> str:
    df = _read("table11_random_encoder_draws.csv")
    rows = [[str(int(r.encoder_draw)), _fmt(r.rmse_median, 4)] for r in df.itertuples()]
    rows.append(["**median**", f"**{df.rmse_median.median():.4f}**"])
    return _md(["Random encoder draw", "median RMSE over 10 eval seeds"], rows)


def table_surrogate() -> str:
    df = _read("table9_surrogate_divergence.csv")
    rows = [[r.reference, r.surrogate, str(int(r.n_differences)),
             str(int(r.n_differences_at_catalytic_site)),
             f"{int(r.min_separation_from_catalytic)} residues",
             str(int(r.n_differences_at_zinc_site))] for r in df.itertuples()]
    return _md(["Reference", "Surrogate", "differences", "at catalytic site",
                "nearest catalytic", "at Zn site"], rows)


def table_cliff_pairs() -> str:
    df = _read("table7_cliff_pairs.csv")
    rows = [[_fmt(r.tanimoto_threshold, 1), f"{int(r.similar_pairs):,}",
             f"{int(r.cliff_pairs):,}",
             f"{100 * r.cliff_fraction_of_similar:.1f}%",
             f"{int(r.compounds_in_a_cliff)} / {int(r.n_compounds)}"]
            for r in df.itertuples()]
    return _md(["Tanimoto ≥", "similar pairs", "cliff pairs (Δp > 1)",
                "cliff share of similar", "compounds in ≥1 cliff"], rows)


def table_cliff_strata() -> str:
    """Per-stratum RMSE and skill vs B0 at the primary threshold."""
    df = _read("table6_activity_cliffs.csv")
    df = df[df.tanimoto_threshold == CLIFF_T]
    rows = []
    for arm in ORDER:
        d = df[df.arm == arm].set_index("stratum")
        if d.empty:
            continue
        cells = []
        for s in STRATA:
            if s not in d.index:
                cells.append("—")
                continue
            r = d.loc[s]
            skill = r["skill_vs_b0_median"]
            cells.append(_fmt(r["rmse_median"]) +
                         ("" if pd.isna(skill) else f" ({skill:+.2f})"))
        rows.append([ARM_LABEL[arm]] + cells)
    counts = df[df.arm == "B0_median"].set_index("stratum")["median_compounds"]
    head = [f"{s} (n≈{counts.get(s, float('nan')):.0f})" for s in STRATA]
    return _md(["Arm — RMSE ↓ (skill vs B0 ↑)"] + head, rows)


def table_cliff_paired() -> str:
    df = _read("table8_cliff_paired.csv")
    df = df[(df.tanimoto_threshold == CLIFF_T) & (df.arm != "B0_median")]
    rows = []
    for arm in ORDER:
        d = df[df.arm == arm].set_index("stratum")
        if d.empty:
            continue
        cells = []
        for s in STRATA:
            if s not in d.index:
                cells.append("—")
                continue
            r = d.loc[s]
            cells.append(f"{r['median_delta_rmse']:+.4f} (p={r['p_holm']:.3f})")
        rows.append([ARM_LABEL[arm]] + cells)
    return _md(["Arm — median ΔRMSE vs B1 (Holm p)"] + list(STRATA), rows)



RENDERERS = {
    "contamination": table_contamination,
    "tuning": table_tuning,
    "full_data": table_full_data,
    "paired": table_paired,
    "curve": table_curve,
    "splits": table_splits,
    "audit": table_audit,
    "der": table_der,
    "surrogate": table_surrogate,
    "indomain_curve": table_indomain_curve,
    "indomain_contrasts": table_indomain_contrasts,
    "random_draws": table_random_draws,
    "cliff_pairs": table_cliff_pairs,
    "cliff_strata": table_cliff_strata,
    "cliff_paired": table_cliff_paired,
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
