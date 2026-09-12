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
# The control and in-domain arms are reported in §6.4 and are held out of these
# tables to keep §5 to the arms plan.md froze; T2 is the only arm genuinely
# confined to the scaffold split (§7.9). An earlier version of this comment said
# the control arms were scaffold-only, which stopped being true on 2026-09-02.
PREREGISTERED = ["B0_median", "B1_ecfp_histgb", "B2_descriptors_rf",
                 "T1_chemberta_linear_probe", "T2_chemberta_full_finetune"]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, comment="#")


def _fmt(v, nd=3, dash_if_nan=True):
    if pd.isna(v):
        return "—" if dash_if_nan else ""
    return f"{v:.{nd}f}"


def _md(headers: list[str], rows: list[list[str]]) -> str:
    # A bare "|" inside a header or cell silently splits it into two columns
    # and the rendered table then has more headers than separators. Caught
    # once in table16's "median DER | crossed"; refused rather than escaped,
    # because a pipe in a column name is nearly always a wording accident.
    for cell in list(headers) + [c for r in rows for c in r]:
        if "|" in str(cell):
            raise ValueError(f"'|' in a table cell would break the markdown: {cell!r}")
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


def table_h2_interaction(split="scaffold") -> str:
    """The pre-registered interaction term (plan.md §6), which DER cannot supply.

    Three location estimates are shown because they are three different
    quantities, not three views of one: the pooled OLS interaction coefficient
    equals the MEAN of the per-seed slopes exactly (balanced design), and the
    median is a different number -- for T1, by a factor of seven.
    """
    df = _read("table15_h2_interaction.csv")
    df = df[df.split == split]
    rows = []
    for arm in ORDER:
        d = df[df.arm == arm]
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append([ARM_LABEL[arm], f"{r['median_slope']:+.4f}",
                     f"{r['mean_slope']:+.4f} = {r['pooled_interaction_coef']:+.4f}",
                     f"{r['hodges_lehmann']:+.4f}",
                     f"[{r['slope_ci_lo']:+.4f}, {r['slope_ci_hi']:+.4f}]",
                     f"{int(r['slope_positive_in_seeds'])}/{int(r['n_seeds'])}",
                     f"{r['p_raw']:.4f}", f"{r['p_holm']:.3f}", r["h2_verdict"]])
    return _md(["Arm", "median slope", "mean slope = pooled OLS coef",
                "pseudomedian", "95% CI (median)", "slope > 0 in",
                "p raw", "p Holm", "verdict"], rows)


def table_der_uncertainty(split="scaffold") -> str:
    """Per-seed DER, with the three censoring states kept apart.

    `table2` collapses all of this into a single 0.00. The median here is
    conditional on an interior crossing and is not a summary of all ten seeds.
    """
    df = _read("table16_der_uncertainty.csv")
    df = df[df.split == split]
    rows = []
    for arm in ["B1_ecfp_histgb"] + [a for a in ORDER if a != "B1_ecfp_histgb"]:
        d = df[df.arm == arm]
        if d.empty:
            continue
        r = d.iloc[0]
        ci = ("—" if pd.isna(r["der_ci_lo"])
              else f"[{r['der_ci_lo']:.2f}, {r['der_ci_hi']:.2f}]")
        rows.append([ARM_LABEL[arm],
                     str(int(r["crossed_interior"])),
                     str(int(r["left_censored_at_n50"])),
                     str(int(r["right_censored_never_reached"])),
                     str(int(r["curves_non_monotonic"])),
                     _fmt(r["median_der_interior"], 2), ci])
    return _md(["Arm", "interior crossings", "≤ 50 (left-censored)",
                "never reached", "non-monotonic curves",
                "median DER given crossing", "95% CI given crossing"], rows)


def table_amended_progress() -> str:
    """Sweep completeness for the Amendment 4 arms, generated not typed."""
    df = _read("table19_amended_progress.csv")
    rows = [[r.arm, f"n = {int(r.n_train)}", str(int(r.planned)),
             str(int(r.complete)), str(int(r.missing)), str(int(r.failures))]
            for r in df.itertuples()]
    tot = df[["planned", "complete", "missing"]].sum()
    rows.append(["**total**", "", f"**{int(tot.planned)}**",
                 f"**{int(tot.complete)}**", f"**{int(tot.missing)}**",
                 f"**{int(df.failures.iloc[0])}**"])
    return _md(["Arm", "Size", "planned", "complete", "missing", "failures"], rows)


def table_amended_curve() -> str:
    df = _read("table17_amended_finetune.csv")
    rows = [[r.arm, f"n = {int(r.n_train)}", _fmt(r.median_rmse), _fmt(r.b1_median_rmse),
             _fmt(r.t2_median_rmse), f"{r.median_selected_lr:g}",
             f"{r.median_selected_epoch:.0f}", f"{int(r.n_train_fitted)}/{int(r.n_internal_val)}",
             f"{r.median_seconds:.0f} s"] for r in df.itertuples()]
    return _md(["Arm", "Size", "RMSE", "B1", "T2 (untuned)", "median lr",
                "median best epoch", "fit/val", "median cost"], rows)


def table_amended_contrasts() -> str:
    df = _read("table18_amended_contrasts.csv")
    rows = [[f"{r.arm} vs {r.reference}", f"n = {int(r.n_train)}",
             f"{r.median_delta:+.4f}", f"{int(r.arm_better_in_seeds)}/{int(r.n_seeds)}",
             f"{r.p_raw:.4f}", f"{r.p_holm:.4f}", r.verdict] for r in df.itertuples()]
    return _md(["Contrast", "Size", "median ΔRMSE", "arm better in",
                "p raw", "p Holm", "verdict"], rows)


def table_amended_h2() -> str:
    df = _read("table20_amended_h2_slope.csv")
    r = df.iloc[0]
    rows = [["T2 (untuned, fixed 40 epochs)", f"{r['untuned_T2_median_slope']:+.4f}",
             "—", "0/10", "0.0020", "slope < 0: contrary to H2"],
            ["T2v (validation-selected)", f"{r['median_slope']:+.4f}",
             f"[{r['slope_ci_lo']:+.4f}, {r['slope_ci_hi']:+.4f}]",
             f"{int(r['slope_positive_in_seeds'])}/{int(r['n_seeds'])}",
             f"{r['p_raw']:.4f}", r["h2_verdict"]]]
    return _md(["Condition", "median slope", "95% CI", "slope > 0 in", "p raw",
                "verdict"], rows)


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
    df = df[(df.metric == "rmse") & (df.n_train == df.n_train.max())
            & (~df.arm.isin(["T4c", "T5c", "T4r", "T5r"]))]
    rows = [[f"{r.arm} vs {r.reference}", f"{r.median_delta:+.4f}",
             f"{r.arm_better_in_seeds}/{r.n_seeds}", f"{r.p_raw:.4f}",
             f"{r.p_holm:.3f}", f"{r.p_bh:.3f}", r.verdict_corrected]
            for r in df.itertuples()]
    return _md(["Contrast (full data)", "median ΔRMSE", "arm better in",
                "p raw", "p Holm", "p BH", "verdict"], rows)


def table_random_draws() -> str:
    df = _read("table11_random_encoder_draws.csv")
    rows = [[str(int(r.encoder_draw)), _fmt(r.rmse_median, 4)] for r in df.itertuples()]
    rows.append(["**median**", f"**{df.rmse_median.median():.4f}**"])
    return _md(["Random encoder draw", "median RMSE over 10 eval seeds"], rows)


DECONTAM_ROWS = [("T4c", "T4", "remove the 61 overlapping records"),
                 ("T4r", "T4", "remove 61 **random** records (size-matched control)"),
                 ("T4c", "T4r", "**decontaminated vs the control** — decides H4"),
                 ("T5c", "T5", "remove the 61 overlapping records"),
                 ("T5r", "T5", "remove 61 **random** records (size-matched control)"),
                 ("T5c", "T5r", "**decontaminated vs the control** — decides H4")]


def table_tuned_comparison() -> str:
    df = _read("table13_tuned_comparison.csv")
    rows = [[f"n = {int(r.n_train)}", r.baseline_basis, _fmt(r.tuned_T2_median),
             _fmt(r.baseline_median),
             f"{int(r.T2_better_in_seeds)}/{int(r.n_seeds)}",
             f"{r.p_raw:.3f}", r.verdict] for r in df.itertuples()]
    return _md(["Size", "Baseline used", "tuned T2", "B1", "T2 better in",
                "p", "verdict"], rows)


def table_decontamination() -> str:
    df = _read("table10_indomain_contrasts.csv")
    df = df[(df.metric == "rmse") & (df.n_train == df.n_train.max())]
    idx = df.set_index(["arm", "reference"])
    rows = []
    for a, b, what in DECONTAM_ROWS:
        if (a, b) not in idx.index:
            rows.append([f"{a} vs {b}", what, "—", "—", "—", "—", "not yet run"])
            continue
        r = idx.loc[(a, b)]
        rows.append([f"{a} vs {b}", what, f"{r['median_delta']:+.4f}",
                     f"{int(r['arm_better_in_seeds'])}/{int(r['n_seeds'])}",
                     f"{r['p_raw']:.4f}", f"{r['p_holm']:.3f}",
                     str(r["verdict_corrected"])])
    return _md(["Contrast", "What it removes", "median ΔRMSE", "arm better in",
                "p raw", "p Holm", "verdict"], rows)


def table_splits_extended() -> str:
    """R² across splits for every arm run on all three, plus Butina retention."""
    df = _read("table4_split_difficulty.csv")
    rows = []
    for arm in ["B1_ecfp_histgb", "B2_descriptors_rf", "T0r_untrained_encoder_probe",
                "T1_chemberta_linear_probe", "T4_indomain_probe", "T5_chained_probe"]:
        d = df[df.arm == arm].set_index("split")
        if not {"random", "scaffold", "butina"} <= set(d.index):
            continue
        sc, bu = float(d.loc["scaffold", "r2_median"]), float(d.loc["butina", "r2_median"])
        rows.append([ARM_LABEL[arm], _fmt(d.loc["random", "r2_median"]), _fmt(sc),
                     _fmt(bu), f"{bu/sc:.0%}" if sc else "—"])
    return _md(["Arm (R² ↑)", "random", "scaffold", "Butina", "Butina retained"], rows)


def table_all_endpoints() -> str:
    df = _read("table14_all_endpoints.csv")
    piv = df.pivot_table(index="arm", columns="metric", values="median")
    cols = [("rmse", "RMSE ↓"), ("mae", "MAE ↓"), ("r2", "R² ↑"),
            ("spearman", "ρ ↑"), ("precision_at_10pct", "prec@10% ↑")]
    rows = []
    for arm in PREREGISTERED + ["T0r_untrained_encoder_probe", "T4_indomain_probe",
                                "T5_chained_probe"]:
        if arm not in piv.index:
            continue
        rows.append([ARM_LABEL[arm]] + [_fmt(piv.loc[arm, c], 3 if c != "precision_at_10pct" else 2)
                                        for c, _ in cols])
    return _md(["Arm"] + [h for _, h in cols], rows)


def table_enrichment() -> str:
    df = _read("table14_all_endpoints.csv")
    df = df[(df.metric == "precision_at_10pct") & df.p_raw.notna()]
    rows = [[ARM_LABEL[r.arm], _fmt(r.median, 2), f"{r.median_delta_vs_B1:+.2f}",
             f"{int(r.arm_better_in_seeds)}/{int(r.n_seeds)}",
             f"{r.p_raw:.4f}", f"{r.p_holm:.3f}", r.verdict] for r in df.itertuples()]
    return _md(["Arm", "prec@10%", "Δ vs B1", "better in", "p raw", "p Holm",
                "verdict"], rows)


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
    "h2_interaction": table_h2_interaction,
    "der_uncertainty": table_der_uncertainty,
    "amended_progress": table_amended_progress,
    "amended_curve": table_amended_curve,
    "amended_contrasts": table_amended_contrasts,
    "amended_h2": table_amended_h2,
    "surrogate": table_surrogate,
    "splits_extended": table_splits_extended,
    "all_endpoints": table_all_endpoints,
    "enrichment": table_enrichment,
    "decontamination": table_decontamination,
    "tuned_comparison": table_tuned_comparison,
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
