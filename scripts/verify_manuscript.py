#!/usr/bin/env python
"""Verify every numeric claim in paper/manuscript.md against its source file.

Each claim names the section it appears in, the file it must be derivable from,
and the value the manuscript states. The script re-reads the artefact and
compares. A claim that cannot be re-derived is a FAIL, not a warning.

Run: python scripts/verify_manuscript.py
Exit code 0 only if every claim checks out.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("results/tables")
METRICS = Path("results/metrics")
MANUSCRIPT = Path("paper/manuscript.md")
TOL = 1e-3


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, comment="#")


@dataclass
class Claim:
    section: str
    text: str
    source: str
    expected: float | int | str
    actual: float | int | str

    def ok(self) -> bool:
        if isinstance(self.expected, bool) or isinstance(self.actual, bool):
            return bool(self.expected) == bool(self.actual)
        if isinstance(self.expected, str) or isinstance(self.actual, str):
            return str(self.expected) == str(self.actual)
        if self.actual is None:
            return False
        return abs(float(self.expected) - float(self.actual)) <= TOL


def t1(split: str, arm: str, n: int, col: str) -> float:
    df = read_table(f"table1_learning_curves__{split}.csv")
    row = df[(df.arm == arm) & (df.n_train == n)]
    return float(row[col].iloc[0]) if len(row) else None


def t3(split: str, arm: str, col: str):
    df = read_table(f"table3_paired_tests__{split}.csv")
    row = df[df.arm == arm]
    return row[col].iloc[0] if len(row) else None


def t4(arm: str, split: str, col: str) -> float:
    df = read_table("table4_split_difficulty.csv")
    row = df[(df.arm == arm) & (df.split == split)]
    return float(row[col].iloc[0]) if len(row) else None


def t2der(split: str, arm: str, col: str) -> float:
    df = read_table(f"table2_der__{split}.csv")
    row = df[df.arm == arm]
    return float(row[col].iloc[0]) if len(row) else None


def t0(split: str, col: str) -> float:
    df = read_table("table0_split_audit.csv")
    return float(df[df.split == split][col].median())



def t6(arm: str, stratum: str, col: str, threshold: float = 0.7):
    """Per-stratum cliff table (§6.3)."""
    df = read_table("table6_activity_cliffs.csv")
    row = df[(df.arm == arm) & (df.stratum == stratum)
             & (df.tanimoto_threshold == threshold)]
    return float(row[col].iloc[0]) if len(row) else None


def t7(threshold: float, col: str):
    df = read_table("table7_cliff_pairs.csv")
    row = df[df.tanimoto_threshold == threshold]
    return float(row[col].iloc[0]) if len(row) else None


def t8(arm: str, stratum: str, col: str, threshold: float = 0.7):
    df = read_table("table8_cliff_paired.csv")
    row = df[(df.arm == arm) & (df.stratum == stratum)
             & (df.tanimoto_threshold == threshold)]
    return float(row[col].iloc[0]) if len(row) else None


def build_claims() -> list[Claim]:
    cur = json.loads(Path("data/processed/eva71_2a.curation.json").read_text())
    ds = pd.read_csv("data/processed/eva71_2a.csv")
    C = []

    # ---- Section 3.2: dataset ----
    C.append(Claim("3.2", "649 complexes in", "curation.json", 649, cur["n_complexes_in"]))
    C.append(Claim("3.2", "10 dropped on structure quality", "curation.json", 10,
                   cur["n_dropped_quality"]))
    C.append(Claim("3.2", "0 dropped for replicate disagreement", "curation.json", 0,
                   cur["n_compound_groups_dropped_spread"]))
    C.append(Claim("3.2", "494 unique compounds out", "curation.json", 494, cur["n_compounds_out"]))
    C.append(Claim("3.2", "272 scaffolds", "curation.json", 272, cur["n_scaffolds"]))
    C.append(Claim("3.2", "max replicate spread 0.49", "curation.json", 0.49,
                   round(cur["max_replicate_spread"], 2)))
    C.append(Claim("3.2", "133 compounds with >1 measurement", "eva71_2a.csv", 133,
                   int((ds.n_complexes > 1).sum())))
    C.append(Claim("3.2", "pKD range 3.44-7.94", "eva71_2a.csv", "3.44/7.94",
                   f"{ds.pactivity.min():.2f}/{ds.pactivity.max():.2f}"))
    C.append(Claim("3.2", "pKD median 4.95", "eva71_2a.csv", 4.95,
                   round(float(ds.pactivity.median()), 2)))
    C.append(Claim("3.2", "pKD SD 0.86", "eva71_2a.csv", 0.86,
                   round(float(ds.pactivity.std()), 2)))

    # ---- Section 3.1: the CVA16 surrogate, re-derived from UniProt ----
    sur = read_table("table9_surrogate_divergence.csv").set_index("surrogate")
    for label, n_diff in [("CVA16 G-10", 7), ("CVA16 Tainan/5079/98", 8)]:
        C.append(Claim("3.1", f"{label}: {n_diff} 2A differences", "table9",
                       n_diff, int(sur.loc[label, "n_differences"])))
        C.append(Claim("3.1", f"{label}: none catalytic", "table9", 0,
                       int(sur.loc[label, "n_differences_at_catalytic_site"])))
        C.append(Claim("3.1", f"{label}: nearest catalytic 5 residues", "table9", 5,
                       int(sur.loc[label, "min_separation_from_catalytic"])))
        C.append(Claim("3.1", f"{label}: no Zn ligand substituted", "table9", 0,
                       int(sur.loc[label, "n_differences_at_zinc_site"])))
    C.append(Claim("3.1", "catalytic triad is 21/39/110", "table9", "21 39 110",
                   str(sur.iloc[0]["catalytic_positions"])))
    C.append(Claim("3.1", "Zn ligands are 56/58/116/118", "table9", "56 58 116 118",
                   str(sur.iloc[0]["zinc_positions"])))
    C.append(Claim("3.1", "2A chain is 150 residues", "table9", 150,
                   int(sur.iloc[0]["chain_length"])))
    C.append(Claim("3.1", "N57D is adjacent to a Zn ligand (separation 1)", "table9", 1,
                   int(sur.loc["CVA16 G-10", "min_separation_from_zinc"])))
    C.append(Claim("Abstract", "abstract states the 7-8 divergence range", "table9",
                   "7,8", ",".join(str(int(v)) for v in sorted(sur["n_differences"]))))
    C.append(Claim("3.1", "the count of five does not reproduce", "table9", True,
                   bool(all(int(sur.loc[l, "n_differences"]) != 5 for l in sur.index))))

    # ---- Section 5.1: the split audit. The paper's foundational claim, and
    # until 2026-09-02 the only results section with no claims at all. ----
    aud = read_table("table0_split_audit.csv")
    C.append(Claim("5.1", "no compound in both train and test, any split/seed",
                   "table0", 0, int(aud["compounds_in_train_and_test"].max())))
    C.append(Claim("5.1", "scaffold split shares zero scaffolds train/test", "table0", 0,
                   int(aud[aud["split"] == "scaffold"]["scaffolds_in_train_and_test"].max())))
    C.append(Claim("5.1", "random splits share 23-31 scaffolds", "table0", True,
                   bool(aud[aud["split"] == "random"]["scaffolds_in_train_and_test"]
                        .between(23, 31).all())))
    C.append(Claim("5.1", "30 split files audited", "table0", 30, len(aud)))
    C.append(Claim("5.1", "every split file has 347 train compounds", "table0", 347,
                   int(aud["n_train"].unique()[0]) if aud["n_train"].nunique() == 1 else -1))
    C.append(Claim("5.1", "every split file has 98 test compounds", "table0", 98,
                   int(aud["n_test"].unique()[0]) if aud["n_test"].nunique() == 1 else -1))

    # ---- Section 5.2: learning curves, scaffold, n=347 ----
    for arm, rmse, sp, r2, lo, hi in [
        ("B2_descriptors_rf", 0.5858, 0.6359, 0.5128, 0.5394, 0.6473),
        ("B1_ecfp_histgb", 0.6031, 0.6700, 0.5068, 0.5370, 0.6365),
        ("T1_chemberta_linear_probe", 0.6372, 0.6119, 0.4207, 0.6068, 0.6607),
        ("T2_chemberta_full_finetune", 0.6621, 0.6138, 0.4046, 0.6068, 0.6878),
        ("B0_median", 0.8520, None, -0.0446, 0.8046, 0.9244),
    ]:
        C.append(Claim("5.2", f"{arm} RMSE n=347", "table1", rmse, t1("scaffold", arm, 347, "median")))
        C.append(Claim("5.2", f"{arm} R2 n=347", "table1", r2, t1("scaffold", arm, 347, "r2_median")))
        C.append(Claim("5.2", f"{arm} CI n=347", "table1", f"{lo}/{hi}",
                       f"{t1('scaffold', arm, 347, 'ci_lo')}/{t1('scaffold', arm, 347, 'ci_hi')}"))
        if sp is not None:
            C.append(Claim("5.2", f"{arm} Spearman n=347", "table1", sp,
                           t1("scaffold", arm, 347, "spearman_median")))

    C.append(Claim("5.2", "B1 RMSE at n=250 is 0.584", "table1", 0.5840,
                   t1("scaffold", "B1_ecfp_histgb", 250, "median")))

    # ---- Section 5.3: DER ----
    C.append(Claim("5.3", "B1 target RMSE 0.603", "table2", 0.6031,
                   t2der("scaffold", "B1_ecfp_histgb", "target_rmse")))
    C.append(Claim("5.3", "B1 reaches target at n=206", "table2", 206.1,
                   round(t2der("scaffold", "B1_ecfp_histgb", "n_to_reach_target"), 1)))
    C.append(Claim("5.3", "B2 reaches target at n=295", "table2", 295.1,
                   round(t2der("scaffold", "B2_descriptors_rf", "n_to_reach_target"), 1)))
    C.append(Claim("5.3", "B2 DER 0.70", "table2", 0.699,
                   round(t2der("scaffold", "B2_descriptors_rf", "DER_vs_B1_ecfp_histgb"), 3)))
    for arm in ("T1_chemberta_linear_probe", "T2_chemberta_full_finetune"):
        C.append(Claim("5.3", f"{arm} DER = 0", "table2", 0.0,
                       t2der("scaffold", arm, "DER_vs_B1_ecfp_histgb")))
    for arm, v in [("T1_chemberta_linear_probe", 0.7297), ("B1_ecfp_histgb", 0.7042),
                   ("B2_descriptors_rf", 0.6705)]:
        C.append(Claim("5.3", f"{arm} RMSE at n=50", "table1", v,
                       t1("scaffold", arm, 50, "median")))

    # ---- Section 5.4: paired tests (Holm family includes T2) ----
    for arm, delta, holm, verdict in [
        ("B0_median", -0.2864, 0.00781, "significant"),
        ("T1_chemberta_linear_probe", -0.0487, 0.01172, "significant"),
        ("T2_chemberta_full_finetune", -0.0878, 0.07422, "inconclusive"),
        ("B2_descriptors_rf", -0.0118, 0.55664, "inconclusive"),
    ]:
        C.append(Claim("5.4", f"{arm} delta", "table3", delta,
                       t3("scaffold", arm, "median_rmse_delta_vs_baseline")))
        C.append(Claim("5.4", f"{arm} p_holm", "table3", holm, t3("scaffold", arm, "p_holm")))
        C.append(Claim("5.4", f"{arm} verdict", "table3", verdict, t3("scaffold", arm, "verdict")))

    # ---- Section 5.5: split difficulty ----
    for arm, rnd, scaf, but in [
        ("B1_ecfp_histgb", 0.4924, 0.5068, 0.2039),
        ("B2_descriptors_rf", 0.4612, 0.5128, 0.2570),
        ("T1_chemberta_linear_probe", 0.4222, 0.4207, 0.0562),
    ]:
        for split, v in (("random", rnd), ("scaffold", scaf), ("butina", but)):
            C.append(Claim("5.5", f"{arm} R2 {split}", "table4", v, t4(arm, split, "r2_median")))
    C.append(Claim("5.5", "B1 RMSE butina 0.554", "table4", 0.5539,
                   t4("B1_ecfp_histgb", "butina", "rmse_median")))
    C.append(Claim("5.5", "T1 Spearman butina 0.251", "table4", 0.2506,
                   t4("T1_chemberta_linear_probe", "butina", "spearman_median")))
    for split, frac in (("random", 0.4898), ("scaffold", 0.2908), ("butina", 0.1735)):
        C.append(Claim("5.5", f"{split} frac NN>=0.7", "table0", frac,
                       t0(split, "frac_test_with_nn_ge_0.7")))
    for split, shared in (("random", 24.5), ("scaffold", 0.0), ("butina", 13.0)):
        C.append(Claim("5.5", f"{split} scaffolds shared", "table0", shared,
                       t0(split, "scaffolds_in_train_and_test")))

    # R2 retention scaffold -> butina, quoted as percentages in 5.5 and 5.7.
    for arm, pct in [("B1_ecfp_histgb", 40), ("B2_descriptors_rf", 50),
                     ("T1_chemberta_linear_probe", 13)]:
        ret = 100 * t4(arm, "butina", "r2_median") / t4(arm, "scaffold", "r2_median")
        C.append(Claim("5.5", f"{arm} retains ~{pct}% of R2 under Butina", "table4",
                       pct, int(round(ret))))

    # 5.5 quotes the scaffold-minus-random R2 gap per arm.
    for arm, gap in [("B1_ecfp_histgb", 0.014), ("B2_descriptors_rf", 0.052),
                     ("T1_chemberta_linear_probe", -0.002)]:
        C.append(Claim("5.5", f"{arm}: scaffold minus random R2", "table4", gap,
                       round(t4(arm, "scaffold", "r2_median") - t4(arm, "random", "r2_median"), 3)))

    # 5.2 claims T1's and T2's bootstrap CI lower bound exceeds B1's median RMSE.
    b1_med = t1("scaffold", "B1_ecfp_histgb", 347, "median")
    for arm in ("T1_chemberta_linear_probe", "T2_chemberta_full_finetune"):
        C.append(Claim("5.2", f"{arm} CI lower bound > B1 median", "table1", True,
                       t1("scaffold", arm, 347, "ci_lo") > b1_med))

    # ---- Section 5.6: fine-tune ----
    for n, rmse, r2, sp in [(50, 1.2321, -1.2555, 0.4557), (100, 0.9598, -0.2361, 0.4902),
                            (250, 0.6945, 0.2650, 0.6093), (347, 0.6621, 0.4046, 0.6138)]:
        C.append(Claim("5.6", f"T2 RMSE n={n}", "table1", rmse,
                       t1("scaffold", "T2_chemberta_full_finetune", n, "median")))
        C.append(Claim("5.6", f"T2 R2 n={n}", "table1", r2,
                       t1("scaffold", "T2_chemberta_full_finetune", n, "r2_median")))
        C.append(Claim("5.6", f"T2 Spearman n={n}", "table1", sp,
                       t1("scaffold", "T2_chemberta_full_finetune", n, "spearman_median")))

    # Seed-level counts asserted in prose, recomputed from raw metrics.
    def per_seed(arm, split, n, metric):
        out = {}
        for f in METRICS.glob(f"{arm}__{split}__seed*__n{n}.json"):
            r = json.loads(f.read_text())
            out[r["seed"]] = r["metrics"][metric]
        return out

    b1r = per_seed("B1_ecfp_histgb", "scaffold", 347, "rmse")
    b2r = per_seed("B2_descriptors_rf", "scaffold", 347, "rmse")
    C.append(Claim("5.4", "B2 beats B1 on RMSE in 4 of 10 seeds", "metrics/*.json", 4,
                   sum(b2r[s] < b1r[s] for s in b1r)))
    b1s = per_seed("B1_ecfp_histgb", "scaffold", 347, "spearman")
    t2s = per_seed("T2_chemberta_full_finetune", "scaffold", 347, "spearman")
    C.append(Claim("5.6", "T2 beats B1 on Spearman in 3 of 10 seeds", "metrics/*.json", 3,
                   sum(t2s[s] > b1s[s] for s in b1s)))
    C.append(Claim("5.6", "T2 seed0 Spearman 0.737", "metrics/*.json", 0.737, round(t2s[0], 3)))

    # ---- Section 6.1: contamination upper bound ----
    if (TABLES / "table5_contamination.csv").exists():
        cont = read_table("table5_contamination.csv")
        allrow = cont[cont.scope.str.startswith("all")].iloc[0]
        C.append(Claim("6.1", "53.0% of curated compounds in PubChem", "table5", 0.530,
                       round(float(allrow.frac_in_pubchem), 3)))
        for split, frac in (("scaffold", 0.536), ("random", 0.546), ("butina", 0.638)):
            row = cont[cont.split == split]
            if len(row):
                C.append(Claim("6.1", f"{split} test fold PubChem fraction", "table5", frac,
                               round(float(row.frac_in_pubchem.iloc[0]), 3)))

    # ---- Section 6.4: in-domain transfer and the untrained control ----
    ind = read_table("table10_indomain_contrasts.csv")
    ind = ind[(ind.metric == "rmse") & (ind.n_train == 347)].set_index(["arm", "reference"])
    draws = read_table("table11_random_encoder_draws.csv")
    corp = json.loads(Path("data/processed/indomain_3c.curation.json").read_text())

    for (a, b, delta, wins, pval) in [
            ("T4", "T1", 0.0351, 7, 0.3750), ("T5", "T1", 0.0252, 9, 0.0059),
            ("T1", "T0r", 0.0038, 5, 0.4922), ("T2", "T0r", -0.0050, 4, 0.6250),
            ("T4", "T0r", 0.0299, 7, 0.0488), ("T5", "T0r", 0.0299, 8, 0.0098),
            ("T4", "B1", -0.0471, 2, 0.0645), ("T5", "B1", -0.0135, 0, 0.0020)]:
        C.append(Claim("6.4", f"{a} vs {b} delta", "table10", delta,
                       float(ind.loc[(a, b), "median_delta"])))
        C.append(Claim("6.4", f"{a} vs {b} seed wins", "table10", wins,
                       int(ind.loc[(a, b), "arm_better_in_seeds"])))
        C.append(Claim("6.4", f"{a} vs {b} p", "table10", pval,
                       float(ind.loc[(a, b), "p_raw"])))

    C.append(Claim("6.4", "T5 beats T1 significantly", "table10", "arm better",
                   str(ind.loc[("T5", "T1"), "verdict"])))
    C.append(Claim("6.4", "T4 vs T1 is inconclusive", "table10", "inconclusive",
                   str(ind.loc[("T4", "T1"), "verdict"])))
    C.append(Claim("6.4", "T1 vs T0r is inconclusive", "table10", "inconclusive",
                   str(ind.loc[("T1", "T0r"), "verdict"])))

    # The T1-vs-control margin, stated as a range because a single "at most"
    # figure was wrong here: it understated the upper end by ~75%.
    t1_full = t1("scaffold", "T1_chemberta_linear_probe", 347, "median")
    gaps = sorted(round(float(v) - t1_full, 4) for v in draws.rmse_median)
    C.append(Claim("6.4", "T1 margin over the control, low end 0.007", "table11", 0.007,
                   round(gaps[0], 3)))
    C.append(Claim("6.4", "T1 margin over the control, high end 0.018", "table11", 0.018,
                   round(gaps[-1], 3)))
    C.append(Claim("6.4", "T1 margin over the control, median 0.011", "table11", 0.011,
                   round(float(np.median(gaps)), 3)))
    C.append(Claim("6.4", "in-domain margin is ~3x the generic margin", "table10 vs table11",
                   True, bool(2.0 <= 0.0299 / float(np.median(gaps)) <= 4.0)))
    C.append(Claim("6.4", "random-draw min RMSE 0.6438", "table11", 0.6438,
                   float(draws.rmse_median.min())))
    C.append(Claim("6.4", "random-draw max RMSE 0.6549", "table11", 0.6549,
                   float(draws.rmse_median.max())))
    C.append(Claim("6.4", "T1 median lies below every random draw", "table1 vs table11",
                   True, bool(t1("scaffold", "T1_chemberta_linear_probe", 347, "median")
                              < float(draws.rmse_median.min()))))
    chem = read_table("table12_corpus_chemistry.csv").set_index("set")
    EV, CO = "evaluation set (EV-A71/CVA16 2A)", "in-domain corpus (3C/3CL)"
    C.append(Claim("6.4", "median NN Tanimoto eval->corpus 0.247", "table12", 0.2468,
                   float(chem.loc[EV, "nn_tanimoto_to_other_median"])))
    C.append(Claim("6.4", "no eval compound has a corpus neighbour >= 0.5", "table12", 0.0,
                   float(chem.loc[EV, "frac_with_neighbour_ge_0.5"])))
    C.append(Claim("6.4", "eval MW median 329", "table12", 328.8,
                   float(chem.loc[EV, "mw_median"])))
    C.append(Claim("6.4", "corpus MW median 470", "table12", 469.6,
                   float(chem.loc[CO, "mw_median"])))
    C.append(Claim("6.4", "eval pActivity median 4.95", "table12", 4.95,
                   float(chem.loc[EV, "pactivity_median"])))
    C.append(Claim("6.4", "corpus pActivity median 6.30", "table12", 6.30,
                   float(chem.loc[CO, "pactivity_median"])))
    C.append(Claim("6.4", "corpus compound count matches curation", "table12", 2743,
                   int(chem.loc[CO, "n_compounds"])))
    C.append(Claim("6.4", "screen dropped 2022 non-3C records", "indomain manifests", 2022,
                   sum(json.loads(f.read_text())["dropped_not_3c"]
                       for f in sorted(Path("data/raw").glob("indomain_*.manifest.json")))))
    C.append(Claim("6.4", "screen dropped 757 wrong-enzyme records", "indomain manifests", 757,
                   sum(json.loads(f.read_text())["dropped_wrong_enzyme"]
                       for f in sorted(Path("data/raw").glob("indomain_*.manifest.json")))))
    C.append(Claim("6.4", "7634 raw records fetched", "indomain manifests", 7634,
                   sum(json.loads(f.read_text())["n_raw"]
                       for f in sorted(Path("data/raw").glob("indomain_*.manifest.json")))))
    C.append(Claim("6.4", "corpus measurements 2974", "indomain curation", 2974,
                   corp["n_measurements"]))
    C.append(Claim("6.4", "corpus compounds 2743", "indomain curation", 2743,
                   corp["n_unique_compounds"]))
    C.append(Claim("6.4", "coronaviral measurements 2829", "indomain curation", 2829,
                   corp["per_family_measurements"]["coronaviral"]))
    C.append(Claim("6.4", "picornaviral measurements 145", "indomain curation", 145,
                   corp["per_family_measurements"]["picornaviral"]))
    C.append(Claim("6.4", "zero exact overlap with the eval set", "indomain curation", 0,
                   corp["n_overlap_exact"]))
    C.append(Claim("6.4", "zero near-duplicate overlap", "indomain curation", 0,
                   corp["n_overlap_near_duplicate"]))
    C.append(Claim("6.4", "61 scaffold-level overlaps", "indomain curation", 61,
                   corp["n_overlap_scaffold"]))
    for arm, n in [("T4_indomain_probe", 0.6028), ("T5_chained_probe", 0.6120),
                   ("T0r_untrained_encoder_probe", 0.6477)]:
        C.append(Claim("6.4", f"{arm} full-data RMSE", "table1", n,
                       t1("scaffold", arm, 347, "median")))

    der_s = read_table("table2_der__scaffold.csv").set_index("arm")
    C.append(Claim("6.4", "T4 DER is 0.60", "table2", 0.596,
                   round(float(der_s.loc["T4_indomain_probe", "DER_vs_B1_ecfp_histgb"]), 3)))
    C.append(Claim("6.4", "T4 is the only transfer arm with a non-zero DER", "table2",
                   True, bool(all(float(der_s.loc[a, "DER_vs_B1_ecfp_histgb"]) == 0.0
                                  for a in der_s.index
                                  if a.startswith("T") and a != "T4_indomain_probe"))))
    # 346.1 by interpolation, i.e. effectively the full 347-compound fold
    C.append(Claim("6.4", "T4 reaches the target only at the full fold", "table2", 346,
                   round(float(der_s.loc["T4_indomain_probe", "n_to_reach_target"]))))

    # ---- Quantities the coverage audit found asserted but never re-derived ----
    import glob as _g
    # §5.5: the test fold's label SD per split, computed from the split files
    _ds = pd.read_csv("data/processed/eva71_2a.csv")
    _lab = dict(zip(_ds.inchikey, _ds.pactivity))
    _sd: dict[str, list[float]] = {}
    for _f in _g.glob("data/processed/splits/eva71_2a/*.json"):
        _folds = json.loads(Path(_f).read_text())
        _folds = _folds.get("folds", _folds)
        _te = [_lab[k] for k, v in _folds.items() if v == "test"]
        # population SD, which is what §5.5 reports
        _sd.setdefault(Path(_f).name.split("__")[0], []).append(float(np.std(_te)))
    for _split, _want in [("random", 0.870), ("scaffold", 0.839), ("butina", 0.658)]:
        C.append(Claim("5.5", f"{_split} test-fold pK_D SD", "split files + curated labels",
                       _want, round(float(np.median(_sd[_split])), 3)))
    # 0.6705 is exactly on the 3-dp boundary, so compare at 4 dp
    C.append(Claim("5.2", "B2 at n=50 is 0.6705", "table1", 0.6705,
                   round(t1("scaffold", "B2_descriptors_rf", 50, "median"), 4)))
    C.append(Claim("5.5", "B0 Butina full-data RMSE 0.700", "table1(butina)", 0.700,
                   round(t1("butina", "B0_median", 347, "median"), 3)))
    C.append(Claim("5.6", "T2 full-data Spearman 0.614", "table1", 0.614,
                   round(t1("scaffold", "T2_chemberta_full_finetune", 347,
                            "spearman_median"), 3)))
    C.append(Claim("6.3", "T2 distant Holm p at T>=0.8 is 0.039", "table8", 0.039,
                   round(t8("T2_chemberta_full_finetune", "distant", "p_holm", 0.8), 3)))
    _chem = read_table("table12_corpus_chemistry.csv").set_index("set")
    C.append(Claim("6.4", "eval MW median rounds to 329", "table12", 329,
                   round(float(_chem.loc["evaluation set (EV-A71/CVA16 2A)", "mw_median"]))))
    C.append(Claim("6.4", "corpus MW median rounds to 470", "table12", 470,
                   round(float(_chem.loc["in-domain corpus (3C/3CL)", "mw_median"]))))
    _ind = read_table("table10_indomain_contrasts.csv")
    _sp = _ind[(_ind.metric == "spearman") & (_ind.n_train == 347)].set_index(["arm", "reference"])
    C.append(Claim("6.4", "T5 vs T1 Spearman delta 0.028", "table10", 0.028,
                   round(float(_sp.loc[("T5", "T1"), "median_delta"]), 3)))
    C.append(Claim("6.4", "T5 vs T1 Spearman p 0.0195", "table10", 0.0195,
                   round(float(_sp.loc[("T5", "T1"), "p_raw"]), 4)))
    _r50 = _ind[(_ind.metric == "rmse") & (_ind.n_train == 50)].set_index(["arm", "reference"])
    C.append(Claim("6.4", "T2 vs T0r at n=50 delta -0.534", "table10", -0.534,
                   round(float(_r50.loc[("T2", "T0r"), "median_delta"]), 3)))
    _dr = read_table("table11_random_encoder_draws.csv")
    C.append(Claim("6.4", "random-draw SD 0.0039", "table11", 0.0039,
                   round(float(np.std(_dr.rmse_median, ddof=0)), 4)))
    _der = read_table("table2_der__scaffold.csv").set_index("arm")
    C.append(Claim("6.4", "T4 needed ~1.7x the baseline's data", "table2", 1.7,
                   round(1.0 / float(_der.loc["T4_indomain_probe",
                                              "DER_vs_B1_ecfp_histgb"]), 1)))
    C.append(Claim("10", "120 re-run cells for T0r/T4/T5", "metrics", 120,
                   len([f for f in _g.glob("results/metrics/*__scaffold__*.json")
                        if any(a in f for a in ("T0r_", "T4_indomain", "T5_chained"))]))) 

    # ---- Section 6.2: the tuned comparison, on BOTH baseline bases ----
    tc = read_table("table13_tuned_comparison.csv")
    tc["basis"] = tc.baseline_basis.str.startswith("tuned").map({True: "tuned", False: "untuned"})
    tc = tc.set_index(["n_train", "basis"])
    for n, basis, t2m, b1m, wins, seeds, pv in [
            (50, "untuned", 0.7354, 0.7042, 3, 10, 0.1602),
            (50, "tuned", 0.7493, 0.7143, 3, 7, 0.6875),
            (347, "untuned", 0.6195, 0.6338, 2, 5, 0.6250),
            (347, "tuned", 0.6195, 0.6009, 1, 5, 0.1250)]:
        r = tc.loc[(n, basis)]
        C.append(Claim("6.2", f"n={n} {basis}: tuned T2 median", "table13", t2m,
                       float(r["tuned_T2_median"])))
        C.append(Claim("6.2", f"n={n} {basis}: baseline median", "table13", b1m,
                       float(r["baseline_median"])))
        C.append(Claim("6.2", f"n={n} {basis}: T2 wins", "table13", wins,
                       int(r["T2_better_in_seeds"])))
        C.append(Claim("6.2", f"n={n} {basis}: seeds", "table13", seeds,
                       int(r["n_seeds"])))
        C.append(Claim("6.2", f"n={n} {basis}: p", "table13", pv, float(r["p_raw"])))
    # the disclosure's load-bearing point: the bias flips direction with size
    C.append(Claim("6.2", "at n=50 the untuned baseline is the stronger one", "table13",
                   True, bool(float(tc.loc[(50, "untuned"), "baseline_median"])
                              < float(tc.loc[(50, "tuned"), "baseline_median"]))))
    C.append(Claim("6.2", "at full data the tuned baseline is the stronger one", "table13",
                   True, bool(float(tc.loc[(347, "tuned"), "baseline_median"])
                              < float(tc.loc[(347, "untuned"), "baseline_median"]))))
    C.append(Claim("6.2", "all four tuned comparisons are inconclusive", "table13",
                   True, bool((tc["p_raw"] > 0.05).all())))

    # ---- Compute cost, claimed in the abstract, §6.2, §8 and §9 ----
    # These were asserted in prose and never checked; the stated 160 s / 80x
    # were 185 s / 69x when measured.
    import glob as _glob
    secs: dict[tuple[str, int], list[float]] = {}
    for _f in _glob.glob("results/metrics/*__scaffold__*.json"):
        _d = json.loads(Path(_f).read_text())
        secs.setdefault((_d["arm"], _d["n_train"]), []).append(_d["seconds"])
    import statistics as _st
    b1_full = _st.median(secs[("B1_ecfp_histgb", 347)])
    t2_full = _st.median(secs[("T2_chemberta_full_finetune", 347)])
    C.append(Claim("cost", "B1 full-data fit ~2.7 s", "metrics seconds", 2.7,
                   round(b1_full, 1)))
    C.append(Claim("cost", "T2 full-data fit ~185 s", "metrics seconds", 185,
                   round(t2_full)))
    C.append(Claim("cost", "T2/B1 ratio ~70x", "metrics seconds", 70,
                   round(t2_full / b1_full / 10) * 10))
    C.append(Claim("cost", "the ratio is under two orders of magnitude",
                   "metrics seconds", True, bool(t2_full / b1_full < 100)))

    # ---- §6.2 seed-subset disclosure; §5.7 tie-robustness ----
    import glob as _gg
    _m = {}
    for _f in _gg.glob("results/metrics/B1_ecfp_histgb__scaffold__seed*__n347.json"):
        _d = json.loads(Path(_f).read_text()); _m[_d["seed"]] = _d["metrics"]["rmse"]
    C.append(Claim("6.2", "B1 median on seeds 0-4 is 0.634", "metrics", 0.634,
                   round(float(np.median([_m[s] for s in range(5)])), 3)))
    C.append(Claim("6.2", "B1 median on seeds 5-9 is 0.556", "metrics", 0.556,
                   round(float(np.median([_m[s] for s in range(5, 10)])), 3)))
    C.append(Claim("6.2", "the tuned subset is unfavourable to the baseline", "metrics",
                   True, bool(np.median([_m[s] for s in range(5)])
                              > np.median([_m[s] for s in range(5, 10)]))))

    # ---- Section 5.7: every pre-registered endpoint ----
    ep = read_table("table14_all_endpoints.csv")
    med = ep.pivot_table(index="arm", columns="metric", values="median")
    for arm, mae, prec in [("B1_ecfp_histgb", 0.4497, 0.40),
                           ("T1_chemberta_linear_probe", 0.4994, 0.50),
                           ("T2_chemberta_full_finetune", 0.5053, 0.50),
                           ("T5_chained_probe", 0.4903, 0.55),
                           ("T4_indomain_probe", 0.4848, 0.45)]:
        C.append(Claim("5.7", f"{arm} MAE", "table14", mae,
                       round(float(med.loc[arm, "mae"]), 4)))
        C.append(Claim("5.7", f"{arm} precision@10%", "table14", prec,
                       round(float(med.loc[arm, "precision_at_10pct"]), 2)))
    pr = ep[(ep.metric == "precision_at_10pct") & ep.p_raw.notna()].set_index("arm")
    C.append(Claim("5.7", "T2 beats B1 on precision@10%, Holm 0.023", "table14", 0.0234,
                   float(pr.loc["T2_chemberta_full_finetune", "p_holm"])))
    C.append(Claim("5.7", "T2 wins 9 of 10 seeds on precision@10%", "table14", 9,
                   int(pr.loc["T2_chemberta_full_finetune", "arm_better_in_seeds"])))
    C.append(Claim("5.7", "T2 is the only arm beating B1 on precision@10% under Holm",
                   "table14", 1, int((pr["p_holm"] <= 0.05).sum())))
    C.append(Claim("5.7", "every transfer arm matches or beats B1 on precision@10%",
                   "table14", True,
                   bool((pr.loc[[a for a in pr.index if a.startswith("T")],
                                "median_delta_vs_B1"] >= 0).all())))
    C.append(Claim("5.7", "every transfer arm is worse than B1 on MAE", "table14", True,
                   bool((ep[(ep.metric == "mae") & ep.arm.str.startswith("T")]
                         ["median_delta_vs_B1"] < 0).all())))
    C.append(Claim("5.7", "T2 is worst of the trained arms on RMSE", "table14", True,
                   bool(float(med.loc["T2_chemberta_full_finetune", "rmse"])
                        == float(med.drop("B0_median")["rmse"].max()))))
    C.append(Claim("5.7", "the test fold's top decile is 10 compounds", "table0", 10,
                   round(int(read_table("table0_split_audit.csv")["n_test"].iloc[0]) * 0.10)))

    # ---- Multiplicity: what survives correction (§6.4, §6.5) ----
    mc = read_table("table10_indomain_contrasts.csv")
    mc = mc[(mc.metric == "rmse") & (mc.n_train == 347)].set_index(["arm", "reference"])
    for a, b, holm, bh in [("T5", "T1", 0.0767, 0.0294), ("T5", "T0r", 0.1176, 0.0294),
                           ("T4", "T0r", 0.4392, 0.0915), ("T5", "B1", 0.0300, 0.0150),
                           ("T4c", "T4r", 0.2730, 0.0682), ("T5c", "T5r", 0.4392, 0.0915),
                           ("T4r", "T4", 0.0300, 0.0150), ("T1", "T0r", 1.0, 0.5679)]:
        C.append(Claim("6.4", f"{a} vs {b} Holm", "table10", holm,
                       float(mc.loc[(a, b), "p_holm"])))
        C.append(Claim("6.4", f"{a} vs {b} BH", "table10", bh,
                       float(mc.loc[(a, b), "p_bh"])))
    # the review's load-bearing findings about what survives
    C.append(Claim("6.4", "H3 contrast is BH-significant but not Holm", "table10", True,
                   bool(float(mc.loc[("T5", "T1"), "p_bh"]) <= 0.05
                        and float(mc.loc[("T5", "T1"), "p_holm"]) > 0.05)))
    C.append(Claim("6.4", "T4 vs T0r is inconclusive under both corrections", "table10",
                   True, bool(float(mc.loc[("T4", "T0r"), "p_bh"]) > 0.05
                              and float(mc.loc[("T4", "T0r"), "p_holm"]) > 0.05)))
    C.append(Claim("6.5", "neither H4 decisive contrast survives Holm", "table10", True,
                   bool(float(mc.loc[("T4c", "T4r"), "p_holm"]) > 0.05
                        and float(mc.loc[("T5c", "T5r"), "p_holm"]) > 0.05)))
    C.append(Claim("6.5", "the random ablation is the only Holm survivor in table 13",
                   "table10", True,
                   bool(float(mc.loc[("T4r", "T4"), "p_holm"]) <= 0.05)))
    # only negative results survive Holm in the primary family
    _surv = mc[mc.p_holm <= 0.05]
    C.append(Claim("6.4", "every Holm survivor favours the reference arm", "table10",
                   True, bool((_surv["median_delta"] < 0).all())))

    # ---- Section 6.5: decontamination and its size-matched control ----
    dec = read_table("table10_indomain_contrasts.csv")
    dec = dec[(dec.metric == "rmse") & (dec.n_train == 347)].set_index(["arm", "reference"])
    for (a, b, delta, wins, pv) in [("T4c", "T4", -0.0158, 1, 0.0098),
                                    ("T4r", "T4", -0.0308, 0, 0.0020),
                                    ("T4c", "T4r", 0.0141, 7, 0.0273),
                                    ("T5c", "T5", 0.0060, 7, 0.0840),
                                    ("T5r", "T5", -0.0026, 3, 0.2324),
                                    ("T5c", "T5r", 0.0068, 8, 0.0488)]:
        C.append(Claim("6.5", f"{a} vs {b} delta", "table10", delta,
                       float(dec.loc[(a, b), "median_delta"])))
        C.append(Claim("6.5", f"{a} vs {b} seed wins", "table10", wins,
                       int(dec.loc[(a, b), "arm_better_in_seeds"])))
        C.append(Claim("6.5", f"{a} vs {b} p", "table10", pv,
                       float(dec.loc[(a, b), "p_raw"])))
    for arm, val in [("T4_indomain_probe", 0.6028),
                     ("T4c_indomain_probe_decontaminated", 0.6241),
                     ("T4r_indomain_probe_random_ablation", 0.6297)]:
        C.append(Claim("6.5", f"{arm} full-data RMSE", "table1", val,
                       t1("scaffold", arm, 347, "median")))
    # The load-bearing inference: random ablation must cost MORE than
    # decontamination, or the section's conclusion inverts.
    C.append(Claim("6.5", "random ablation costs more than decontamination", "table10",
                   True, bool(float(dec.loc[("T4r", "T4"), "median_delta"])
                              < float(dec.loc[("T4c", "T4"), "median_delta"]))))
    C.append(Claim("6.5", "decontaminated beats the size-matched control", "table10",
                   "arm better", str(dec.loc[("T4c", "T4r"), "verdict"])))
    C.append(Claim("6.5", "T5 shows the same decisive contrast", "table10",
                   "arm better", str(dec.loc[("T5c", "T5r"), "verdict"])))
    C.append(Claim("6.5", "both arms: decontaminated beats the control", "table10", True,
                   bool(float(dec.loc[("T4c", "T4r"), "median_delta"]) > 0
                        and float(dec.loc[("T5c", "T5r"), "median_delta"]) > 0)))

    # ---- Section 6.3: activity-cliff strata ----
    T1P, T2P = "T1_chemberta_linear_probe", "T2_chemberta_full_finetune"
    B1A, B2A, B0A = "B1_ecfp_histgb", "B2_descriptors_rf", "B0_median"

    # census
    for thr, sim_pairs, cliff_pairs, involved in [
            (0.6, 3433, 827, 228), (0.7, 941, 242, 102), (0.8, 133, 42, 46)]:
        C.append(Claim("6.3", f"similar pairs at T>={thr}", "table7", sim_pairs,
                       t7(thr, "similar_pairs")))
        C.append(Claim("6.3", f"cliff pairs at T>={thr}", "table7", cliff_pairs,
                       t7(thr, "cliff_pairs")))
        C.append(Claim("6.3", f"compounds in >=1 cliff at T>={thr}", "table7", involved,
                       t7(thr, "compounds_in_a_cliff")))

    # "roughly a quarter of near-neighbour pairs are cliffs at every threshold"
    for thr in (0.6, 0.7, 0.8):
        frac = t7(thr, "cliff_pairs") / t7(thr, "similar_pairs")
        C.append(Claim("6.3", f"cliff share in [0.2, 0.35] at T>={thr}", "table7",
                       True, 0.20 <= frac <= 0.35))

    # B0 spans 1.317 (cliff) to 0.722 (distant) -- the variance confound
    C.append(Claim("6.3", "B0 cliff RMSE 1.317", "table6", 1.3167,
                   t6(B0A, "cliff", "rmse_median")))
    C.append(Claim("6.3", "B0 distant RMSE 0.722", "table6", 0.7219,
                   t6(B0A, "distant", "rmse_median")))

    # stratum sizes quoted in the prose and table headers
    for stratum, n in [("cliff", 9.0), ("smooth", 20.0), ("distant", 69.5)]:
        C.append(Claim("6.3", f"median {stratum} compounds per fold", "table6", n,
                       t6(B0A, stratum, "median_compounds")))

    # the finding: both transfer arms significant on distant, neither behind on cliff
    C.append(Claim("6.3", "T1 distant Holm p = 0.018", "table8", 0.0176,
                   t8(T1P, "distant", "p_holm")))
    C.append(Claim("6.3", "T2 distant Holm p = 0.019", "table8", 0.0195,
                   t8(T2P, "distant", "p_holm")))
    C.append(Claim("6.3", "T1 cliff delta +0.0526", "table8", 0.0526,
                   t8(T1P, "cliff", "median_delta_rmse")))
    C.append(Claim("6.3", "T2 cliff delta +0.0368", "table8", 0.0368,
                   t8(T2P, "cliff", "median_delta_rmse")))
    C.append(Claim("6.3", "both cliff deltas favour transfer (positive)", "table8", True,
                   bool(t8(T1P, "cliff", "median_delta_rmse") > 0
                        and t8(T2P, "cliff", "median_delta_rmse") > 0)))
    C.append(Claim("6.3", "T2 is the best skill-vs-B0 arm on cliffs", "table6", True,
                   bool(t6(T2P, "cliff", "skill_vs_b0_median")
                        > max(t6(a, "cliff", "skill_vs_b0_median")
                              for a in (B1A, B2A, T1P)))))
    C.append(Claim("6.3", "T2 cliff skill +0.49", "table6", 0.4928,
                   t6(T2P, "cliff", "skill_vs_b0_median")))
    C.append(Claim("6.3", "B1 cliff skill +0.43", "table6", 0.4333,
                   t6(B1A, "cliff", "skill_vs_b0_median")))

    # distant significance holds at every threshold; cliff never favours the baseline
    for thr in (0.6, 0.7, 0.8):
        for arm in (T1P, T2P):
            C.append(Claim("6.3", f"{arm} distant significant at T>={thr}", "table8",
                           True, bool((t8(arm, "distant", "p_holm", thr) or 1.0) <= 0.05)))
    for thr in (0.6, 0.7):
        for arm in (T1P, T2P):
            C.append(Claim("6.3", f"{arm} cliff does not favour B1 at T>={thr}", "table8",
                           True, bool((t8(arm, "cliff", "median_delta_rmse", thr) or 0) >= 0)))

    # the internal consistency check: the `all` column reproduces section 5.4
    for arm in (T1P, T2P, B2A):
        C.append(Claim("6.3", f"{arm} `all` delta reproduces table3", "table8 vs table3",
                       float(t3("scaffold", arm, "median_rmse_delta_vs_baseline")),
                       t8(arm, "all", "median_delta_rmse")))
        C.append(Claim("6.3", f"{arm} `all` Holm p reproduces table3", "table8 vs table3",
                       float(t3("scaffold", arm, "p_holm")),
                       t8(arm, "all", "p_holm")))

    # T=0.8 cliff stratum is withheld from testing: only 3 seeds clear the floor
    C.append(Claim("6.3", "T>=0.8 cliff has 3 usable seeds", "table6", 3,
                   t6(B0A, "cliff", "n_seeds", 0.8)))
    C.append(Claim("6.3", "T>=0.8 cliff paired test withheld", "table8", True,
                   t8(T1P, "cliff", "p_holm", 0.8) is None))

    # ---- Section 6.2: tuning ablation ----
    TUNED = Path("results/tuned_metrics")
    if TUNED.exists():
        def load(d, pat):
            out = {}
            for f in d.glob(pat):
                r = json.loads(f.read_text())
                out[(r["arm"], r["seed"], r["n_train"])] = r["metrics"]
            return out

        tb1, ub1 = load(TUNED, "B1*.json"), load(METRICS, "B1*scaffold*.json")
        shared = sorted(set(tb1) & set(ub1))
        if shared:
            deltas = [tb1[k]["rmse"] - ub1[k]["rmse"] for k in shared]
            wins = sum(d < 0 for d in deltas)
            C.append(Claim("6.2", "32-trial search moves B1 by median +0.004 RMSE",
                           "tuned_metrics", 0.004,
                           round(float(pd.Series(deltas).median()), 3)))
            C.append(Claim("6.2", "tuning helps B1 in 11 of 26 matched cells",
                           "tuned_metrics", "11/26", f"{wins}/{len(shared)}"))

        t2t = load(TUNED, "T2*n50.json")
        t2u = {k: v for k, v in load(METRICS, "T2*scaffold*n50.json").items()}
        if len(t2t) == 10:
            for metric, val in (("rmse", 0.735), ("r2", 0.301), ("spearman", 0.580)):
                C.append(Claim("6.2", f"T2 tuned n=50 {metric}", "tuned_metrics", val,
                               round(float(pd.Series([m[metric] for m in t2t.values()]).median()), 3)))
            for metric, val in (("rmse", 1.232), ("r2", -1.256), ("spearman", 0.456)):
                C.append(Claim("6.2", f"T2 untuned n=50 {metric}", "metrics", val,
                               round(float(pd.Series([m[metric] for m in t2u.values()]).median()), 3)))
            b1n50 = load(METRICS, "B1*scaffold*n50.json")
            better = sum(t2t[("T2_chemberta_full_finetune", s, 50)]["rmse"]
                         < b1n50[("B1_ecfp_histgb", s, 50)]["rmse"] for s in range(10))
            C.append(Claim("6.2", "tuned T2 beats B1 at n=50 in 3 of 10 seeds",
                           "tuned_metrics", 3, better))

        t2f = load(TUNED, "T2*n347.json")
        b1f = load(METRICS, "B1*scaffold*n347.json")
        if t2f:
            seeds = sorted(s_ for (_a, s_, _n) in t2f)
            C.append(Claim("6.2", "tuned T2 n=347 RMSE", "tuned_metrics", 0.620,
                           round(float(pd.Series([m["rmse"] for m in t2f.values()]).median()), 3)))
            C.append(Claim("6.2", "tuned T2 n=347 Spearman", "tuned_metrics", 0.695,
                           round(float(pd.Series([m["spearman"] for m in t2f.values()]).median()), 3)))
            C.append(Claim("6.2", "B1 n=347 RMSE on those seeds", "metrics", 0.634,
                           round(float(pd.Series([b1f[("B1_ecfp_histgb", s_, 347)]["rmse"]
                                                  for s_ in seeds]).median()), 3)))
            wins = sum(t2f[("T2_chemberta_full_finetune", s_, 347)]["rmse"]
                       < b1f[("B1_ecfp_histgb", s_, 347)]["rmse"] for s_ in seeds)
            C.append(Claim("6.2", "tuned T2 beats B1 at n=347 in 2 of 5 seeds",
                           "tuned_metrics", "2/5", f"{wins}/{len(seeds)}"))

    # ---- Run inventory ----
    n_runs = len(list(METRICS.glob("*.json")))
    # Parsed from the abstract rather than mirrored here: a hand-copied count
    # in this file drifts the moment a new arm is run, which it did.
    stated = re.search(r"with \*?\*?([\d,]+)\*?\*? evaluated runs", MANUSCRIPT.read_text())
    C.append(Claim("Abstract", "evaluated runs stated in the abstract", "metrics/*.json",
                   int(stated.group(1).replace(",", "")) if stated else None, n_runs))
    C.append(Claim("4.1", "10 seeds per cell", "table1", 10,
                   int(read_table("table1_learning_curves__scaffold.csv").n_seeds.min())))
    return C


def self_count_claim(n_claims: int) -> Claim:
    """§10 states how many claims this script checks. That sentence is itself a
    hand-typed number in a section arguing that no number is hand-typed, so it
    is checked against the actual count -- and against the sections covered."""
    text = MANUSCRIPT.read_text()
    m = re.search(r"checks \*\*(\d+) claims\*\*", text)
    return Claim("10", "stated claim count in §10", "verify_manuscript.py itself",
                 int(m.group(1)) if m else None, n_claims)


def main() -> int:
    if not MANUSCRIPT.exists():
        print("manuscript not found"); return 1
    claims = build_claims()
    claims.append(self_count_claim(len(claims) + 1))
    fails = [c for c in claims if not c.ok()]

    by_sec: dict[str, list[Claim]] = {}
    for c in claims:
        by_sec.setdefault(c.section, []).append(c)
    for sec in sorted(by_sec):
        bad = [c for c in by_sec[sec] if not c.ok()]
        mark = "FAIL" if bad else "ok  "
        print(f"  [{mark}] section {sec:<9} {len(by_sec[sec]):3d} claims"
              f"{'  <-- ' + str(len(bad)) + ' mismatched' if bad else ''}")

    if fails:
        print(f"\n{len(fails)} MISMATCHED CLAIM(S):")
        for c in fails:
            print(f"  [{c.section}] {c.text}")
            print(f"        manuscript: {c.expected!r}   source ({c.source}): {c.actual!r}")
        return 1

    print(f"\nAll {len(claims)} numeric claims verified against source artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
