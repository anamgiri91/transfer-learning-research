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

    # ---- Run inventory ----
    n_runs = len(list(METRICS.glob("*.json")))
    C.append(Claim("Abstract", "520 evaluated runs", "metrics/*.json", 520, n_runs))
    C.append(Claim("4.1", "10 seeds per cell", "table1", 10,
                   int(read_table("table1_learning_curves__scaffold.csv").n_seeds.min())))
    return C


def main() -> int:
    if not MANUSCRIPT.exists():
        print("manuscript not found"); return 1
    claims = build_claims()
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
