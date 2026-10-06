#!/usr/bin/env python
"""How much the reported conclusions depend on the arithmetic — plan.md Amendment 8.

Two retraining cells stopped matching their saved outputs on 2026-10-04.
Neither training script fixes the CPU thread count, so parallel floating-point
reductions sum in a different order from one configuration to the next. For
`T2v` that stays in the last float32 digit; for `B3` early stopping amplifies
it into a different fit. Amendment 8 fixes two analyses before either is run:

  A  re-fit all 40 `B3` cells on one thread and recompute Amendment 5's five
     contrasts, unchanged, against the same saved `B1` and `T2v` cells
  B  bound precision@10% over every order in which predictions tied at the
     top-decile cutoff (within 1e-5 pKD) could be broken, and recompute every
     reported precision@10% contrast at both extremes

The saved outputs stay the primary results. Nothing here overwrites them, and
nothing here restores bitwise regeneration of the saved files.

Writes docs/numerical-sensitivity.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from evapro.evaluation.metrics import precision_at_k_frac, precision_at_k_frac_bounds

sys.path.insert(0, "scripts")
from analyse_b3 import B3_FAMILY, holm, paired_ci  # noqa: E402

OUT = Path("docs/numerical-sensitivity.json")
PINNED = Path("results/sensitivity/b3_threads1")
SEEDS = list(range(10))
SIZES = [50, 100, 250, 347]
EPS = 1e-5          # fixed in Amendment 8
ALPHA = 0.05
METRIC = "precision_at_10pct"

# §5.7's scaffold family, as scripts/analyse_endpoints.py tests it.
SCAFFOLD_FAMILY = ["B2_descriptors_rf", "T0r_untrained_encoder_probe",
                   "T1_chemberta_linear_probe", "T2_chemberta_full_finetune",
                   "T4_indomain_probe", "T5_chained_probe"]
# Amendment 6's five, as scripts/analyse_enrichment.py tests them.
AMENDMENT6 = [("T2", "random", "results/predictions/T2_chemberta_full_finetune"),
              ("T2", "butina", "results/predictions/T2_chemberta_full_finetune"),
              ("T2v", "scaffold", "results/predictions_ft/T2v"),
              ("T4ft", "scaffold", "results/predictions_ft/T4ft"),
              ("T5ft", "scaffold", "results/predictions_ft/T5ft")]
B1 = "B1_ecfp_histgb"


def wilcoxon_p(x, y) -> float:
    return 1.0 if not np.any(x != y) else float(stats.wilcoxon(x, y).pvalue)


# ---- A: thread-pinned B3 ---------------------------------------------------
def rmse_cells(pattern: str) -> dict:
    out = {}
    for f in glob.glob(pattern):
        if f.endswith(".FAILED.json"):
            continue
        d = json.loads(Path(f).read_text())
        key = (d["seed"], d["n_train"])
        if key in out or not np.isfinite(d["metrics"]["rmse"]):
            raise ValueError(f"duplicate cell or nonfinite RMSE: {f}")
        out[key] = d["metrics"]["rmse"]
    return out


def b3_contrasts(b3: dict, refs: dict) -> list[dict]:
    rows = []
    for arm, ref, n in B3_FAMILY:
        a = np.array([b3[(s, n)] for s in SEEDS])
        b = np.array([refs[ref][(s, n)] for s in SEEDS])
        d = a - b                                   # positive => B3 worse
        lo, hi = paired_ci(d)
        rows.append(dict(arm=arm, reference=ref, n_train=n,
                         median_paired_delta=float(np.median(d)),
                         paired_ci_lo=lo, paired_ci_hi=hi,
                         arm_worse_in_seeds=int((d > 0).sum()),
                         p_raw=wilcoxon_p(a, b)))
    for r, p in zip(rows, holm([r["p_raw"] for r in rows])):
        r["p_holm"] = float(p)
        r["verdict"] = ("no detectable difference" if p > ALPHA else
                        "arm worse" if r["median_paired_delta"] > 0 else "arm better")
    return rows


def analysis_a() -> dict:
    saved = rmse_cells("results/metrics_b3/B3__scaffold__*.json")
    pinned = rmse_cells(str(PINNED / "metrics/B3__scaffold__*.json"))
    cells = [(s, n) for s in SEEDS for n in SIZES]
    missing = [c for c in cells if c not in pinned]
    if missing:
        raise ValueError(f"pinned B3 incomplete: missing {missing}")
    if set(pinned) != set(cells) or set(saved) != set(cells):
        raise ValueError("B3 must have exactly the 40 specified cells")
    if list((PINNED / "metrics").glob("*.FAILED.json")):
        raise ValueError("pinned B3 contains failed cells")
    for s, n in cells:
        tag = f"B3__scaffold__seed{s}__n{n}"
        rec = json.loads((PINNED / "metrics" / f"{tag}.json").read_text())
        original = json.loads(Path(f"results/metrics_b3/{tag}.json").read_text())
        if rec.get("torch_threads") != 1:
            raise ValueError(f"{tag}: pinned record must specify one thread")
        for key in ("arm", "target", "split", "seed", "n_train", "n_test",
                    "n_train_fitted", "n_internal_val", "lr_grid", "max_epochs",
                    "patience", "batch_size", "internal_val_frac", "implementation"):
            if rec[key] != original[key]:
                raise ValueError(f"{tag}: changed experiment identity/recipe: {key}")
        with np.load(PINNED / "predictions" / f"{tag}.npz", allow_pickle=True) as z, \
                np.load(f"results/predictions_b3/{tag}.npz", allow_pickle=True) as old:
            if (set(z.files) != set(old.files)
                    or not np.array_equal(z["inchikey"], old["inchikey"])
                    or not np.array_equal(z["y_true"], old["y_true"])
                    or z["y_pred"].shape != old["y_pred"].shape
                    or not np.isfinite(z["y_pred"]).all()):
                raise ValueError(f"{tag}: prediction identity/shape/finite check failed")
            rmse = float(np.sqrt(np.mean((z["y_true"] - z["y_pred"]) ** 2)))
            if rmse != pinned[(s, n)]:
                raise ValueError(f"{tag}: stored RMSE disagrees with predictions")
    refs = {"B1": rmse_cells(f"results/metrics/{B1}__scaffold__*.json"),
            "T2v": rmse_cells("results/metrics_ft/T2v__scaffold__*.json")}
    rs, rp = b3_contrasts(saved, refs), b3_contrasts(pinned, refs)
    contrasts = []
    for a, b in zip(rs, rp):
        same_sign = np.sign(a["median_paired_delta"]) == np.sign(b["median_paired_delta"])
        contrasts.append(dict(
            arm=a["arm"], reference=a["reference"], n_train=a["n_train"],
            saved={k: a[k] for k in a if k not in ("arm", "reference", "n_train")},
            pinned={k: b[k] for k in b if k not in ("arm", "reference", "n_train")},
            numerically_robust=bool(same_sign and a["verdict"] == b["verdict"])))
    shift = {}
    for n in SIZES:
        d = np.array([pinned[(s, n)] - saved[(s, n)] for s in SEEDS])
        shift[str(n)] = dict(
            saved_median_rmse=float(np.median([saved[(s, n)] for s in SEEDS])),
            pinned_median_rmse=float(np.median([pinned[(s, n)] for s in SEEDS])),
            median_abs_cell_shift=float(np.median(np.abs(d))),
            max_abs_cell_shift=float(np.abs(d).max()),
            median_signed_cell_shift=float(np.median(d)))
    return {"status": "complete", "pinned_cells": len(pinned),
            "torch_threads": 1, "pinned_root": str(PINNED),
            "family": "plan.md Amendment 5, Holm m=5, unchanged",
            "contrasts": contrasts, "rmse_shift_by_n_train": shift,
            "all_numerically_robust": all(c["numerically_robust"] for c in contrasts)}


# ---- B: ties at the top-decile cutoff --------------------------------------
def bounds(path: str) -> tuple[float, float, float]:
    z = np.load(path, allow_pickle=True)
    y, yhat = z["y_true"], z["y_pred"]
    lo, hi = precision_at_k_frac_bounds(y, yhat, eps=EPS)
    point = precision_at_k_frac(y, yhat)
    assert lo <= point <= hi, path
    return point, lo, hi


def tied_spearman_shift(path: str) -> float | None:
    z = np.load(path, allow_pickle=True)
    y, yhat = z["y_true"], np.asarray(z["y_pred"], float)
    if np.ptp(yhat) == 0:
        return None                      # constant predictor: undefined
    order = np.argsort(yhat)
    merged = yhat.copy()
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or yhat[order[i]] - yhat[order[i - 1]] > EPS:
            merged[order[start:i]] = yhat[order[start:i]].mean()
            start = i
    return float(abs(stats.spearmanr(y, merged).statistic
                     - stats.spearmanr(y, yhat).statistic))


def scenario_tests(arms: list[tuple[str, dict, dict]]) -> list[dict]:
    """arms: (label, {seed: (point, lo, hi)} for the arm, same for B1)."""
    out = []
    for name, pick_a, pick_b in (("saved", 0, 0), ("against_arm", 1, 2),
                                 ("for_arm", 2, 1)):
        ps, deltas = [], []
        for _label, a, b in arms:
            x = np.array([a[s][pick_a] for s in SEEDS])
            y = np.array([b[s][pick_b] for s in SEEDS])
            ps.append(wilcoxon_p(x, y))
            deltas.append(float(np.median(x - y)))   # positive => arm better
        for (label, _a, _b), p, ph, d in zip(arms, ps, holm(ps), deltas):
            verdict = ("no detectable difference" if ph > ALPHA else
                       "arm better" if d > 0 else "arm worse")
            out.append(dict(test=label, scenario=name, median_paired_delta=d,
                            p_raw=p, p_holm=float(ph), verdict=verdict))
    return out


def summarise(tests: list[dict]) -> list[dict]:
    rows = []
    for label in dict.fromkeys(t["test"] for t in tests):
        by = {t["scenario"]: t for t in tests if t["test"] == label}
        rows.append(dict(
            test=label,
            **{f"{k}_{s}": by[s][k] for s in by
               for k in ("median_paired_delta", "p_raw", "p_holm", "verdict")},
            tie_robust=len({t["verdict"] for t in by.values()}) == 1))
    return rows


def exhaustive_bounds(arms: list[tuple[str, dict, dict]], k_by_seed: dict) -> list[dict]:
    """Enumerate every hit-count assignment, then conservatively bound Holm.

    Keep float subtraction and SciPy defaults identical to the historical test.
    Holm is coordinatewise monotone in raw p-values. Its marginal bound vectors
    may be unattainable jointly when tests share B1; that makes the bounds wider,
    never falsely reassuring. This is additional to Amendment 8's two scenarios.
    """
    out = []
    for label, a, b in arms:
        choices = []
        for s in SEEDS:
            k = k_by_seed[s]
            av = [v / k for v in range(round(a[s][1] * k), round(a[s][2] * k) + 1)]
            bv = [v / k for v in range(round(b[s][1] * k), round(b[s][2] * k) + 1)]
            choices.append(list(itertools.product(av, bv)))
        ps, medians = [], []
        for assignment in itertools.product(*choices):
            x, y = np.asarray(assignment).T
            ps.append(wilcoxon_p(x, y))
            medians.append(float(np.median(x - y)))
        out.append(dict(test=label, configurations=len(ps),
                        p_raw_min=min(ps), p_raw_max=max(ps),
                        median_paired_delta_min=min(medians),
                        median_paired_delta_max=max(medians)))
    lower = holm([r["p_raw_min"] for r in out])
    upper = holm([r["p_raw_max"] for r in out])
    saved = {r["test"]: r for r in scenario_tests(arms) if r["scenario"] == "saved"}
    for r, lo, hi in zip(out, lower, upper):
        r.update(p_holm_lower_bound=float(lo), p_holm_upper_bound=float(hi))
        if lo > ALPHA:
            verdict = "no detectable difference"
        elif hi <= ALPHA and r["median_paired_delta_min"] > 0:
            verdict = "arm better"
        elif hi <= ALPHA and r["median_paired_delta_max"] < 0:
            verdict = "arm worse"
        else:
            verdict = "not certified over all configurations"
        r.update(bounded_verdict=verdict,
                 all_configurations_robust=verdict == saved[r["test"]]["verdict"])
    return out


def analysis_b() -> dict:
    files = sorted(glob.glob("results/predictions/*.npz")
                   + glob.glob("results/predictions_ft/*.npz")
                   + glob.glob("results/predictions_b3/*.npz"))
    per_arm, shifts = {}, []
    for f in files:
        arm = Path(f).name.split("__")[0]
        point, lo, hi = bounds(f)
        rec = per_arm.setdefault(arm, dict(files=0, tie_sensitive=0))
        rec["files"] += 1
        rec["tie_sensitive"] += int(hi > lo)
        s = tied_spearman_shift(f)
        if s is not None and np.isfinite(s):
            shifts.append(s)

    def cells(stem: str, split: str) -> dict:
        return {s: bounds(f"{stem}__{split}__seed{s}__n347.npz") for s in SEEDS}

    def fixed(split: str) -> dict:
        """B1 where no predictions were saved: the stored metric, unbounded."""
        out = {}
        for s in SEEDS:
            v = json.loads(Path(
                f"results/metrics/{B1}__{split}__seed{s}__n347.json"
            ).read_text())["metrics"][METRIC]
            out[s] = (v, v, v)
        return out

    b1_scaffold = cells(f"results/predictions/{B1}", "scaffold")
    scaffold_arms = [(a, cells(f"results/predictions/{a}", "scaffold"),
                      b1_scaffold) for a in SCAFFOLD_FAMILY]
    am6_arms = [
        (f"{arm} ({split})", cells(stem, split),
         b1_scaffold if split == "scaffold" else fixed(split))
        for arm, split, stem in AMENDMENT6]
    scaffold = scenario_tests(scaffold_arms)
    am6 = scenario_tests(am6_arms)
    # All reported contrasts have k=10, checked from every arm's actual rows.
    # Do not silently use this grid if future prediction bundles change size.
    for stem, split in ([(f"results/predictions/{a}", "scaffold")
                         for a in [B1] + SCAFFOLD_FAMILY]
                        + [(stem, split) for _, split, stem in AMENDMENT6]):
        for s in SEEDS:
            with np.load(f"{stem}__{split}__seed{s}__n347.npz", allow_pickle=True) as z:
                if max(1, round(len(z["y_true"]) * 0.1)) != 10:
                    raise ValueError("exhaustive hit-count grid requires k=10")
                if split == "scaffold":
                    with np.load(f"results/predictions/{B1}__{split}__seed{s}__n347.npz",
                                 allow_pickle=True) as ref:
                        if (not np.array_equal(z["inchikey"], ref["inchikey"])
                                or not np.array_equal(z["y_true"], ref["y_true"])):
                            raise ValueError("paired prediction identities differ")
    exhaustive = {
        "scaffold_family_5_7": exhaustive_bounds(scaffold_arms, dict.fromkeys(SEEDS, 10)),
        "amendment_6": exhaustive_bounds(am6_arms, dict.fromkeys(SEEDS, 10)),
    }

    fresh = json.loads(Path(
        "docs/publication-validation/T2v__scaffold__seed0__n50.fresh.json").read_text())
    point, lo, hi = bounds("results/predictions_ft/T2v__scaffold__seed0__n50.npz")
    s57, a6 = summarise(scaffold), summarise(am6)
    return {
        "eps_pkd": EPS,
        "prediction_files": len(files),
        "tie_sensitive_files": int(sum(r["tie_sensitive"] for r in per_arm.values())),
        "tie_sensitive_by_arm": {k: v for k, v in sorted(per_arm.items())
                                 if v["tie_sensitive"]},
        "spearman_shift_with_tied_ranks": dict(
            files=len(shifts), median=float(np.median(shifts)),
            max=float(np.max(shifts))),
        "scaffold_family_5_7": dict(family="Holm m=6", tests=s57),
        "amendment_6": dict(
            family="Holm m=5", tests=a6,
            b1_bounded="scaffold only; B1 predictions were not saved on the "
                       "random and Butina splits, so B1 is held at its stored value"),
        "all_tie_robust": all(t["tie_robust"] for t in s57 + a6),
        "exhaustive_validation": dict(
            method="All per-seed attainable hit counts; conservative Holm bounds "
                   "from marginal raw-p extrema, with historical SciPy defaults. "
                   "B1 held fixed on random/Butina; not a bound on arbitrary retraining.",
            **exhaustive,
            all_configurations_robust=all(r["all_configurations_robust"]
                for family in exhaustive.values() for r in family)),
        "retrain_cell_T2v_seed0_n50": dict(
            saved=point, lower=lo, upper=hi,
            fresh=fresh["metrics"][METRIC],
            fresh_within_saved_bounds=bool(lo <= fresh["metrics"][METRIC] <= hi)),
    }


def build_report() -> dict:
    report = {"amendment": "plan.md Amendment 8 (2026-10-05)",
              "note": "Saved outputs remain the primary results and are not "
                      "overwritten. These analyses bound the dependence of the "
                      "reported conclusions on floating-point arithmetic; they do "
                      "not restore bitwise regeneration of the saved files.",
              "thread_pinned_b3": analysis_a(),
              "precision_tie_bounds": analysis_b()}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if evidence is missing or stale")
    args = ap.parse_args()
    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != encoded:
            print(f"FAIL: {OUT} missing or stale; run make sensitivity")
            return 1
        print(f"Verified {OUT} against prediction and metric artifacts")
    else:
        OUT.write_text(encoded)
        print(f"wrote {OUT}")
    a, b = report["thread_pinned_b3"], report["precision_tie_bounds"]
    if a["status"] != "complete":
        print(f"A: pinned B3 incomplete ({a['pinned_cells']}/40)")
    else:
        print("\n-- A: Amendment 5 contrasts, saved vs one thread --")
        for c in a["contrasts"]:
            print(f"  B3 vs {c['reference']:3s} n={c['n_train']:3d}  "
                  f"saved {c['saved']['median_paired_delta']:+.4f} "
                  f"Holm {c['saved']['p_holm']:.4f}   "
                  f"pinned {c['pinned']['median_paired_delta']:+.4f} "
                  f"Holm {c['pinned']['p_holm']:.4f}   "
                  f"{'robust' if c['numerically_robust'] else 'NOT ROBUST'}")
    print(f"\n-- B: {b['tie_sensitive_files']} of {b['prediction_files']} prediction "
          f"files are tie-sensitive at eps={EPS:g} --")
    for fam in ("scaffold_family_5_7", "amendment_6"):
        for t in b[fam]["tests"]:
            print(f"  {t['test']:32s} Holm saved {t['p_holm_saved']:.4f}  "
                  f"against {t['p_holm_against_arm']:.4f}  for {t['p_holm_for_arm']:.4f}  "
                  f"{'robust' if t['tie_robust'] else 'NOT ROBUST'}")
    print("\n-- Additional validation: all attainable hit-count assignments --")
    for fam in ("scaffold_family_5_7", "amendment_6"):
        for t in b["exhaustive_validation"][fam]:
            print(f"  {t['test']:32s} configurations={t['configurations']:2d}  "
                  f"Holm [{t['p_holm_lower_bound']:.6f}, "
                  f"{t['p_holm_upper_bound']:.6f}]  {t['bounded_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
