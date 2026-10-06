#!/usr/bin/env python
"""Conditional design simulation for the unresolved frozen-probe H2 slope.

This is a post-result planning calculation, not observed power or new training.
The pilot supplies the standard deviation, never the alternative's effect size.
Assumed independent Gaussian seed slopes are a working model conditional on
this dataset; they do not represent independent chemical or target populations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyse_h2 import BASELINE, SIZES, load, per_seed_slopes

ARM = "T1_chemberta_linear_probe"
SEEDS = list(range(10))
BUDGETS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
EFFECTS = [0.0, 0.005, 0.01, 0.02]
SD_FACTORS = [1.0, 1.5]
SIMULATIONS = 10000
RNG_SEED = 20261005
OUT = Path("docs/h2-power-planning.json")


def wilson_interval(hits, total):
    p = hits / total
    z = float(stats.norm.ppf(0.975))
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - half), float(centre + half)]


def build_report():
    pilot = load("scaffold")
    for arm in (ARM, BASELINE):
        if not all(s in pilot[arm] and set(SIZES) <= pilot[arm][s].keys() for s in SEEDS):
            raise ValueError("all ten paired four-budget pilot curves are required")
    slopes = per_seed_slopes(pilot, ARM, SEEDS)
    sd = float(np.std(slopes, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("pilot slope standard deviation must be positive")
    rows = []
    for n in BUDGETS:
        # Identical standard-normal draws across alternatives reduce Monte Carlo
        # noise when comparing designs; each budget gets its own fixed stream.
        noise = np.random.default_rng(np.random.SeedSequence([RNG_SEED, n])).normal(
            size=(SIMULATIONS, n))
        for factor in SD_FACTORS:
            for effect in EFFECTS:
                values = effect + factor * sd * noise
                p = stats.wilcoxon(values, axis=1, method="auto").pvalue
                positive = np.median(values, axis=1) > 0
                for family in (1, 7):
                    rejected = p <= 0.05 / family
                    hits = int(np.count_nonzero(rejected & positive))
                    rows.append(dict(seeds=n, effect_per_doubling=effect,
                        sd_factor=factor, family_allowance=family,
                        threshold=0.05 / family,
                        h2_detection_probability=hits / SIMULATIONS,
                        monte_carlo_95_interval=wilson_interval(hits, SIMULATIONS),
                        two_sided_rejection_probability=float(np.mean(rejected))))
    # Selection rule declared before simulation: 80% lower Monte Carlo bound,
    # effect +0.01, pilot SD inflated by 50%, one prospective primary contrast.
    eligible = [r["seeds"] for r in rows if r["effect_per_doubling"] == 0.01
                and r["sd_factor"] == 1.5 and r["family_allowance"] == 1
                and r["monte_carlo_95_interval"][0] >= 0.8]
    paths = [Path(f"results/metrics/{a}__scaffold__seed{s}__n{n}.json")
             for a in (ARM, BASELINE) for s in SEEDS for n in SIZES]
    return dict(status="planning_only_not_executed", arm=ARM, reference=BASELINE,
        pilot_seeds=SEEDS, training_sizes=SIZES, pilot_slopes=slopes.tolist(),
        pilot_slope_sd=sd, simulations=SIMULATIONS, rng_seed=RNG_SEED,
        model="Independent Gaussian seed slopes with a fixed location shift; "
              "conditional on this dataset, not a population-power guarantee.",
        test="Two-sided scipy.stats.wilcoxon(method='auto'); positive median "
             "required for H2 detection. Family 7 uses conservative Bonferroni, not simulated Holm.",
        design_effect=0.01, design_sd_factor=1.5, design_family_allowance=1,
        selection_rule="Smallest candidate budget whose lower 95% Monte Carlo "
                       "Wilson bound for H2 detection reaches 0.80.",
        proposed_fresh_seed_count=min(eligible) if eligible else None,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scenarios=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != encoded:
            raise SystemExit("H2 planning report is missing or stale")
    else:
        OUT.write_text(encoded)
    print(f"{OUT}: pilot slope SD {report['pilot_slope_sd']:.6f}; "
          f"proposed fresh seeds {report['proposed_fresh_seed_count']}")


if __name__ == "__main__":
    main()
