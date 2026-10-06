# Frozen-probe data efficiency: a prospective design proposal

> **Superseded 2026-10-05, before any of it was executed.** The design audit
> found that the estimand this proposal buys precision on — the per-seed slope
> of ΔRMSE on log₂(n) — has no specificity for transfer: `B0`, a constant
> predictor, attains the largest positive slope in the study, because a
> positive slope means only "flatter curve than `B1`" and the flattest curve
> belongs to the arm that learns least. Every learning arm is significantly
> *less* flat than that reference (manuscript §5.3,
> `results/tables/table15_h2_interaction.csv`). Collecting 150 fresh seeds
> would therefore sharpen a quantity that does not answer H2. The calculation
> below is retained as a variance-scale reference for whatever estimand
> replaces it, and as a worked example of planning before execution; it is no
> longer a recommended experiment. A replacement estimand must be specified
> and dated before any new seed is run.

Status: **planning only; no new model fits or seed extension executed.**
Written 2026-10-05 after the original results. This is a separate proposal,
not a change to the historical analysis or the capped 30-seed extension in
`seed-extension-plan.md`. A new experiment would require freezing and dating
this design before any new training results are examined.

## Specification fixed before the planning simulation

- Primary contrast: frozen `T1` versus `B1` on scaffold splits, at the same
  four labelled budgets 50, 100, 250 and 347, with test rows fixed within each
  learning curve and training subsets matched between arms.
- Estimand: per-seed OLS slope of `RMSE_T1 - RMSE_B1` on `log2(n)`. A positive
  slope means a smaller relative deficit at low data; it does not by itself
  establish absolute superiority or threshold crossing.
- Design alternative: +0.01 pKD RMSE per doubling. This is a proposed practical
  threshold, not a field-wide standard or a threshold inferred from the
  observed +0.007 median. Also show +0.005 and +0.02 and the zero-effect case.
- Planning model: independent Gaussian slopes; estimate their SD from the
  existing ten paired T1/B1 curves, and examine that SD and 1.5 times that SD.
  Use 10,000 simulated experiments per scenario and RNG seed 20261005.
- Candidate fresh seed budgets: 10, 20, 30, 50, 75, 100, 150, 200 and 300.
  Choose the smallest candidate whose lower 95% Monte Carlo Wilson bound for
  positive-slope detection reaches 80% at +0.01 and the inflated SD.
- Analysis: two-sided signed-rank test at 0.05, with positive paired location
  required to support H2. This future experiment has one primary contrast;
  historical seven-arm Holm results remain unchanged. Also show the more
  conservative 0.05/7 planning threshold so a larger family is not free.
- Execution if adopted: fresh seed IDs starting at 10; original seeds 0–9 are
  pilot evidence and reported separately. Run every planned seed without
  interim significance checks. Fix checkpoint revision, software, CPU threads,
  preprocessing, validation rules and numerical-sensitivity treatment in the
  new protocol. Save all splits, predictions, metrics and failed cells.
- Report median, mean and Hodges–Lehmann slopes separately with uncertainty,
  the paired differences at every budget, the test result and censored DER.
  A nonsignificant result remains inconclusive unless an independently specified
  equivalence analysis rules out a practically relevant slope.

## Scope

The simulation measures design performance **under its assumed model**. Its
Monte Carlo interval measures simulation error, not uncertainty in the pilot SD,
the Gaussian assumption or dependence caused by reusing compounds. More seeds
improve characterization of a fixed benchmark; they add no new compounds,
assays or targets. A population-level conclusion requires new, independently
collected chemical series or datasets and a new cluster-level power calculation.
No universal number of compounds can be inferred from these ten seed slopes.

## Conditional planning result

The pilot slope SD is 0.026737. At the +0.01 design effect, the inflated-SD
scenario gives 24.6% positive-slope detection at 30 fresh seeds and 84.3% at
150 (95% Monte Carlo interval 83.6–85.0%). The fixed grid rule selects **150
fresh seeds, IDs 10–159**, for the single primary contrast. This means 1,200
arm–seed–budget fits: two arms × 150 seeds × four budgets. No such fits have
been executed.

Using 0.05/7 instead gives only 59.2% detection at 150; the first candidate
meeting the same criterion is 300. These are separate prospective planning
scenarios, not alternative corrections selected for the existing results.
The full grid also shows sensitivity to the effect size and the uninflated
pilot SD. The 50% SD inflation is a stress scenario, not an upper confidence
bound on the unknown variance. Under the zero-effect simulations, two-sided
rejection rates are 4.4–5.3% at the nominal 0.05 threshold.

Rebuild the conditional calculation with:

```bash
.venv/bin/python scripts/plan_h2_power.py
```

The complete scenario grid and pilot-input hashes are in
[`h2-power-planning.json`](h2-power-planning.json). The signed-rank implementation
and assumptions follow the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
