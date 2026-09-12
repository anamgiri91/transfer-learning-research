# Seed extension plan — fixed before running

Status: **declared, not executed.** Written 2026-09-11, after the fourth audit
and before any seed above 9 exists. Nothing here may be revised once the runs
start; if it is, the revision gets its own dated entry in
[`decision-log.md`](decision-log.md) and the original stays visible.

The point of writing this down first is narrow and specific. The study is
powered only for large effects (§7 limitation 6), several of the comparisons
that matter are inconclusive rather than null, and adding seeds after seeing
which way they lean is seed-shopping. A fixed count, fixed arms and a
commitment to report the outcome either way is what separates an extension from
a fishing expedition.

## 1. What is fixed

| | |
|---|---|
| **Additional seeds** | **20**, ids `10`–`29`, giving 30 total |
| **Arms extended** | `B0`, `B1`, `B2`, `T0r`, `T1`, `T4`, `T5` on all three splits; `T4c`, `T4r`, `T5c`, `T5r` on scaffold |
| **Arms NOT extended** | `T2` (ChemBERTa fine-tune) — stays at seeds 0–9 |
| **Sizes** | unchanged: 50, 100, 250, 347 |
| **Splits** | unchanged: scaffold (primary), random, Butina |
| **Analyses re-run** | tables 1–4, 10, 11, 14, 15, 16 and every paired test in §5.4, §5.7, §6.4, §6.5, §5.3 |
| **Stopping rule** | none. All 20 seeds run; no interim look decides whether to continue |
| **Reporting rule** | every affected number is reported at 30 seeds **whether or not significance changes**, with the 10-seed value retained beside it |

The pre-registered 10-seed results are **not replaced.** `plan.md` §3.3 fixes
10 seeds, so the 10-seed analysis remains the pre-registered one and the 30-seed
analysis is reported alongside it as a declared extension. Where they disagree,
both are shown and the disagreement is the finding.

## 2. What 20 more seeds can and cannot buy

A seed here controls three things at once, because `run_arms.py` derives all of
them from it: the **split draw** (which compounds land in test), the **training
subsample** at each size below the full fold, and **model initialisation and
fitting order**. Extending seeds therefore tightens the estimate of variance
from those three sources jointly — it cannot separate them, and this plan does
not claim to.

What it does **not** buy, stated because the temptation to imply otherwise is
real:

- **Not more data.** There are still 494 compounds and 272 scaffolds. Every
  test fold is 98 compounds drawn from the same set, and across 30 seeds the
  folds overlap heavily — a given compound sits in roughly six of them. Seeds
  are not independent replicates in the way 30 fresh datasets would be, and the
  paired Wilcoxon's independence assumption is approximate for the same reason
  at 10 seeds as at 30.
- **Not external validity.** Nothing here speaks to another target, another
  assay, or another encoder. §7 limitation 1 is untouched by seed count.
- **Not tuning variance.** The tuned comparison of §6.2 draws its own
  hyperparameters per cell; extending the untuned sweep says nothing about it.
  §6.2 stays at its current coverage.

Bouthillier et al. (MLSys 2021) is the reference for why a benchmark's
uncertainty has several sources and why quoting one seed budget as though it
addressed all of them overstates what was measured. This extension addresses
split draw, subsample draw and initialisation, and says so.

## 3. `T2` stays at 10 seeds, and what that costs

`T2` is the only arm that is not cheap: 709.8 s per seed against 29.7 s for the
next most expensive, because each fine-tune fit is ~185 s. Extending it to 30
seeds would take ~3.9 h against ~13 min for everything else, and it is the arm
whose schedule §6.2 already shows to be misconfigured — spending four hours
re-measuring an artefact is the wrong purchase.

The consequence has to be handled rather than hidden:

- Every comparison involving `T2` is computed on the **10 shared seeds only**,
  keeping it paired. It is never compared against another arm's 30-seed
  distribution.
- Tables carry a per-arm seed count, and any table mixing 10- and 30-seed arms
  is marked as doing so. `make_report.py --require-seeds` already refuses to
  publish an arm below the stated count; that guard stays.
- `T2` carries the one Holm-significant positive result in the paper (§5.7
  enrichment). Its replication is therefore **not** improved by this extension.
  That remains open, and the honest way to close it is §5 below, not more
  seeds on the arms that were already cheap.

## 4. Cost, and the basis for the estimate

An earlier estimate of "~15 minutes" was quoted without stating what it
covered. The measured basis:

| Component | Measured | In the `seconds` field? |
|---|---|---|
| Model fitting, 11 extended arms, per seed | 39.4 s | yes |
| × 20 seeds | **13.1 min** | |
| Featurisation (ECFP 0.07 s, RDKit descriptors 2.7 s, ChemBERTa 4.3 s, in-domain 1.6 s) | ~9 s once per process | **no** — cached per process, and independent of seed |
| Building 60 additional split files | ~10 s | no |
| Re-running every analysis script | ~60 s | no |
| **Total** | **~15 min** | |

The `seconds` field measures fit time inside the seed loop and nothing else. It
is the right basis *for this extension* — the encoders are reused unchanged
across seeds, and features are computed once per process — and it is **not** a
basis for the cost of the study as a whole. That total is ~5.3 CPU-hours: 125
min of evaluation runs, ~86 min pretraining six in-domain encoders (796–934 s
each), 101 min of hyperparameter search.

## 5. What this extension is expected to move, and what it is not

Declared in advance so that a null result is legible as one:

| Comparison | At 10 seeds | Why more seeds might matter |
|---|---|---|
| H3: `T5` vs `T1` (§6.4) | raw p = 0.006; BH yes, Holm no | effect is 0.025 RMSE; the margin between BH and Holm here is a power question |
| H2: `T1` interaction slope (§5.3) | +0.007, CI [−0.017, +0.019] | a real crossover would sit inside that interval; 30 seeds narrows it by ~√3 |
| H4: `T4c` vs `T4r` (§6.5) | raw 0.027, Holm 0.273 | the decisive contrast in the decontamination reading |
| `T2` vs `B1`, RMSE and enrichment | Holm 0.074 / 0.023 | **not extended** — see §3 |

**Expected not to move:** `B1` vs `B2` (p = 0.56, and the effect is 0.012
RMSE), and H2 for the fine-tune, which is confounded by schedule rather than by
noise.

An extension that leaves every verdict where it was is a publishable outcome
and will be reported as one. The failure mode this document exists to prevent
is the other one: running 20 seeds, finding H3 still short of Holm, and
quietly not mentioning it.

## 6. What would actually close the open questions

More seeds is the cheap intervention, not the decisive one. In rough order of
value per hour:

1. **`T2` on random and Butina** (~2.9 h) — tests whether the paper's one
   positive claim replicates across splits. Nothing in this plan substitutes
   for it.
2. **An in-domain fine-tune** — the arm `plan.md` §4 specifies as `T4` and that
   was never built (Amendment 2). Until it exists, H3's pre-registered form is
   unperformed rather than inconclusive.
3. **`B3`, a D-MPNN from scratch** — needs a `graph` representation, which
   `evapro.features` does not have. Without it the study cannot separate
   "pretraining does not help here" from "deep models do not work at n = 347
   here".
4. **A chemically in-domain corpus** — ours is 95% coronaviral with median
   nearest-neighbour Tanimoto 0.247 to the evaluation set (§6.4).

---

## Amendment A — 2026-09-12, deprioritised, and why

**The extension specified above is not executed in this iteration, and its
priority has dropped.** The plan itself stands unchanged; this amendment
records what the fourth audit and Amendment 4 changed about its value. Written
before any extension seed was run — no seed above 9 exists.

### What changed

Three findings arrived after the plan was written:

1. **H3's pre-registered comparison was never performed** (`plan.md`
   Amendment 2). The in-domain arms are frozen probes; the comparison the
   protocol names needs an in-domain *fine-tune*, which did not exist. Seeds
   cannot supply a missing arm.
2. **H2's answer changed** (`manuscript.md` §5.3). With the pre-registered
   interaction term finally run, the frozen probe is inconclusive rather than
   negative, and the one significant slope belongs to `T2` — measured under a
   training schedule §6.2 shows to be a harness artefact. Seeds cannot repair a
   confounded schedule.
3. **The DER's zeros were censoring, not measurements.** No DER interval on any
   split excludes 1. Precision on a statistic that is censored on 6–9 of 10
   seeds is not what the study lacks.

### The consequence for what more seeds would buy

The extension's own §5 listed four comparisons it might move. Re-reading that
list against the three findings:

| Comparison | Seeds help? | Why |
|---|---|---|
| H3: `T5` vs `T1` | **No longer the right question** | It is a probe-vs-probe contrast. The pre-registered contrast is `T4ft` vs `T2v` (Amendment 4), which needs the arms, not the seeds |
| H2: `T1` interaction slope | **Yes, narrowly** | CI [−0.017, +0.019] would tighten by ~√3. This is the one place the extension still earns its cost |
| H4: `T4c` vs `T4r` | **Yes** | raw 0.027 / Holm 0.273 is a power problem, and the arms exist |
| `T2` vs `B1` | **No** | `T2` is excluded from the extension, and Amendment 4 supersedes it with `T2v` |

Two of four have been overtaken by the missing-arm and confounded-schedule
problems, which are experiment-design faults rather than sampling noise. **More
seeds for cheap arms cannot repair either.** That is the reason for
deprioritising, and it is the reason the ~7 h of local compute this iteration
had available went to Amendment 4's fine-tuning arms instead.

### What is unchanged

Everything in §1–§5 above. If the extension is run later it runs **as specified
there**: 20 seeds, ids 10–29, the same eleven arms, `T2` excluded, no stopping
rule, every affected number reported at 30 seeds beside its 10-seed value
whether or not significance changes, and the pre-registered 10-seed analysis
retained as the pre-registered one.

Two conditions are added, and neither loosens anything:

- **Order.** It runs *after* Amendment 4's arms are complete and analysed, so
  that the seed budget is spent on comparisons that are correctly specified
  rather than on tightening intervals around a contrast that turns out to be
  the wrong one.
- **Scope may narrow, never widen.** If Amendment 4 resolves H3, the extension
  may be cut to the arms bearing on H2 and H4. It may not gain arms, seeds
  beyond id 29, or a stopping rule, and no decision about running it may be
  taken on the basis of a p-value from a partial run.
