"""Endpoint metrics with bootstrap CIs (plan.md §6)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman(y, yhat) -> float:
    return float(stats.spearmanr(y, yhat).statistic)


def pearson(y, yhat) -> float:
    return float(stats.pearsonr(y, yhat).statistic)


def precision_at_k_frac(y, yhat, frac: float = 0.10) -> float:
    """Fraction of the top-`frac` predicted that are truly in the top-`frac`.

    The enrichment view: what a screening campaign would actually experience.
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    k = max(1, int(round(len(y) * frac)))
    true_top = set(np.argsort(-y)[:k].tolist())
    pred_top = np.argsort(-yhat)[:k]
    return float(sum(i in true_top for i in pred_top.tolist()) / k)


METRICS = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "spearman": spearman,
    "pearson": pearson,
    "precision_at_10pct": precision_at_k_frac,
}


def compute_all(y, yhat, names: list[str] | None = None) -> dict[str, float]:
    return {n: METRICS[n](y, yhat) for n in (names or list(METRICS))}


def bootstrap_ci(y, yhat, metric: str = "rmse", n_resamples: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI over TEST-ROW resamples.

    NOT the interval reported anywhere in the paper, and kept only because a
    row-level interval is the right tool for a single fixed test fold. Every
    published CI resamples **seeds** instead -- the seed is the unit of
    replication, since it draws an independent split and fit, and resampling
    rows within one fold would understate split variance. That estimator lives
    in `scripts/make_report.py::_boot_median_ci`; this one has no callers
    outside tests, and reaching for it by mistake would silently answer a
    different question, so it says so here.
    """
    fn = METRICS[metric]
    y, yhat = np.asarray(y), np.asarray(yhat)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_resamples, len(y)))
    draws = np.array([fn(y[i], yhat[i]) for i in idx])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, yhat)), float(lo), float(hi)
