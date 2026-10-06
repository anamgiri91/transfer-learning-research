"""Endpoint metrics with bootstrap CIs (plan.md §6)."""
from __future__ import annotations

import numpy as np
from math import comb
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


def _precision_bounds_given_true_top(yhat, true_top, k, eps):
    """Lowest and highest precision for one fixed true-top set, over pred ties."""
    cut = np.sort(yhat)[-k]
    sure = yhat > cut + eps
    tied = ~sure & (yhat >= cut - eps)
    slots = k - int(sure.sum())
    hits = int((sure & true_top).sum())
    tied_hits = int((tied & true_top).sum())
    tied_miss = int(tied.sum()) - tied_hits
    lo = hits + max(0, slots - tied_miss)
    hi = hits + min(slots, tied_hits)
    return lo / k, hi / k


def _true_top_sets(y, k, eps_y, max_sets=4096):
    """Every top-k membership the labels permit, when labels tie at the cutoff.

    `precision_at_k_frac` fixes the true top set with `np.argsort(-y)`, so
    compounds tied at the k-th largest label are admitted in an arbitrary
    order. That is the same arbitrariness the prediction side has, on the
    other side of the comparison (plan.md Amendment 8).
    """
    cut = np.sort(y)[-k]
    sure = y > cut + eps_y
    tied = np.flatnonzero(~sure & (y >= cut - eps_y))
    slots = k - int(sure.sum())
    if slots <= 0 or len(tied) == slots:
        base = sure.copy()
        base[tied] = True
        return [base]
    from itertools import combinations
    n_sets = comb(len(tied), slots)
    if n_sets > max_sets:                       # fall back to the fixed set
        base = sure.copy()
        base[np.argsort(-y)[:k]] = True
        return [base]
    out = []
    for pick in combinations(tied, slots):
        m = sure.copy()
        m[list(pick)] = True
        out.append(m)
    return out


def precision_at_k_frac_bounds(y, yhat, frac: float = 0.10,
                               eps: float = 1e-5,
                               eps_y: float = 0.0) -> tuple[float, float]:
    """Lowest and highest `precision_at_k_frac` any tie-breaking could give.

    `precision_at_k_frac` cuts both rankings with `np.argsort`, so compounds
    tied at the cutoff are admitted in an arbitrary order and a last-digit
    difference between two reruns can move the metric by 1/k. Predictions
    within `eps` of the k-th largest prediction are treated as tied, and
    labels within `eps_y` of the k-th largest label likewise; exact label ties
    are always resolved both ways, whatever `eps_y` is (plan.md Amendment 8).
    """
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    if y.ndim != 1 or y.shape != yhat.shape or not len(y):
        raise ValueError("expected nonempty matching one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("tie bounds require finite labels and predictions")
    if not np.isfinite(eps) or eps < 0 or not 0 < frac <= 1:
        raise ValueError("require eps >= 0 and 0 < frac <= 1")
    if not np.isfinite(eps_y) or eps_y < 0:
        raise ValueError("require eps_y >= 0")
    k = max(1, int(round(len(y) * frac)))
    bounds = [_precision_bounds_given_true_top(yhat, t, k, eps)
              for t in _true_top_sets(y, k, eps_y)]
    return float(min(b[0] for b in bounds)), float(max(b[1] for b in bounds))


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
