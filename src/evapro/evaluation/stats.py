"""Learning-curve summarisation and paired arm comparison (plan.md §6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ArmComparison:
    arm: str
    baseline: str
    median_delta: float
    p_raw: float
    p_holm: float = float("nan")
    n_seeds: int = 0

    @property
    def inconclusive(self) -> bool:
        """With 10 seeds we are powered only for large effects; a non-significant
        result is reported as inconclusive, not as 'no difference'."""
        return self.p_holm > 0.05


def paired_compare(arm_scores: dict[str, list[float]], baseline: str,
                   lower_is_better: bool = True) -> list[ArmComparison]:
    """Paired Wilcoxon of each arm vs the baseline across seeds, Holm-corrected.

    `arm_scores` maps arm id -> per-seed scores, aligned by seed order.
    """
    base = np.asarray(arm_scores[baseline], dtype=float)
    results = []
    for arm, scores in arm_scores.items():
        if arm == baseline:
            continue
        vals = np.asarray(scores, dtype=float)
        if len(vals) != len(base):
            raise ValueError(f"arm {arm!r} has {len(vals)} seeds, baseline has {len(base)}")
        delta = float(np.median(base - vals)) if lower_is_better else float(np.median(vals - base))
        p = float(stats.wilcoxon(vals, base).pvalue) if np.any(vals != base) else 1.0
        results.append(ArmComparison(arm, baseline, delta, p, n_seeds=len(vals)))

    return _holm(results)


def _holm(results: list[ArmComparison]) -> list[ArmComparison]:
    """Holm-Bonferroni over the frozen arm family."""
    m = len(results)
    for rank, res in enumerate(sorted(results, key=lambda r: r.p_raw)):
        res.p_holm = min(1.0, res.p_raw * (m - rank))
    # Enforce monotonicity of adjusted p-values.
    running = 0.0
    for res in sorted(results, key=lambda r: r.p_raw):
        running = res.p_holm = max(running, res.p_holm)
    return sorted(results, key=lambda r: r.p_raw)


def n_to_reach(train_sizes, scores, target: float, lower_is_better: bool = True) -> float:
    """Training-set size at which a learning curve first reaches `target`.

    Linear interpolation between bracketing points; inf if never reached.
    """
    sizes = np.asarray(train_sizes, dtype=float)
    vals = np.asarray(scores, dtype=float)
    order = np.argsort(sizes)
    sizes, vals = sizes[order], vals[order]

    hit = vals <= target if lower_is_better else vals >= target
    if not hit.any():
        return float("inf")
    j = int(np.argmax(hit))
    if j == 0:
        return float(sizes[0])
    x0, x1, y0, y1 = sizes[j - 1], sizes[j], vals[j - 1], vals[j]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def data_efficiency_ratio(baseline_curve, transfer_curve, target: float) -> float:
    """DER = N_baseline / N_transfer at a fixed target score (plan.md §6).

    Each curve is (train_sizes, scores). DER > 1 means transfer bought data.
    """
    n_base = n_to_reach(*baseline_curve, target=target)
    n_tl = n_to_reach(*transfer_curve, target=target)
    if n_tl in (0.0, float("inf")):
        return float("nan") if n_tl == 0.0 else 0.0
    return float(n_base / n_tl)
