"""From-scratch baselines (plan.md §4). B1 is the arm to beat."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class MedianRegressor:
    """B0 sanity floor. Any arm that fails to beat this is broken, not weak."""

    def fit(self, X, y):
        self._v = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._v)


def build_lightgbm(**params):
    """B1: ECFP4 counts + LightGBM."""
    import lightgbm as lgb

    defaults = dict(objective="regression", n_estimators=1000, learning_rate=0.05,
                    num_leaves=31, min_child_samples=5, verbose=-1)
    return lgb.LGBMRegressor(**{**defaults, **params})


def build_random_forest(**params):
    """B2: RDKit descriptors + Random Forest."""
    defaults = dict(n_estimators=500, min_samples_leaf=1, n_jobs=-1, random_state=0)
    return RandomForestRegressor(**{**defaults, **params})


REGISTRY = {
    "median": lambda **kw: MedianRegressor(),
    "lightgbm": build_lightgbm,
    "random_forest": build_random_forest,
}


def build_model(spec: dict):
    kind = spec.get("type")
    if kind not in REGISTRY:
        raise NotImplementedError(
            f"model {kind!r} is declared in config but not implemented yet; "
            f"available: {sorted(REGISTRY)}"
        )
    return REGISTRY[kind](**{k: v for k, v in spec.items() if k not in {"type", "task"}})
