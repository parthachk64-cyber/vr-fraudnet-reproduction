"""The nine baselines of Table 5, each a real, runnable adapter with one interface:

    model = make_baseline(name, seed=...).fit(X_train, y_train, X_val, y_val)
    scores = model.score(X)            # FRAUD orientation: higher = more fraud-like (Isolation Forest is negated here)

The paper gives no architectures or hyperparameters for the neural baselines; the settings below are this repository's reconstruction
(docs/DECISIONS.md D-16) and are the same on every dataset. Tree and linear models use their library defaults unless noted.
All baselines see the SAME preprocessed matrices as VR-FraudNet's triage stage (same train-only fit, same partitions).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..errors import OptionalDependencyError
from ..published import BASELINES


class Baseline(Protocol):
    name: str

    def fit(self, X, y, X_val=None, y_val=None) -> "Baseline": ...
    def score(self, X) -> np.ndarray: ...


class _SK:
    def __init__(self, name, est, proba=True):
        self.name, self.est, self.proba = name, est, proba

    def fit(self, X, y, X_val=None, y_val=None):
        self.est.fit(X, y)
        return self

    def score(self, X):
        if not self.proba:                                           # Isolation Forest: score_samples is HIGHER for normal points -> negate
            return -self.est.score_samples(X)
        return self.est.predict_proba(X)[:, 1]


class _XGB(_SK):
    pass


def make_baseline(name: str, seed: int = 0, *, epochs: int = 30, n_jobs: int = 1) -> Baseline:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINES}")
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if name == "logistic_regression":
        return _SK(name, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    if name == "isolation_forest":
        return _SK(name, IsolationForest(n_estimators=200, random_state=seed, n_jobs=n_jobs), proba=False)
    if name == "random_forest":
        return _SK(name, RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=seed, n_jobs=n_jobs))
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise OptionalDependencyError("xgboost", "trees", "the XGBoost baseline") from e
        return _SK(name, xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                                           random_state=seed, n_jobs=n_jobs, verbosity=0))
    if name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise OptionalDependencyError("lightgbm", "trees", "the LightGBM baseline") from e
        return _SK(name, lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                            is_unbalance=True, random_state=seed, n_jobs=n_jobs, verbose=-1))
    from .neural import NeuralBaseline

    return NeuralBaseline(name, seed=seed, epochs=epochs)


def all_baselines(seed: int = 0, **kw) -> dict:
    return {n: make_baseline(n, seed, **kw) for n in BASELINES}
