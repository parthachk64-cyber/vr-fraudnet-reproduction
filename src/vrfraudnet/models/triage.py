"""Stage 1: LightGBM triage with a class-balanced focal surrogate, plus the threshold gate (Eq. 5 as in Sec. 3.8.1).

Routing (inclusive review band, Sec. 3.8.1):   p < T_low  -> ACCEPT      p > T_high -> REJECT      T_low <= p <= T_high -> ESCALATE
Eq. 5 as printed is garbled (docs/AUDIT.md A-20); this is the rule the text states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..errors import OptionalDependencyError, ProtocolError
from ..evaluation.metrics import auprc
from ..published import TRIAGE_PARAMS
from ..types import Route

ACCEPT, REJECT, ESCALATE = "accept", "reject", "escalate"


def route_probability(p, low: float, high: float):
    """Vectorised gate. Both endpoints belong to the review band. Returns an array of 'accept'/'reject'/'escalate'."""
    if not low < high:
        raise ProtocolError(f"need T_low < T_high, got {low} and {high}")
    p = np.asarray(p, float)
    return np.where(p < low, ACCEPT, np.where(p > high, REJECT, ESCALATE))


# ------------------------------------------------------------------------------------ focal loss
def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def focal_loss(y, z, gamma: float = 2.0, alpha_pos: float = 1.0, alpha_neg: float = 1.0) -> np.ndarray:
    """Per-sample binary focal loss on the raw score z (logit): -a (1-p_t)^gamma log p_t."""
    p = np.clip(_sigmoid(z), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return np.where(y == 1, -alpha_pos * (1 - p) ** gamma * np.log(p), -alpha_neg * p ** gamma * np.log(1 - p))


def focal_grad_hess(y, z, gamma: float = 2.0, alpha_pos: float = 1.0, alpha_neg: float = 1.0, min_hess: float = 1e-6):
    """Analytic gradient and Hessian of `focal_loss` w.r.t. z. The true Hessian can be negative for focal loss, so it is floored at
    `min_hess` (the usual practice for boosted trees); the gradient is exact and finite-difference tested."""
    p = np.clip(_sigmoid(z), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    g1 = alpha_pos * (gamma * p * (1 - p) ** gamma * np.log(p) - (1 - p) ** (gamma + 1))
    g0 = alpha_neg * (-gamma * p ** gamma * (1 - p) * np.log(1 - p) + p ** (gamma + 1))
    dg1 = alpha_pos * ((1 - p) ** (gamma - 1) * (gamma * (1 - p) * np.log(p) - gamma ** 2 * p * np.log(p) + (2 * gamma + 1) * (1 - p)))
    dg0 = alpha_neg * (p ** (gamma - 1) * (-gamma ** 2 * (1 - p) * np.log(1 - p) + gamma * p * np.log(1 - p) + (2 * gamma + 1) * p))
    grad = np.where(y == 1, g1, g0)
    hess = np.where(y == 1, dg1, dg0) * p * (1 - p)
    return grad, np.maximum(hess, min_hess)


class FocalObjective:
    """Picklable LightGBM objective (a closure would make the fitted model impossible to save)."""

    def __init__(self, gamma: float, a_pos: float, a_neg: float):
        self.gamma, self.a_pos, self.a_neg = gamma, a_pos, a_neg

    def __call__(self, y_true, z):
        return focal_grad_hess(y_true, z, self.gamma, self.a_pos, self.a_neg)


class _AuprcMetric:
    def __init__(self, sigmoid: bool):
        self.sigmoid = sigmoid

    def __call__(self, y_true, z):
        return "auprc", auprc(y_true, _sigmoid(z) if self.sigmoid else z), True


def class_balanced_alpha(y, beta: float = 0.9999) -> tuple[float, float]:
    """Class-balanced weights (Cui et al., 2019): w_c proportional to (1 - beta) / (1 - beta^n_c), normalised to mean 1 over the samples."""
    y = np.asarray(y)
    n1, n0 = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    w1, w0 = (1 - beta) / (1 - beta ** n1), (1 - beta) / (1 - beta ** n0)
    mean = (w1 * n1 + w0 * n0) / (n1 + n0)
    return w1 / mean, w0 / mean


def cost_sample_weights(y, c_fn: float, c_fp: float, lam: float = 0.5) -> np.ndarray:
    """Cost-sensitive sample weights (decision D-06): w = (1 - lam) + lam * c_hat, with c_hat = cost / max(c_FN, c_FP) per class
    (Sec. 3.6.1 normalisation). lam is the paper's cost-sensitive weight (0.50). The paper gives the weight but not its functional form."""
    m = max(c_fn, c_fp)
    y = np.asarray(y)
    return (1 - lam) + lam * np.where(y == 1, c_fn / m, c_fp / m)


# ------------------------------------------------------------------------------------ the model
@dataclass
class TriageFit:
    best_iteration: int
    val_auprc: float
    params: dict


class TriageModel:
    def __init__(self, params: Optional[dict] = None, *, objective: str = "focal", gamma: float = 2.0, seed: int = 0, n_jobs: int = 1):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise OptionalDependencyError("lightgbm", "trees", "the LightGBM triage classifier") from e
        if objective not in ("focal", "binary"):
            raise ValueError("objective must be 'focal' or 'binary'")
        self.params = {**TRIAGE_PARAMS, **(params or {})}
        self.objective, self.gamma, self.seed, self.n_jobs = objective, gamma, seed, n_jobs
        self.model_ = None
        self.fit_: Optional[TriageFit] = None

    def fit(self, X, y, X_val, y_val, sample_weight=None) -> "TriageModel":
        import lightgbm as lgb

        y = np.asarray(y).astype(int)
        p = self.params
        a1, a0 = class_balanced_alpha(y)
        obj, metric = FocalObjective(self.gamma, a1, a0), _AuprcMetric(self.objective == "focal")

        kw = dict(n_estimators=p["n_estimators"], max_depth=p["max_depth"], num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
                  min_child_samples=p["min_child_samples"], colsample_bytree=p["colsample_bytree"], subsample=p["subsample"],
                  subsample_freq=p["subsample_freq"], reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"], random_state=self.seed,
                  n_jobs=self.n_jobs, verbose=-1)
        if self.objective == "focal":
            clf = lgb.LGBMClassifier(objective=obj, **kw)
        else:
            clf = lgb.LGBMClassifier(objective="binary", **kw)
        clf.fit(X, y, sample_weight=sample_weight, eval_set=[(X_val, np.asarray(y_val).astype(int))], eval_metric=metric,
                callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)])
        self.model_ = clf
        best = clf.best_iteration_ or p["n_estimators"]
        self.fit_ = TriageFit(int(best), float(auprc(y_val, self.predict_proba(X_val))), dict(p))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise ProtocolError("TriageModel.fit must be called first")
        raw = self.model_.predict(X, raw_score=True)
        return _sigmoid(raw)


# ------------------------------------------------------------------------------------ threshold selection
def select_thresholds(p_val, y_val, *, max_escalation: float = 0.05, c_fn: float = 1.0, c_fp: float = 1.0 / 60, c_review: float = 1.0 / 60,
                      grid: int = 41) -> dict:
    """Choose (T_low, T_high) on validation: minimise expected operational cost subject to escalating at most `max_escalation` of the rows.

    Common-path cost: accepted frauds cost c_FN each, rejected legitimates cost c_FP each; every escalated row costs c_review (the LM/verifier/
    human path). The paper says threshold selection 'prioritized validation AUPRC and expected operational cost' under an escalation cap but does
    not give the objective in closed form: this explicit objective is a RECONSTRUCTION (docs/DECISIONS.md D-02). The published pairs are in
    `published.PUBLISHED_ROUTING_THRESHOLDS`; use them directly to follow the paper, or this function to refit.
    """
    p, y = np.asarray(p_val, float), np.asarray(y_val).astype(int)
    qs = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    best = None
    for i, lo in enumerate(qs):
        for hi in qs[i + 1:]:
            r = route_probability(p, lo, hi)
            esc = float((r == ESCALATE).mean())
            if esc > max_escalation:
                continue
            cost = c_fn * np.sum((r == ACCEPT) & (y == 1)) + c_fp * np.sum((r == REJECT) & (y == 0)) + c_review * np.sum(r == ESCALATE)
            if best is None or cost < best["cost"] - 1e-12:
                best = {"low": float(lo), "high": float(hi), "cost": float(cost), "escalation_rate": esc}
    if best is None:
        raise ProtocolError(f"no (T_low, T_high) pair satisfies escalation <= {max_escalation}")
    return best
