"""Split-conformal prediction sets for a binary fraud probability (Stage 4, second half).

Nonconformity of (x, y) is 1 - p_hat(y | x), the complement of the probability assigned to the TRUE class.
With n calibration scores, the threshold is the ceil((n + 1)(1 - alpha))-th smallest score (finite-sample rule); if
that rank exceeds n the threshold is +inf and every set is {0, 1}. A set may be EMPTY if neither class is
conforming; empty sets are kept and counted, not repaired.

Guarantee: under exchangeability of calibration and test points, P(y in set) >= 1 - alpha (MARGINAL, not
per-class and not conditional). Under temporal drift or an adversary that guarantee does not apply; empirical
coverage on a drifted window is a measurement, not a bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..errors import ProtocolError


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ProtocolError("conformal calibration needs at least one score")
    if not 0.0 < alpha < 1.0:
        raise ProtocolError("alpha must be in (0, 1)")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def weighted_conformal_quantile(scores: Sequence[float], weights: Sequence[float], alpha: float, test_weight: float = 1.0) -> float:
    """Weighted split conformal (Tibshirani et al., 2019 style): quantile of the weighted score distribution with the
    test point's weight placed at +inf. Used for the 'drift-weighted' variant; the paper does not define its weights
    (docs/DECISIONS.md D-07)."""
    s = np.asarray(scores, float)
    w = np.asarray(weights, float)
    if len(s) == 0 or (w < 0).any() or w.sum() <= 0:
        raise ProtocolError("invalid scores/weights")
    order = np.argsort(s, kind="mergesort")
    s, w = s[order], w[order]
    p = w / (w.sum() + test_weight)
    cum = np.cumsum(p)
    idx = np.searchsorted(cum, 1.0 - alpha, side="left")
    return float(s[idx]) if idx < len(s) else float("inf")


def recency_weights(times: Sequence[float], half_life: float) -> np.ndarray:
    """w_i = 0.5 ** ((t_max - t_i) / half_life): recent calibration points count more."""
    t = np.asarray(times, float)
    return 0.5 ** ((t.max() - t) / half_life)


@dataclass
class SplitConformal:
    alpha: float
    threshold: Optional[float] = None
    n_cal: int = 0

    def fit(self, p_fraud, y, weights: Optional[Sequence[float]] = None) -> "SplitConformal":
        p = np.asarray(p_fraud, float)
        y = np.asarray(y).astype(int)
        if p.shape != y.shape:
            raise ProtocolError("probabilities and labels differ in length")
        if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
            raise ProtocolError("probabilities must be finite and in [0, 1]")
        scores = 1.0 - np.where(y == 1, p, 1.0 - p)
        self.threshold = (conformal_quantile(scores, self.alpha) if weights is None
                          else weighted_conformal_quantile(scores, weights, self.alpha))
        self.n_cal = len(p)
        return self

    def predict_sets(self, p_fraud) -> list[tuple[int, ...]]:
        if self.threshold is None:
            raise ProtocolError("SplitConformal.fit must be called first")
        p = np.asarray(p_fraud, float)
        out = []
        for pi in p:
            members = tuple(c for c, pc in ((0, 1.0 - pi), (1, pi)) if 1.0 - pc <= self.threshold)
            out.append(members)
        return out

    def coverage(self, p_fraud, y) -> dict:
        sets = self.predict_sets(p_fraud)
        y = np.asarray(y).astype(int)
        covered = np.array([int(yi) in s for yi, s in zip(y, sets)])
        return {"empirical_coverage": float(covered.mean()), "empirical_miscoverage": float(1 - covered.mean()), "target_miscoverage": self.alpha,
                "mean_set_size": float(np.mean([len(s) for s in sets])), "empty_sets": int(sum(len(s) == 0 for s in sets)), "n": int(len(y)),
                "per_class_coverage": {int(c): float(covered[y == c].mean()) for c in (0, 1) if (y == c).any()}}
