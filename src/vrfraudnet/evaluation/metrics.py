"""Metric definitions, written out so every convention is explicit and testable.

Orientation: label 1 = fraud = the positive class; a HIGHER score means MORE fraud-like. Isolation Forest and
any other anomaly detector must be negated by its adapter before it reaches these functions.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np
from scipy.stats import rankdata

from ..errors import ProtocolError


def _prep(y, s):
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s, dtype=float).ravel()
    if y.shape != s.shape:
        raise ProtocolError(f"labels and scores differ in length: {y.shape} vs {s.shape}")
    if not np.isfinite(s).all():
        raise ProtocolError("scores contain NaN or infinity")
    if not set(np.unique(y)) <= {0, 1}:
        raise ProtocolError("labels must be 0/1 with 1 = fraud")
    return y, s


def pr_points(y, s):
    """Precision/recall at every DISTINCT score threshold (tied scores form one threshold)."""
    y, s = _prep(y, s)
    p = int(y.sum())
    if p == 0:
        raise ProtocolError("no positive labels: AUPRC / recall are undefined for this partition")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]       # last index of each tie group
    tp, fp = tp[last], fp[last]
    return tp / (tp + fp), tp / p


def auprc(y, s, convention: Literal["average_precision", "trapezoid"] = "average_precision") -> float:
    """Area under the precision-recall curve.

    "average_precision" (default): AP = sum_k (R_k - R_{k-1}) P_k over distinct thresholds (step interpolation, no
    linear interpolation). "trapezoid": linear interpolation between the PR points, starting from (R=0, P=1). Neither
    convention dominates the other: they can differ in either direction on the same scores, so the convention is recorded with every result. The paper says only that AUPRC was computed "across all unique score thresholds"; the
    default follows that wording (docs/DECISIONS.md D-08).
    """
    prec, rec = pr_points(y, s)
    if convention == "average_precision":
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    if convention == "trapezoid":
        r = np.r_[0.0, rec]
        pp = np.r_[1.0, prec]
        return float(np.sum(np.diff(r) * (pp[1:] + pp[:-1]) / 2.0))
    raise ValueError(f"unknown AUPRC convention {convention!r}")


def roc_auc(y, s) -> float:
    y, s = _prep(y, s)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        raise ProtocolError("ROC-AUC needs both classes")
    ranks = rankdata(s)                                    # average ranks: ties count 1/2
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def confusion(y, s, threshold: float):
    """Predict fraud when score >= threshold."""
    y, s = _prep(y, s)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def mcc_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    d = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def threshold_metrics(y, s, threshold: float) -> dict:
    tp, fp, fn, tn = confusion(y, s, threshold)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": f1_from_counts(tp, fp, fn), "mcc": mcc_from_counts(tp, fp, fn, tn)}


def select_f1_threshold(y_val, s_val) -> float:
    """Validation-selected operating threshold: maximises fraud-class F1. Among exact ties the LARGEST threshold wins.

    Distinct from the Stage-1 routing thresholds (T_low, T_high), which govern the triage gate.
    """
    y, s = _prep(y_val, s_val)
    if y.sum() == 0:
        raise ProtocolError("cannot select an F1 threshold without positives in validation")
    best_f1, best_t = -1.0, float(s.max())
    for t in np.unique(s)[::-1]:                         # descending: first strict improvement keeps the larger threshold
        tp, fp, fn, _ = confusion(y, s, t)
        f1 = f1_from_counts(tp, fp, fn)
        if f1 > best_f1 + 1e-15:
            best_f1, best_t = f1, float(t)
    return best_t


def top_k_size(n: int, fraction: float = 0.01) -> int:
    """k = ceil(fraction * N) (the paper's 'highest-scoring ceiling of 1%')."""
    return int(math.ceil(fraction * n - 1e-12))


def top_k_indices(s, k: int) -> np.ndarray:
    """Deterministic review set: sort by (-score, index); ties are broken toward the LOWER row index."""
    s = np.asarray(s, dtype=float)
    return np.lexsort((np.arange(len(s)), -s))[:k]


def recall_at_top_fraction(y, s, fraction: float = 0.01) -> float:
    y, s = _prep(y, s)
    p = int(y.sum())
    if p == 0:
        raise ProtocolError("no positives in this window: Recall@top-k is undefined")
    k = top_k_size(len(y), fraction)
    return float(y[top_k_indices(s, k)].sum() / p)


def recall_ceiling(n: int, positives: int, fraction: float = 0.01) -> float:
    """Maximum attainable Recall@top-fraction: min(1, k / P), k = ceil(fraction * N)."""
    if positives <= 0:
        raise ProtocolError("ceiling is undefined without positives")
    return min(1.0, top_k_size(n, fraction) / positives)


def recall_at_top_fraction_bounds(y, s, fraction: float = 0.01) -> tuple[float, float]:
    """(worst, best) Recall@top-k over all orderings of scores that are tied across the cut-off."""
    y, s = _prep(y, s)
    p, k = int(y.sum()), top_k_size(len(y), fraction)
    if p == 0:
        raise ProtocolError("no positives in this window: Recall@top-k is undefined")
    order = np.argsort(-s, kind="mergesort")
    cut = s[order][k - 1] if k else np.inf
    above = int(y[s > cut].sum())
    tied = s == cut
    n_tied, p_tied = int(tied.sum()), int(y[tied].sum())
    slots = k - int((s > cut).sum())
    best = above + min(p_tied, slots)
    worst = above + max(0, slots - (n_tied - p_tied))
    return worst / p, best / p


def all_metrics(y_test, s_test, threshold: float, convention: str = "average_precision") -> dict:
    """The five Table-5 metrics, with the F1/MCC threshold supplied (frozen from validation)."""
    tm = threshold_metrics(y_test, s_test, threshold)
    return {"AUPRC": auprc(y_test, s_test, convention), "ROC-AUC": roc_auc(y_test, s_test), "F1": tm["f1"],
            "Recall@top-1%": recall_at_top_fraction(y_test, s_test), "MCC": tm["mcc"], "threshold": threshold}


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Equal-width-bin ECE (the paper does not state the binning; docs/DECISIONS.md D-09)."""
    y = np.asarray(y).astype(int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def prevalence(y) -> Optional[float]:
    y = np.asarray(y).ravel()
    return float(y.mean()) if len(y) else None
