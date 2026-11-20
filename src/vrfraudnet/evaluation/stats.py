"""Paired statistics for the repeated-seed comparisons (Table 6).

Seeds repeated on the SAME records are dependent. These tests describe the consistency and direction of paired
differences under stochastic retraining; they are not evidence from independent samples (the paper says so too).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from scipy.stats import rankdata

from ..errors import ProtocolError


def wilcoxon_one_sided(diffs: Sequence[float]) -> dict:
    """One-sided Wilcoxon signed-rank test of H1: median(diff) > 0.

    V is the sum of the ranks of the POSITIVE differences (maximum n(n+1)/2). Zero differences are dropped
    (Wilcoxon's rule); ties in |d| use average ranks. The p-value is exact when there are no ties/zeros and
    n <= 50, otherwise the normal approximation with continuity correction is used. `method` says which.
    """
    d = np.asarray(diffs, dtype=float).ravel()
    if not np.isfinite(d).all():
        raise ProtocolError("differences contain NaN or infinity")
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        raise ProtocolError("all paired differences are zero")
    ranks = rankdata(np.abs(nz))
    v = float(ranks[nz > 0].sum())
    has_ties = len(np.unique(np.abs(nz))) != n
    exact = (not has_ties) and n <= 50 and len(nz) == len(d)
    res = stats.wilcoxon(nz, alternative="greater", method="exact" if exact else "approx", zero_method="wilcox")
    return {"V": v, "V_max": n * (n + 1) / 2, "n": int(len(d)), "n_effective": int(n), "p_one_sided": float(res.pvalue),
            "method": "exact" if exact else "normal_approx", "ties": bool(has_ties), "zeros_dropped": int(len(d) - n)}


def exact_p_from_V(v: int, n: int) -> float:
    """P(V >= v) under H0 for n non-zero, tie-free pairs, by exact enumeration of the signed-rank distribution."""
    dist = np.zeros(n * (n + 1) // 2 + 1)
    dist[0] = 1.0
    for r in range(1, n + 1):
        new = dist.copy()
        new[r:] += dist[: len(dist) - r]
        dist = new
    dist /= 2.0 ** n
    return float(dist[int(v):].sum())


def implied_V_from_p(p: float, n: int) -> int:
    """The V whose exact one-sided tail probability is closest (in log scale) to `p`, for n tie-free pairs."""
    best, best_err = n * (n + 1) // 2, float("inf")
    for v in range(n * (n + 1) // 2, -1, -1):
        err = abs(np.log(exact_p_from_V(v, n)) - np.log(p))
        if err < best_err:
            best, best_err = v, err
        elif err > best_err and v < best - 3:
            break
    return best


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, cumulative maximum, clipped at 1), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def rank_biserial(v: float, n: int) -> float:
    """r = (W+ - W-) / (n(n+1)/2). One of several 'effect size' conventions; the paper names none."""
    total = n * (n + 1) / 2
    return float((v - (total - v)) / total)


def z_over_sqrt_n(v: float, n: int) -> float:
    """Matched-pairs r = Z / sqrt(n) with the normal approximation of V."""
    mean = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return float(((v - mean) / sd) / np.sqrt(n))


def compare_to_baselines(model_scores: dict[str, Sequence[float]], baseline_scores: dict[str, dict[str, Sequence[float]]]) -> list[dict]:
    """Pair VR-FraudNet AUPRC with each baseline by (dataset, seed) and run the pooled one-sided Wilcoxon + Holm.

    `model_scores[dataset]` and `baseline_scores[baseline][dataset]` are AUPRC lists in the SAME seed order.
    """
    rows = []
    for name, per_ds in baseline_scores.items():
        d = np.concatenate([np.asarray(model_scores[ds], float) - np.asarray(per_ds[ds], float) for ds in sorted(model_scores)])
        w = wilcoxon_one_sided(d)
        rows.append({"baseline": name, "mean_delta_auprc_pp": float(d.mean() * 100), **w,
                     "effect_rank_biserial": rank_biserial(w["V"], w["n_effective"]), "effect_z_over_sqrt_n": z_over_sqrt_n(w["V"], w["n_effective"])})
    adj = holm([r["p_one_sided"] for r in rows])
    for r, a in zip(rows, adj):
        r["holm_adjusted_p"] = a
    return rows


def sample_sd(x: Sequence[float]) -> float:
    return float(np.std(np.asarray(x, float), ddof=1))
