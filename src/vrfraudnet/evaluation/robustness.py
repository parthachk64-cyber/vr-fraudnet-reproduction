"""Scoring of SUPPLIED, FIXED robustness benchmarks (Table 9 / Supplementary S4).

This module only SCORES files that someone else produced; it does not generate edited transactions, search for variants, or suggest edits.
A fixed file lists, for each original fraud, the model's score on the original and on each frozen variant. The evasion rate and its
denominator are therefore explicit, and inapplicable (dataset, family) cells are reported as such rather than as zeros.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from ..errors import ProtocolError

FAMILIES = {"e1_split": "transaction splitting", "e2_delay": "execution delay", "e3_mule": "mule-routing modification", "e5_feat": "bounded feature perturbation"}

# (dataset, family) -> reason it cannot be applied. A RECONSTRUCTION from what each dataset contains (docs/DECISIONS.md D-17); the paper's Table 9 names no dataset.
NOT_APPLICABLE = {
    ("D1", "e1_split"): "BAF has no transaction amount to split (applications, not payments)",
    ("D1", "e2_delay"): "BAF has no transaction clock finer than the month",
    ("D1", "e3_mule"): "BAF has no entity graph",
    ("D3", "e3_mule"): "IEEE-CIS has no account-to-account graph",
    ("D4", "e1_split"): "Elliptic++ amounts are anonymised features",
}


def applicable(dataset: str, family: str) -> tuple[bool, Optional[str]]:
    r = NOT_APPLICABLE.get((dataset.upper(), family))
    return (r is None, r)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score_variants(df: pd.DataFrame, threshold: float, dataset: str, denominator: str = "detected_originals") -> dict:
    """Required columns: original_id, family, budget, original_score, variant_score (every row is a fraud).
    evasion = variant score below `threshold`. Denominator:
      "detected_originals": only frauds the model caught BEFORE the edit (the edit's effect);
      "all_frauds": every original fraud (conflates misses that predate the edit)."""
    need = {"original_id", "family", "budget", "original_score", "variant_score"}
    if need - set(df.columns):
        raise ProtocolError(f"variant-score file lacks {sorted(need - set(df.columns))}")
    if denominator not in ("detected_originals", "all_frauds"):
        raise ValueError("denominator must be 'detected_originals' or 'all_frauds'")
    out = {"dataset": dataset, "threshold": threshold, "denominator": denominator, "cells": {}}
    for (fam, b), g in df.groupby(["family", "budget"]):
        ok, why = applicable(dataset, fam)
        key = f"{fam}|B={b}"
        if not ok:
            out["cells"][key] = {"status": "not_applicable", "reason": why}
            continue
        base = g[g["original_score"] >= threshold] if denominator == "detected_originals" else g
        k, n = int((base["variant_score"] < threshold).sum()), len(base)
        lo, hi = wilson(k, n)
        out["cells"][key] = {"status": "scored", "evaded": k, "n": n, "rate": k / n if n else float("nan"), "ci95": [lo, hi]}
    by_budget: dict = {}
    for key, c in out["cells"].items():
        if c["status"] == "scored":
            by_budget.setdefault(key.split("|")[1], []).append(c["rate"])
    out["mean_over_applicable_families"] = {b: float(np.mean(v)) for b, v in by_budget.items()}
    out["families_scored_per_budget"] = {b: len(v) for b, v in by_budget.items()}
    return out


def verifier_rejection_rate(rejected_flags, target: float = 0.05) -> dict:
    """Pr[V(r) = 0] over a fixed set of held-out rationale cases (S4b). `rejected_flags` is a list of booleans (True = verifier rejected)."""
    f = np.asarray(rejected_flags, bool)
    k, n = int(f.sum()), len(f)
    lo, hi = wilson(k, n)
    return {"n": n, "rejected": k, "rate": k / n if n else float("nan"), "ci95": [lo, hi], "target": target, "meets_target": bool(n and k / n <= target)}
