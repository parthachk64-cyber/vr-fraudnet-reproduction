"""Cost-sensitive operating evaluation (Supplementary Table S1) and subgroup fairness (S2).

The cost model the supplement's arithmetic implies (verified in `audit`): total loss = FN * c_FN + FP * c_FP + TP * c_TPreview,
and savings = 1 - total / (P * c_FN), i.e. relative to a system that reviews nothing and loses c_FN on every fraud.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..errors import ProtocolError
from .metrics import auprc, confusion, expected_calibration_error


@dataclass(frozen=True)
class CostModel:
    c_fn: float  # loss for each missed fraud
    c_fp: float  # review cost for each false alarm
    c_tp_review: float  # review cost for each caught fraud

    def normalised(self) -> tuple[float, float]:
        """(FN weight, FP weight) divided by the larger of the two, for training weights (Section 3.6.1)."""
        m = max(self.c_fn, self.c_fp)
        if m <= 0:
            raise ProtocolError("costs must be positive")
        return self.c_fn / m, self.c_fp / m


def evaluate_costs(tp: int, fn: int, fp: int, costs: CostModel) -> dict:
    p = tp + fn
    if p == 0:
        raise ProtocolError("no positives")
    fn_loss, fp_loss, tp_review = fn * costs.c_fn, fp * costs.c_fp, tp * costs.c_tp_review
    total = fn_loss + fp_loss + tp_review
    return {"tp": tp, "fn": fn, "fp": fp, "sum_fn_loss": fn_loss, "sum_fp_loss": fp_loss, "sum_tp_review": tp_review,
            "total_loss": total, "savings_pct": 100.0 * (1.0 - total / (p * costs.c_fn)), "capacity_recall_inputs": {"positives": p}}


def evaluate_costs_at_threshold(y, s, threshold: float, costs: CostModel) -> dict:
    tp, fp, fn, _ = confusion(y, s, threshold)
    return evaluate_costs(tp, fn, fp, costs)


def infer_unit_costs(rows: list[dict]) -> CostModel:
    """Infer per-unit costs from published S1 rows (sums / counts). Raises if rows disagree: it never guesses."""
    cfn = {round(r["sum_fn_loss"] / r["fn"], 6) for r in rows if r["fn"]}
    cfp = {round(r["sum_fp_loss"] / r["fp"], 6) for r in rows if r["fp"]}
    ctp = {round(r["sum_tp_review"] / r["tp"], 6) for r in rows if r["tp"]}
    if len(cfn) != 1 or len(cfp) != 1 or len(ctp) != 1:
        raise ProtocolError(f"rows imply inconsistent unit costs: FN {cfn}, FP {cfp}, TP {ctp}")
    return CostModel(cfn.pop(), cfp.pop(), ctp.pop())


# ------------------------------------------------------------------ fairness
def subgroup_report(y, score, protected_mask, threshold: Optional[float] = None, *, n_bins: int = 10) -> dict:
    """AUPRC per group, the AUPRC gap in percentage points (protected - reference), ECE per group, and
    disparate-impact ratios at `threshold` (both conventions, because the paper does not state which it uses)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, float)
    m = np.asarray(protected_mask, bool)
    out = {}
    for name, sel in (("protected", m), ("reference", ~m)):
        if sel.sum() == 0 or y[sel].sum() == 0:
            raise ProtocolError(f"group {name!r} is empty or has no positives")
        out[f"auprc_{name}"] = auprc(y[sel], s[sel])
        out[f"ece_{name}"] = expected_calibration_error(y[sel], s[sel], n_bins)
        out[f"n_{name}"] = int(sel.sum())
    out["gap_pp"] = 100.0 * (out["auprc_protected"] - out["auprc_reference"])
    out["ece_delta"] = out["ece_protected"] - out["ece_reference"]
    if threshold is not None:
        flag_p, flag_r = float((s[m] >= threshold).mean()), float((s[~m] >= threshold).mean())
        out["flag_rate_protected"], out["flag_rate_reference"] = flag_p, flag_r
        out["di_flag_rate_ratio"] = flag_p / flag_r if flag_r else float("nan")
        out["di_unflagged_ratio"] = (1 - flag_p) / (1 - flag_r) if flag_r < 1 else float("nan")
    return out
