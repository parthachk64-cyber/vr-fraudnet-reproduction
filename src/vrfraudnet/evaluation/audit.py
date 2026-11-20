"""Mathematical audit of the published article. Runs before any expensive training and needs no data.

Everything here recomputes a DERIVED quantity (a difference, a mean, a ceiling, an adjusted p-value) from other
published numbers and compares it with the printed value. It never alters, tunes or "corrects" a published
figure: a mismatch is reported as a finding and both numbers are preserved.

Status vocabulary: consistent | inconsistent | unresolvable | note.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from .. import reported as R
from .metrics import recall_ceiling, top_k_size
from .stats import exact_p_from_V, holm, implied_V_from_p, rank_biserial, z_over_sqrt_n
from .cost import evaluate_costs, infer_unit_costs, CostModel


@dataclass
class Finding:
    id: str
    severity: str  # info | minor | major | blocking
    status: str
    title: str
    detail: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _m(cell) -> float:
    return R.mean(cell)


def a01_recall_ceiling() -> list[Finding]:
    t5, t8, s1 = R.load("table5"), R.load("table8"), {k: R.load(f"supplement_{k}") for k in ("s1a", "s1b", "s1c")}
    out = []
    n_d3 = next(r["nominal_test_n"] for r in t8["panels"]["8d"]["rows"] if r["fold"] == 3)
    spec = {"D1": ("5a", "s1a", None), "D2": ("5b", "s1b", int(0.2 * 5_078_345)), "D3": ("5c", "s1c", n_d3)}
    for ds, (panel, s1k, n) in spec.items():
        full = R.row(s1[s1k]["rows"], "Method", "Full VR-FraudNet")
        p = int(_m(full["TP"]) + _m(full["FN"]))
        n_note = "partition size from Table 8d (fold 3 nominal test n)" if ds == "D3" else "20% of 5,078,345 transactions (Table 1, 60/20/20 split)"
        if n is None:
            n = round(p / 0.011)
            n_note = "INFERRED: positives / 1.10% prevalence (Table 1); the partition size is not printed"
        ceiling = recall_ceiling(n, p, 0.01)
        col = [(r["method"], _m(r["Recall@top-1%"])) for r in t5["panels"][panel]["rows"]]
        over = [(m, v) for m, v in col if v > ceiling + 1e-12]
        prev = p / n
        status = "inconsistent" if over else "consistent"
        out.append(Finding(
            f"A-01{ds}", "blocking" if over else "info", status,
            f"Recall@top-1% ceiling on {ds}",
            f"Held-out positives P={p} (S1 TP+FN), N={n} ({n_note}); partition prevalence {prev:.4%}; k=ceil(0.01N)={top_k_size(n)}; "
            f"ceiling=min(1,k/P)={ceiling:.4f}. Printed Table {panel} values above the ceiling: {len(over)} of {len(col)}"
            + (f" (largest {max(v for _, v in over):.4f}, VR-FraudNet {dict(col)['VR-FraudNet']:.4f})." if over else "."),
            {"positives": p, "n": n, "k": top_k_size(n), "ceiling": ceiling, "partition_prevalence": prev,
             "stated_dataset_prevalence_pct": R.load("table1")["datasets"][ds]["fraud_rate_pct"], "values_above": over}))
    return out


def a02_table6() -> list[Finding]:
    t5, t6 = R.load("table5")["panels"], R.load("table6")
    names = {"Logistic regression": "Logistic regression", "Isolation forest": "Isolation forest", "MLP": "MLP", "1D-CNN": "1D-CNN",
             "LSTM": "LSTM", "Random Forest": "Random forest", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "TabTransformer": "TabTransformer"}
    out, bad = [], []
    for r in t6["rows"]:
        base = names[r["baseline"]]
        deltas = [_m(R.row(t5[p]["rows"], "method", "VR-FraudNet")["AUPRC"]) - _m(R.row(t5[p]["rows"], "method", base)["AUPRC"]) for p in ("5a", "5b", "5c")]
        implied = 100 * sum(deltas) / 3
        if not _close(implied, r["mean_delta_auprc_pp"], 0.006):
            bad.append((base, implied, r["mean_delta_auprc_pp"]))
    out.append(Finding("A-02", "major" if bad else "info", "inconsistent" if bad else "consistent", "Table 6 mean AUPRC deltas vs Table 5",
                       "Mean of the three per-dataset AUPRC differences recomputed from Table 5 and compared with Table 6." + (f" Mismatches: {bad}" if bad else " All nine agree to within rounding.")))
    # Holm and exact p
    raw = [r["raw_p_one_sided"] for r in t6["rows"]]
    adj = holm(raw)
    holm_bad = [(r["baseline"], a, r["holm_adjusted_p"]) for r, a in zip(t6["rows"], adj) if abs(a - r["holm_adjusted_p"]) / r["holm_adjusted_p"] > 0.01]
    out.append(Finding("A-03", "major" if holm_bad else "info", "inconsistent" if holm_bad else "consistent", "Table 6 Holm-adjusted p-values",
                       "Holm step-down recomputed from the printed raw p-values." + (f" Mismatches: {holm_bad}" if holm_bad else " All nine agree to 3 significant figures.")))
    pbad, missing = [], []
    for r in t6["rows"]:
        if r["wilcoxon_V"] is None:
            missing.append(r["baseline"])
            continue
        p_exact = exact_p_from_V(r["wilcoxon_V"], 30)
        if abs(p_exact - r["raw_p_one_sided"]) / r["raw_p_one_sided"] > 0.02:
            pbad.append({"baseline": r["baseline"], "printed_V": r["wilcoxon_V"], "exact_p_for_printed_V": p_exact,
                         "printed_p": r["raw_p_one_sided"], "V_implied_by_printed_p": implied_V_from_p(r["raw_p_one_sided"], 30)})
    out.append(Finding("A-04", "minor", "inconsistent" if pbad else "consistent", "Table 6 raw p-values vs printed V (n=30, exact, no ties assumed)",
                       f"Exact signed-rank tail P(V>=V_printed) compared with the printed raw p. The printed p-values do not belong to the printed V values for "
                       f"{len(pbad)} baseline(s): {pbad or 'none'}. No V is printed for: {missing}. Both numbers are preserved; neither is corrected.",
                       {"missing_V": missing, "mismatches": pbad}))
    eff = []
    for r in t6["rows"]:
        if r["wilcoxon_V"] is None:
            continue
        eff.append((r["baseline"], r["effect_size"], round(rank_biserial(r["wilcoxon_V"], 30), 2), round(z_over_sqrt_n(r["wilcoxon_V"], 30), 2)))
    match_rb = all(abs(e[1] - e[2]) <= 0.01 for e in eff)
    match_z = all(abs(e[1] - e[3]) <= 0.01 for e in eff)
    out.append(Finding("A-05", "minor", "consistent" if (match_rb or match_z) else "unresolvable", "Table 6 effect-size convention",
                       "The column is headed 'Effect size' with no named convention. Rank-biserial (W+ - W-)/(n(n+1)/2) and matched-pairs r = Z/sqrt(n) were "
                       f"each computed from the printed V; {'one matches' if (match_rb or match_z) else 'neither reproduces the printed values'}. "
                       "(printed, rank-biserial, z/sqrt n): " + str(eff), {"rows": eff}))
    return out


def a03_temporal_tables() -> list[Finding]:
    t8, t5 = R.load("table8")["panels"], R.load("table5")["panels"]
    out = []
    bad = []
    for r in t8["8a"]["rows"]:
        for ref_key, tgt_key in (("delta_m6_pp", "test_m6"), ("delta_m7_pp", "test_m7")):
            exp = 100 * (_m(r[tgt_key]) - _m(r["train_auprc"]))
            if not _close(exp, r[ref_key], 0.011):
                bad.append((r["method"], ref_key, exp, r[ref_key]))
    out.append(Finding("A-06", "major" if bad else "info", "inconsistent" if bad else "consistent", "Table 8a month-drift deltas",
                       "delta = 100 x (test month AUPRC - 'Train AUPRC' column). " + (f"Mismatches: {bad}" if bad else "All 12 agree to rounding.")))
    same = all(_close(_m(R.row(t8["8a"]["rows"], "method", m)["train_auprc"]), _m(R.row(t5["5a"]["rows"], "method", n)["AUPRC"]), 1e-9)
               for m, n in (("LightGBM", "LightGBM"), ("TabTransformer", "TabTransformer"), ("Full VR-FraudNet", "VR-FraudNet")))
    out.append(Finding("A-07", "minor", "note" if same else "consistent", "Table 8a reference column equals the Table 5a TEST AUPRC",
                       "The 'Train AUPRC' column of Table 8a equals the Table 5a AUPRC (a TEST-partition value, Sec. 3.8.3) for LightGBM, TabTransformer and VR-FraudNet. "
                       "The column label and the stated reference ('training-reference period', Sec. 4.5) therefore disagree; drift is measured from the held-out value."
                       if same else "No overlap with Table 5a."))
    bad = []
    for r in t8["8b"]["rows"]:
        folds = [_m(r[f"fold{i}"]) for i in range(1, 6)]
        if not _close(sum(folds) / 5, r["mean_across_folds_descriptive"], 0.00006):
            bad.append((r["method"], "mean", sum(folds) / 5, r["mean_across_folds_descriptive"]))
        if not _close(100 * (folds[4] - folds[2]), r["late_delta_f5_minus_f3_pp"], 0.011):
            bad.append((r["method"], "late_delta", 100 * (folds[4] - folds[2]), r["late_delta_f5_minus_f3_pp"]))
    out.append(Finding("A-08", "major" if bad else "info", "inconsistent" if bad else "consistent", "Table 8b fold means and F5-F3 contrast",
                       "Five-fold mean and 100 x (Fold5 - Fold3) recomputed. " + (f"Mismatches: {bad}" if bad else "All agree to rounding.")))
    f3 = _m(R.row(t8["8b"]["rows"], "method", "Full VR-FraudNet")["fold3"])
    out.append(Finding("A-09", "info", "consistent" if _close(f3, _m(R.row(t5["5c"]["rows"], "method", "VR-FraudNet")["AUPRC"]), 1e-9) else "inconsistent",
                       "Fold 3 equals the primary D3 holdout", "Table 8b Fold 3 (q=0.80) equals Table 5c for VR-FraudNet, as the paper states, by construction."))
    bad = []
    for r in t8["8c"]["rows"]:
        for a, b, c in (("pre_shock", "delta_pre_pp", "train_auprc"), ("post_shock", "delta_post_pp", "train_auprc")):
            if not _close(100 * (_m(r[a]) - _m(r[c])), r[b], 0.011):
                bad.append((r["method"], b))
        if not _close(100 * (_m(r["post_shock"]) - _m(r["pre_shock"])), r["shock_impact_pp"], 0.011):
            bad.append((r["method"], "shock_impact"))
    out.append(Finding("A-10", "major" if bad else "info", "inconsistent" if bad else "consistent", "Table 8c shock deltas and impact",
                       "Each contrast uses its own reference (train, pre-shock). " + (f"Mismatches: {bad}" if bad else "All 18 agree to rounding.")))
    full = R.row(t8["8c"]["rows"], "method", "Full VR-FraudNet")
    drop = 100 * (_m(full["train_auprc"]) - _m(full["post_shock"]))
    out.append(Finding("A-11", "info", "note", "D4 post-shock degradation vs the 5-point target",
                       f"Train reference {_m(full['train_auprc']):.4f} -> post-shock {_m(full['post_shock']):.4f} is {drop:.2f} pp, which exceeds the paper's own 5 pp target; "
                       "the paper states this openly and this repository preserves it.", {"drop_pp": drop}))
    rows = R.load("table8")["panels"]["8d"]["rows"]
    tot = {r["nominal_train_n"] + r["nominal_test_n"] for r in rows}
    out.append(Finding("A-12", "info", "consistent" if tot == {590540} else "inconsistent", "Table 8d fold sizes",
                       f"train + test per fold = {sorted(tot)} (IEEE-CIS has 590,540 transactions)."))
    return out


def a04_table7_9() -> list[Finding]:
    out = []
    for key, lab in (("table7a", "A-13"), ("table7b", "A-14")):
        t = R.load(key)
        full = _m(R.row(t["rows"], "id", "Full")["AUPRC"])
        bad = []
        for r in t["rows"]:
            if r["id"] == "Full":
                continue
            vals = r["AUPRC"] if isinstance(r["AUPRC"], list) else [_m(r["AUPRC"])]
            dl = r["delta_auprc_pp"] if isinstance(r["delta_auprc_pp"], list) else [r["delta_auprc_pp"]]
            for v, d in zip(vals, dl):
                if not _close(100 * (v - full), d, 0.011):
                    bad.append((r["id"], v, d))
        out.append(Finding(lab, "major" if bad else "info", "inconsistent" if bad else "consistent", f"{key} AUPRC deltas vs the full model",
                           "delta = 100 x (variant - full). " + (f"Mismatches: {bad}" if bad else "All agree to rounding.")))
    bad = []
    for p, pan in R.load("table9")["panels"].items():
        for r in pan["rows"]:
            mean4 = sum(_m(r[k]) for k in ("split", "delay", "mule", "feat")) / 4
            if not _close(mean4, r["mean"], 0.00006):
                bad.append((p, r["method"], mean4, r["mean"]))
    out.append(Finding("A-15", "major" if bad else "info", "inconsistent" if bad else "consistent", "Table 9 row means",
                       "Mean of the four family rates recomputed. " + (f"Mismatches: {bad}" if bad else "All 15 agree to 4 decimals.")))
    out.append(Finding("A-16", "minor", "unresolvable", "Table 9 dataset and denominator",
                       "Table 9 does not name the dataset(s) behind the evasion rates, nor the denominator (all fraud, or fraud detected before the edit). "
                       "The repository's robustness scorer takes both from the supplied benchmark file and reports them."))
    return out


def a05_supplement() -> list[Finding]:
    out = []
    ok, notes, costs = True, [], {}
    for k, ds in (("s1a", "D1"), ("s1b", "D2"), ("s1c", "D3")):
        rows = R.load(f"supplement_{k}")["rows"]
        flat = [{"tp": _m(r["TP"]), "fn": _m(r["FN"]), "fp": _m(r["FP"]), "sum_fn_loss": _m(r["Sum FN loss"]), "sum_fp_loss": _m(r["Sum FP loss"]),
                 "sum_tp_review": _m(r["Sum TP review"])} for r in rows]
        try:
            cm = infer_unit_costs(flat)
        except Exception as e:  # noqa: BLE001
            ok = False
            notes.append(f"{ds}: {e}")
            continue
        costs[ds] = (cm.c_fn, cm.c_fp, cm.c_tp_review)
        pos = {f["tp"] + f["fn"] for f in flat}
        if len(pos) != 1:
            ok = False
            notes.append(f"{ds}: TP+FN differs across methods {pos}")
        for r, f in zip(rows, flat):
            e = evaluate_costs(int(f["tp"]), int(f["fn"]), int(f["fp"]), cm)
            if not _close(e["total_loss"], _m(r["Total loss"]), 0.5) or not _close(e["savings_pct"], _m(r["Savings (%)"]), 0.006):
                ok = False
                notes.append(f"{ds} {r['Method']}: total/savings mismatch")
    out.append(Finding("A-17", "major" if not ok else "info", "inconsistent" if not ok else "consistent", "Supplementary S1 cost arithmetic",
                       "Unit costs inferred from sums/counts are constant within each dataset; total loss and savings = 1 - total/(P x c_FN) reproduce every row. "
                       f"Inferred (c_FN, c_FP, c_TP-review): {costs}. " + ("; ".join(notes) if notes else "")))
    bad = []
    for r in R.load("supplement_s2a")["rows"]:
        g = 100 * (_m(r["AUPRC protected age >= 50"]) - _m(r["AUPRC reference age < 50"]))
        if not _close(g, _m(r["Gap (pp)"]), 0.011):
            bad.append((r["Variant"], r["Method"]))
    for r in R.load("supplement_s2c")["rows"]:
        if not _close(_m(r["ECE protected"]) - _m(r["ECE reference"]), _m(r["ECE delta"]), 0.00011):
            bad.append(("ECE", r["Variant"], r["Method"]))
    out.append(Finding("A-18", "minor", "inconsistent" if bad else "consistent", "Supplementary S2 gaps and ECE deltas", f"Recomputed from the printed columns. {bad or 'All agree.'}"))
    n_cal_max_d1 = int(0.5 * 0.10 * 1_000_000)
    pub = max(_m(r["n_cal"]) for r in R.load("supplement_s4a")["rows"] if str(r["Dataset"]).startswith("D1"))
    out.append(Finding("A-19", "major", "inconsistent" if pub > n_cal_max_d1 else "consistent", "Supplementary S4a n_cal on D1 vs the described split",
                       f"The conformal calibration set is the second half of the chronological VALIDATION segment, which is the last 10% of the training months (Table 1). "
                       f"D1 has 1,000,000 rows in total, so that half is at most {n_cal_max_d1:,} rows; S4a prints n_cal={int(pub):,}. "
                       "Either n_cal was drawn differently from the described protocol or the split description is incomplete.", {"max_possible": n_cal_max_d1, "published": pub}))
    return out


def static_findings() -> list[Finding]:
    return [
        Finding("A-20", "minor", "note", "Equation 5 is typeset incorrectly",
                "As printed, 'reject' is assigned to P < T_high and the escalate branch reads T_low <= P >= T_high. Section 3.8.1 gives the intended rule "
                "(below T_low accept; above T_high reject; both endpoints in the review band). This repository implements the Section 3.8.1 rule."),
        Finding("A-21", "minor", "note", "LoRA learning rate typeset as '2 x 10^4'",
                "Section 3.3.1 prints 2 x 10^4 (a typesetting loss); Section 3.8.1 prints 0.0002. This repository uses 2e-4."),
        Finding("A-22", "minor", "note", "Feature counts",
                "Table 1/2 state 30 input features for D1 after excluding fraud_bool and protected attributes; the raw BAF base file has 30 features plus the label and month, "
                "so excluding the protected attribute leaves 29. The realised count is reported by `vrfraudnet data check`, never padded."),
        Finding("A-23", "minor", "note", "Wilcoxon statistic missing for one baseline", "Table 6 prints '-' for V of logistic regression (the maximum 465 is implied by p = 9.31e-10 = 2^-30)."),
    ]


def run_audit() -> list[Finding]:
    out: list[Finding] = []
    for fn in (a01_recall_ceiling, a02_table6, a03_temporal_tables, a04_table7_9, a05_supplement, static_findings):
        out += fn()
    return out


def to_markdown(findings: list[Finding]) -> str:
    lines = ["# Mathematical audit of the published article", "",
             "Generated by `vrfraudnet audit`. Every line recomputes a derived quantity from other PUBLISHED numbers; nothing is corrected, tuned or hidden.", "",
             "| ID | Severity | Status | Finding |", "|---|---|---|---|"]
    for f in findings:
        lines.append(f"| {f.id} | {f.severity} | {f.status} | {f.title} |")
    lines.append("")
    for f in findings:
        lines += [f"## {f.id}: {f.title}", "", f"*Severity: {f.severity}. Status: {f.status}.*", "", f.detail, ""]
    return "\n".join(lines)
