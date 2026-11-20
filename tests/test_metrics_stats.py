import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from vrfraudnet.errors import ProtocolError
from vrfraudnet.evaluation import metrics as M
from vrfraudnet.evaluation import stats as S
from vrfraudnet.evaluation.cost import CostModel, evaluate_costs, infer_unit_costs, subgroup_report


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.1).astype(int)
    s = rng.normal(0, 1, 3000) + 1.2 * y
    return y, s


def test_auprc_matches_sklearn_average_precision(data):
    y, s = data
    assert M.auprc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert M.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert abs(M.auprc(y, s, "trapezoid") - M.auprc(y, s)) < 0.05      # conventions differ slightly, in either direction; neither dominates
    with pytest.raises(ValueError):
        M.auprc(y, s, "nope")


def test_auprc_ties_are_one_threshold():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert M.auprc(y, s) == pytest.approx(0.5)                     # one threshold: precision 0.5 at recall 1
    assert M.roc_auc(y, s) == pytest.approx(0.5)


def test_perfect_and_inverted_scores():
    y = np.array([0, 0, 1, 1])
    assert M.auprc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0 and M.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0               # orientation matters: an unnegated anomaly score would read as 0


def test_no_positive_window_raises():
    for fn in (M.auprc, M.recall_at_top_fraction):
        with pytest.raises(ProtocolError):
            fn([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ProtocolError):
        M.roc_auc([1, 1], [0.1, 0.2])
    with pytest.raises(ProtocolError):
        M.select_f1_threshold([0, 0], [0.1, 0.2])


def test_input_validation():
    with pytest.raises(ProtocolError):
        M.auprc([0, 1], [0.1])
    with pytest.raises(ProtocolError):
        M.auprc([0, 1], [0.1, float("nan")])
    with pytest.raises(ProtocolError):
        M.auprc([0, 2], [0.1, 0.2])


def test_top_k_is_a_ceiling_and_deterministic_with_ties():
    assert M.top_k_size(100) == 1 and M.top_k_size(101) == 2 and M.top_k_size(1000) == 10 and M.top_k_size(118108) == 1182
    s = np.array([0.5, 0.5, 0.5, 0.1])
    assert list(M.top_k_indices(s, 2)) == [0, 1]                   # ties broken toward the lower row index
    y = np.array([0, 1, 1, 0])
    worst, best = M.recall_at_top_fraction_bounds(y, s, 0.5)       # k = 2 over a 3-way tie containing both positives
    assert (worst, best) == (0.5, 1.0)
    assert M.recall_at_top_fraction(y, s, 0.5) == 0.5              # the deterministic rule lands inside the bounds


def test_recall_ceiling_formula_and_d3_scale():
    assert M.recall_ceiling(1000, 100, 0.01) == pytest.approx(0.1)
    assert M.recall_ceiling(1000, 5, 0.01) == 1.0
    assert M.recall_ceiling(118108, 4134, 0.01) == pytest.approx(1182 / 4134)
    rng = np.random.default_rng(0)
    y = (rng.random(10000) < 0.035).astype(int)
    s = y + rng.normal(0, 0.1, 10000)                              # near-perfect ranking
    r = M.recall_at_top_fraction(y, s)
    assert r <= M.recall_ceiling(10000, int(y.sum())) + 1e-12      # no ranking can exceed the ceiling


def test_f1_threshold_selection_and_frozen_application(data):
    y, s = data
    t = M.select_f1_threshold(y, s)
    tm = M.threshold_metrics(y, s, t)
    grid = [M.threshold_metrics(y, s, x)["f1"] for x in np.unique(s)[::25]]
    assert tm["f1"] >= max(grid) - 1e-12
    assert (tm["tp"], tm["fp"], tm["fn"], tm["tn"]) == M.confusion(y, s, t)
    assert M.mcc_from_counts(0, 0, 0, 5) == 0.0 and M.f1_from_counts(0, 0, 0) == 0.0


def test_largest_threshold_wins_f1_ties():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])                              # any threshold in (0.2, 0.8] gives F1 = 1
    assert M.select_f1_threshold(y, s) == 0.8


def test_ece_and_all_metrics(data):
    y, s = data
    p = 1 / (1 + np.exp(-s))
    assert 0 <= M.expected_calibration_error(y, p) <= 1
    assert M.expected_calibration_error([1, 0], [1.0, 0.0]) == 0.0
    out = M.all_metrics(y, s, M.select_f1_threshold(y, s))
    assert set(out) >= {"AUPRC", "ROC-AUC", "F1", "Recall@top-1%", "MCC"}
    assert M.prevalence(y) == pytest.approx(y.mean())


# ----------------------------------------------------------------------------------------- statistics
def test_wilcoxon_matches_scipy_and_known_extreme():
    d = np.arange(1, 31, dtype=float)                               # all positive, distinct: V = 465
    r = S.wilcoxon_one_sided(d)
    assert r["V"] == 465 and r["V_max"] == 465 and r["method"] == "exact"
    assert r["p_one_sided"] == pytest.approx(2.0 ** -30, rel=1e-9)
    assert S.exact_p_from_V(465, 30) == pytest.approx(2.0 ** -30)


def test_exact_tail_values_for_near_maximal_V():
    unit = 2.0 ** -30
    assert S.exact_p_from_V(464, 30) == pytest.approx(2 * unit)     # subsets summing to 464, 465: {1}-complement and {}
    assert S.exact_p_from_V(462, 30) == pytest.approx(5 * unit)
    assert S.implied_V_from_p(1.86e-9, 30) == 464


def test_wilcoxon_ties_and_zeros_fall_back_to_normal_approximation():
    r = S.wilcoxon_one_sided([0.0, 1, 1, 2, 3, -0.5, 4, 5])
    assert r["method"] == "normal_approx" and r["zeros_dropped"] == 1 and r["ties"]
    with pytest.raises(ProtocolError):
        S.wilcoxon_one_sided([0, 0, 0])
    with pytest.raises(ProtocolError):
        S.wilcoxon_one_sided([1, float("nan")])


def test_holm_step_down_and_cumulative_maximum():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert S.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    raw = [9.31e-10] * 6 + [1.86e-9, 4.66e-9, 1.40e-7]              # the published raw p-values of Table 6
    adj = S.holm(raw)
    assert adj[0] == pytest.approx(8.38e-9, rel=1e-3) and adj[7] == pytest.approx(9.32e-9, rel=1e-3) and adj[8] == pytest.approx(1.40e-7)


def test_effect_size_conventions_are_distinct():
    assert S.rank_biserial(465, 30) == 1.0 and S.rank_biserial(232.5, 30) == 0.0
    assert S.z_over_sqrt_n(465, 30) == pytest.approx(0.8, abs=0.05) or S.z_over_sqrt_n(465, 30) > 0.8


def test_pooled_comparison_runs_end_to_end():
    rng = np.random.default_rng(0)
    model = {ds: 0.6 + rng.normal(0, 0.002, 10) for ds in ("D1", "D2", "D3")}
    base = {"weak": {ds: model[ds] - 0.05 + rng.normal(0, 0.003, 10) for ds in model}, "same": {ds: model[ds] + rng.normal(0, 0.003, 10) for ds in model}}
    rows = S.compare_to_baselines(model, base)
    assert rows[0]["p_one_sided"] < 1e-6 < rows[1]["p_one_sided"]
    assert all(r["holm_adjusted_p"] >= r["p_one_sided"] for r in rows) and S.sample_sd([1, 2, 3]) == 1.0


# ----------------------------------------------------------------------------------------- cost and fairness
def test_cost_arithmetic_and_inference():
    cm = CostModel(300.0, 5.0, 5.0)
    e = evaluate_costs(tp=2255, fn=495, fp=4187, costs=cm)         # the paper's S1(a) VR-FraudNet counts
    assert e["total_loss"] == 180710 and e["savings_pct"] == pytest.approx(78.10, abs=0.005)
    assert cm.normalised() == (1.0, 5.0 / 300.0)
    rows = [{"tp": 1980, "fn": 770, "fp": 5940, "sum_fn_loss": 231000, "sum_fp_loss": 29700, "sum_tp_review": 9900},
            {"tp": 2255, "fn": 495, "fp": 4187, "sum_fn_loss": 148500, "sum_fp_loss": 20935, "sum_tp_review": 11275}]
    assert infer_unit_costs(rows) == cm
    rows[1]["sum_fp_loss"] = 1
    with pytest.raises(ProtocolError):
        infer_unit_costs(rows)
    with pytest.raises(ProtocolError):
        evaluate_costs(0, 0, 3, cm)


def test_subgroup_report(data):
    y, s = data
    rng = np.random.default_rng(2)
    prot = rng.random(len(y)) < 0.3
    p = 1 / (1 + np.exp(-s))
    r = subgroup_report(y, p, prot, threshold=0.5)
    assert r["gap_pp"] == pytest.approx(100 * (r["auprc_protected"] - r["auprc_reference"]))
    assert r["di_flag_rate_ratio"] > 0 and r["di_unflagged_ratio"] > 0
    with pytest.raises(ProtocolError):
        subgroup_report(y, p, np.zeros(len(y), bool))
