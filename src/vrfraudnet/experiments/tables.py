"""Table builders for REPRODUCED results. Every number comes from saved run files; nothing is read from references/reported.

A builder refuses to run (naming the exact missing run files) when the data it needs is absent, refuses `synthetic_fixture` runs unless
`allow_fixture=True` (and then watermarks the output), and never invents a seed value, tunes to a published number or fills a gap.
Seeds missing from a run set are an error, not a smaller table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..errors import MissingInputError, ProtocolError
from ..evaluation import stats as S
from ..evaluation.cost import CostModel, evaluate_costs
from ..published import ABLATIONS, BASELINES, SEEDS

METRICS = ("AUPRC", "ROC-AUC", "F1", "Recall@top-1%", "MCC")
LABEL = {"logistic_regression": "Logistic regression", "isolation_forest": "Isolation forest", "mlp": "MLP", "cnn1d": "1D-CNN", "lstm": "LSTM",
         "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "tabtransformer": "TabTransformer", "vr_fraudnet": "VR-FraudNet",
         "triage_only": "VR-FraudNet triage only"}
FIXTURE_BANNER = "SYNTHETIC FIXTURE RUN: exercises the code; NOT a reproduction of any published result."


def _run_path(root: Path, dataset: str, seed: int, system: str, tag: str = "") -> Path:
    return Path(root) / dataset / f"seed{seed}{tag}" / f"{system}.json"


def load_runs(root: Path, dataset: str, systems: Sequence[str], seeds: Sequence[int], *, tag: str = "", allow_fixture: bool = False) -> dict:
    missing, out, fixture = [], {}, False
    for sys_ in systems:
        for sd in seeds:
            p = _run_path(root, dataset, sd, sys_, tag)
            if not p.is_file():
                missing.append(f"{dataset}/seed{sd}{tag}/{sys_}.json")
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["provenance"]["kind"] == "synthetic_fixture":
                if not allow_fixture:
                    raise ProtocolError(f"{p} is a synthetic fixture run; reproduced tables refuse it (pass allow_fixture=True to build a watermarked exercise table)")
                fixture = True
            out.setdefault(sys_, {})[sd] = r
    if missing:
        raise MissingInputError(f"{len(missing)} run file(s) under {root}", needed_for="building this table",
                                how_to_fix="run `vrfraudnet run ...` for: " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))
    out["_fixture"] = fixture
    return out


def _ms(v: Sequence[float]) -> dict:
    a = np.asarray(v, float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "n": int(len(a))}


def build_table5(root: Path, dataset: str, *, seeds: Sequence[int] = SEEDS, systems: Optional[Sequence[str]] = None, allow_fixture: bool = False) -> dict:
    systems = list(systems or (list(BASELINES) + ["vr_fraudnet"]))
    runs = load_runs(root, dataset, systems, seeds, allow_fixture=allow_fixture)
    rows = [{"method": LABEL[s], **{m: _ms([runs[s][sd]["metrics"][m] for sd in seeds]) for m in METRICS}} for s in systems]
    return {"table": f"5 ({dataset}), reproduced", "seeds": list(seeds), "rows": rows, "fixture": runs["_fixture"], "sd": "sample SD (ddof=1) across seeds"}


def build_table6(root: Path, datasets: Sequence[str] = ("D1", "D2", "D3"), seeds: Sequence[int] = SEEDS, allow_fixture: bool = False) -> dict:
    systems = list(BASELINES) + ["vr_fraudnet"]
    per = {d: load_runs(root, d, systems, seeds, allow_fixture=allow_fixture) for d in datasets}
    model = {d: [per[d]["vr_fraudnet"][s]["metrics"]["AUPRC"] for s in seeds] for d in datasets}
    base = {b: {d: [per[d][b][s]["metrics"]["AUPRC"] for s in seeds] for d in datasets} for b in BASELINES}
    rows = S.compare_to_baselines(model, base)
    return {"table": "6, reproduced", "n_pairs": len(datasets) * len(seeds), "rows": rows, "fixture": any(per[d]["_fixture"] for d in datasets),
            "note": "seeds repeated on the same records are dependent; secondary sensitivity analysis only. Effect size is reported under both conventions."}


def build_table7(root: Path, dataset: str, seeds: Sequence[int] = SEEDS, allow_fixture: bool = False) -> dict:
    names = ["vr_fraudnet"] + [f"vr_fraudnet_{a}" for a in ABLATIONS if a not in ("A8",)]
    runs = load_runs(root, dataset, names, seeds, allow_fixture=allow_fixture)
    full = np.mean([runs["vr_fraudnet"][s]["metrics"]["AUPRC"] for s in seeds])
    rows = []
    for n in names:
        a = _ms([runs[n][s]["metrics"]["AUPRC"] for s in seeds])
        rows.append({"id": "Full" if n == "vr_fraudnet" else n.replace("vr_fraudnet_", ""), "AUPRC": a, "delta_vs_full_pp": 100 * (a["mean"] - full)})
    return {"table": f"7 ({dataset}), reproduced", "rows": rows, "fixture": runs["_fixture"]}


def build_table8_windows(root: Path, dataset: str, systems: Sequence[str], seeds: Sequence[int], reference: str = "primary_test", allow_fixture: bool = False) -> dict:
    """D1 month drift (8a) or D4 pre/post shock (8c). `reference` is stated, never implied: 'primary_test' (the held-out partition's AUPRC) or 'val_select'."""
    if reference not in ("primary_test", "val_select"):
        raise ValueError("reference must be 'primary_test' or 'val_select'")
    runs = load_runs(root, dataset, systems, seeds, allow_fixture=allow_fixture)
    wins = ("month_6", "month_7") if dataset == "D1" else ("pre_shock", "post_shock")
    rows = []
    for s in systems:
        ref = [runs[s][sd]["metrics"]["AUPRC"] if reference == "primary_test" else runs[s][sd]["reference"]["val_select_auprc"] for sd in seeds]
        row = {"method": LABEL.get(s, s), "reference": _ms(ref)}
        for w in wins:
            vals = []
            for sd in seeds:
                wr = runs[s][sd]["windows"][w]
                if "AUPRC" not in wr:
                    raise ProtocolError(f"{dataset} {s} seed{sd} window {w}: {wr.get('status')}")
                vals.append(wr["AUPRC"])
            row[w] = _ms(vals)
            row[f"delta_{w}_pp"] = 100 * (row[w]["mean"] - row["reference"]["mean"])
        rows.append(row)
    return {"table": f"8 ({dataset} windows), reproduced", "reference_definition": reference, "rows": rows, "fixture": runs["_fixture"]}


def build_table8b(root: Path, systems: Sequence[str], seeds: Sequence[int], qs=(0.60, 0.70, 0.80, 0.85, 0.90), allow_fixture: bool = False) -> dict:
    rows = []
    for s in systems:
        per_q = {}
        for q in qs:
            runs = load_runs(root, "D3", [s], seeds, tag=f"_q{q:.2f}", allow_fixture=allow_fixture)
            per_q[q] = _ms([runs[s][sd]["metrics"]["AUPRC"] for sd in seeds])
        means = [per_q[q]["mean"] for q in qs]
        rows.append({"method": LABEL.get(s, s), "folds": {f"{q:.2f}": per_q[q] for q in qs}, "mean_across_folds_descriptive": float(np.mean(means)),
                     "late_delta_f5_minus_f3_pp": 100 * (means[4] - means[2])})
    return {"table": "8b (D3 expanding windows), reproduced", "rows": rows, "note": "Fold 3 (q=0.80) is the primary holdout; the five-fold mean is descriptive only."}


def build_s1(root: Path, dataset: str, costs: CostModel, seeds: Sequence[int], systems: Sequence[str], allow_fixture: bool = False) -> dict:
    runs = load_runs(root, dataset, systems, seeds, allow_fixture=allow_fixture)
    rows = []
    for s in systems:
        per = [evaluate_costs(r["test_counts"]["tp"], r["test_counts"]["fn"], r["test_counts"]["fp"], costs) for r in (runs[s][sd] for sd in seeds)]
        rows.append({"method": LABEL.get(s, s), "tp": _ms([p["tp"] for p in per]), "fn": _ms([p["fn"] for p in per]), "fp": _ms([p["fp"] for p in per]),
                     "total_loss": _ms([p["total_loss"] for p in per]), "savings_pct": _ms([p["savings_pct"] for p in per])})
    return {"table": f"S1 ({dataset}), reproduced", "costs": costs.__dict__, "rows": rows, "fixture": runs["_fixture"],
            "note": "costs are an explicit experiment ASSUMPTION; where they come from (the supplement's implied unit costs, or yours) is recorded by the caller"}


def build_s4a(root: Path, dataset: str, seeds: Sequence[int], allow_fixture: bool = False) -> dict:
    runs = load_runs(root, dataset, ["vr_fraudnet"], seeds, allow_fixture=allow_fixture)
    out = {}
    for a in ("0.05", "0.1"):
        mis = [runs["vr_fraudnet"][s]["conformal"][a]["empirical_miscoverage"] for s in seeds]
        out[a] = {"target": float(a), "empirical_miscoverage": _ms(mis), "n_cal": runs["vr_fraudnet"][seeds[0]]["conformal"][a]["n_cal"]}
    return {"table": f"S4a ({dataset}), reproduced", "alphas": out, "fixture": runs["_fixture"]}


def to_markdown(t: dict) -> str:
    head = [f"### {t['table']}", ""]
    if t.get("fixture"):
        head += [f"> **{FIXTURE_BANNER}**", ""]
    body = ["```json", json.dumps({k: v for k, v in t.items() if k not in ("table", "fixture")}, indent=1, default=float), "```"]
    return "\n".join(head + body)
