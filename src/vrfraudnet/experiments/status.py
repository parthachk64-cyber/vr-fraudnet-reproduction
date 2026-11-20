"""The status ledger: every figure, table, supplement and model stage of the paper, with where it lives, how to run it, and what is ACTUALLY known.

States:  implemented            code written, unit-tested, no end-to-end run on this item
         fixture_exercised      exercised end to end on the SYNTHETIC fixture (proves plumbing, not results)
         benchmark_run          a run on the real benchmark data exists under runs/ (detected from run manifests, never assumed)
         blocked                cannot run without a specific missing input (named)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BLOCK_D = {"D1": "BAF Base.csv (Kaggle, terms acceptance)", "D2": "AMLworld HI-Small_Trans.csv (Kaggle, terms acceptance)", "D3": "IEEE-CIS train_transaction.csv + train_identity.csv (Kaggle, rules acceptance)",
           "D4": "Elliptic++ files (git-disl/EllipticPlusPlus, cloud-hosted)", "D5": "DGraph-Fin dgraphfin.npz (dgraph.xinye.com registration)"}
LLM = "Meta-Llama-3.1-8B-Instruct weights (gated) + the authors' LoRA adapter (not distributed: docs/UPSTREAM.md) + a GPU"


@dataclass
class Item:
    id: str
    paper: str
    where: str
    test: str
    command: str
    state: str
    blocked_by: str = ""


ITEMS = [
    Item("fig1", "Figure 1 (problem setting, evidence levels)", "docs/ARCHITECTURE.md (original diagram)", "tests/test_demo_docs_cli.py", "vrfraudnet docs", "implemented"),
    Item("fig2", "Figure 2 (pipeline)", "docs/ARCHITECTURE.md + docs/figures/pipeline.svg (generated from STAGES)", "tests/test_demo_docs_cli.py", "vrfraudnet docs", "implemented"),
    Item("fig3", "Figure 3 / Stage 0 spectral encoder", "models/spectral.py; training/encoder_train.py", "tests/test_models.py", "vrfraudnet train-encoder", "fixture_exercised", "benchmark run needs D2/D4/D5 data"),
    Item("fig4", "Figure 4 / Stage 1 triage + gate", "models/triage.py", "tests/test_models.py, tests/test_baselines_pipeline.py", "vrfraudnet train-triage", "fixture_exercised", "benchmark run needs D1-D5 data"),
    Item("tab1", "Table 1 datasets and splits", "data/registry.py, data/splits.py; references/reported/table1.json", "tests/test_data.py", "vrfraudnet data check|prepare", "implemented", "real files needed to check"),
    Item("tab2", "Table 2 preprocessing and leakage controls", "data/transforms.py, data/providers.py", "tests/test_data.py", "vrfraudnet prepare", "implemented", "real files needed"),
    Item("tab3", "Table 3 five-stage summary", "docs/ARCHITECTURE.md", "tests/test_demo_docs_cli.py", "-", "implemented"),
    Item("tab4", "Table 4 verifier accepted/rejected examples", "verifier/; examples/walkthrough", "tests/test_verifier.py, tests/test_demo_docs_cli.py", "vrfraudnet demo", "fixture_exercised"),
    Item("stage2", "Stage 2 rationale LM (LoRA, grammar, retrieval)", "models/rationale.py, models/grammar.py, retrieval/, training/rationale_train.py", "tests/test_llm_pathway.py (tiny random Llama)", "vrfraudnet train-rationale", "blocked", LLM),
    Item("stage3", "Stage 3 deterministic verifier", "verifier/", "tests/test_verifier.py", "vrfraudnet demo", "fixture_exercised"),
    Item("stage4", "Stage 4 mixer + split-conformal", "models/mixer.py, calibration/conformal.py", "tests/test_calibration.py", "vrfraudnet fit-mixer|calibrate", "fixture_exercised"),
    Item("tab5a", "Table 5a (D1)", "experiments/predictive.py, experiments/tables.py", "tests/test_experiments.py", "vrfraudnet run --dataset D1 ...; vrfraudnet tables table5 --dataset D1", "blocked", BLOCK_D["D1"] + "; VR-FraudNet row also " + LLM),
    Item("tab5b", "Table 5b (D2)", "same", "same", "... --dataset D2", "blocked", BLOCK_D["D2"] + "; VR-FraudNet row also " + LLM),
    Item("tab5c", "Table 5c (D3)", "same", "same", "... --dataset D3", "blocked", BLOCK_D["D3"] + "; VR-FraudNet row also " + LLM + ". NOTE: the published Recall@top-1% column is arithmetically impossible (docs/AUDIT.md A-01D3)"),
    Item("tab6", "Table 6 (Wilcoxon, Holm)", "evaluation/stats.py, experiments/tables.py", "tests/test_metrics_stats.py", "vrfraudnet tables table6", "blocked", "ten seeds of Table 5 on D1-D3 (see tab5a-c)"),
    Item("tab7", "Table 7a-b (ablations A1-A9)", "models/pipeline.py (Ablation), experiments/predictive.py", "tests/test_baselines_pipeline.py", "vrfraudnet run --ablation A1 ...", "blocked", "D1/D2 data; " + LLM),
    Item("tab8a", "Table 8a (D1 month drift)", "experiments/predictive.py windows; tables.build_table8_windows", "tests/test_experiments.py", "vrfraudnet tables table8a", "blocked", BLOCK_D["D1"]),
    Item("tab8b", "Table 8b/d (D3 expanding windows)", "data/splits.d3_expanding_fold; tables.build_table8b", "tests/test_data.py", "vrfraudnet run --dataset D3 --q 0.60 ...", "blocked", BLOCK_D["D3"]),
    Item("tab8c", "Table 8c (D4 shock)", "data/splits.d4_shock_windows", "tests/test_data.py", "vrfraudnet tables table8c", "blocked", BLOCK_D["D4"] + "; the shock step is not stated by the paper"),
    Item("tab9", "Table 9a-c (evasion, budgets 1/2/4)", "evaluation/robustness.py (scores FIXED supplied files only)", "tests/test_demo_docs_cli.py", "vrfraudnet robustness score", "blocked", "the frozen edited-variant benchmark files (not distributed by the authors); the dataset behind Table 9 is not stated"),
    Item("s1", "Supplementary S1 (cost)", "evaluation/cost.py", "tests/test_metrics_stats.py", "vrfraudnet tables s1", "blocked", "D1-D3 runs; unit costs are recovered from the supplement itself (audit A-17)"),
    Item("s2", "Supplementary S2 (fairness)", "evaluation/cost.subgroup_report", "tests/test_metrics_stats.py", "vrfraudnet tables s2", "blocked", "BAF Base (Variants I/II/IV/V are separate Kaggle files); the DI convention is unstated"),
    Item("s3", "Supplementary S3 (efficiency, latency)", "evaluation/latency.py", "tests/test_demo_docs_cli.py", "vrfraudnet latency", "blocked", "an A100 80 GB host and " + LLM + "; CPU timings are never compared with the paper's"),
    Item("s4", "Supplementary S4 (coverage, rejection)", "calibration/conformal.py, evaluation/robustness.py", "tests/test_calibration.py", "vrfraudnet tables s4a", "blocked", "D1-D4 runs; the PromptInject v1.1 benchmark cases"),
    Item("audit", "Mathematical audit of the article", "evaluation/audit.py", "tests/test_audit_reported.py", "vrfraudnet audit", "benchmark_run", "runs on published numbers only; needs no data"),
]


def detect_benchmark_runs(runs_root: Path) -> dict:
    found = {}
    for p in Path(runs_root).glob("*/seed*/summary.json") if Path(runs_root).exists() else []:
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("kind") == "benchmark":
            found.setdefault(s["dataset"], []).append(s["seed"])
    return found


def render(runs_root: Optional[Path] = None) -> str:
    found = detect_benchmark_runs(runs_root) if runs_root else {}
    lines = ["# Status ledger", "",
             "Generated by `vrfraudnet status --markdown`. States: **implemented** (unit-tested only), **fixture_exercised** (run end to end on the SYNTHETIC fixture: plumbing, not results), "
             "**benchmark_run** (a run on real benchmark data is recorded under `runs/`), **blocked** (needs the named input).", "",
             f"Benchmark runs detected under `{runs_root}`: {found or 'none'}.", "", "| Item | Paper element | Where | Test | Command | State | Blocked by |", "|---|---|---|---|---|---|---|"]
    for it in ITEMS:
        state = it.state
        if it.id.startswith("tab5") and found.get(it.id[-1:].upper().replace("A", "D1").replace("B", "D2").replace("C", "D3")):
            state = "benchmark_run"
        lines.append(f"| {it.id} | {it.paper} | {it.where} | {it.test} | `{it.command}` | {state} | {it.blocked_by} |")
    return "\n".join(lines) + "\n"
