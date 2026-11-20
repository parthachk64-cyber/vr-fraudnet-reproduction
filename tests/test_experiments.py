"""The experiment layer on the SYNTHETIC fixture and on tiny provider-format files: plumbing, provenance, refusal rules."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import provider_files as PF  # noqa: E402

pytest.importorskip("lightgbm")
pytest.importorskip("torch")

from vrfraudnet import cli
from vrfraudnet.errors import MissingInputError, ProtocolError
from vrfraudnet.experiments import tables as T
from vrfraudnet.experiments.predictive import run_predictive
from vrfraudnet.experiments.runner import Context, plan, sweep
from vrfraudnet.models.pipeline import Ablation
from vrfraudnet.models.rationale import RuleFixtureProvider

FAST = dict(baseline_epochs=2, encoder_epochs=1, triage_params={"n_estimators": 40, "min_child_samples": 20, "early_stopping_rounds": 10})
SYSTEMS = ("logistic_regression", "lightgbm", "triage_only", "vr_fraudnet")


@pytest.fixture(scope="module")
def syn_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    for seed in (1, 2):
        run_predictive("SYN", None, seed, root, systems=SYSTEMS, provider=RuleFixtureProvider(), **FAST)
    return root


def test_synthetic_run_writes_manifests_and_is_stamped_a_fixture(syn_runs):
    d = syn_runs / "SYN" / "seed1"
    assert {p.name for p in d.iterdir()} >= {"summary.json", "predictions.npz", "vr_fraudnet.json", "lightgbm.json", "triage_only.json"}
    r = json.loads((d / "vr_fraudnet.json").read_text())
    prov = r["provenance"]
    assert prov["kind"] == "synthetic_fixture" and len(prov["split_fingerprint"]) == 64 and prov["packages"]["python"] and prov["config_hash"]
    assert prov["metric_definition"]["orientation"].startswith("1 = fraud")
    assert set(r["metrics"]) >= {"AUPRC", "ROC-AUC", "F1", "Recall@top-1%", "MCC"} and r["operating_threshold"] is not None
    assert sum(r["route_counts"].values()) == r["n_test"] and "0.05" in r["conformal"] and "0.1" in r["conformal"]
    assert r["thresholds"]["source"].startswith("fitted")                                     # SYN has no published thresholds
    assert any("reconstruction" in n for n in prov["notes"])                                   # the graph alignment policy is flagged
    p = np.load(d / "predictions.npz")
    assert len(p["y"]) == r["n_test"] and "vr_fraudnet" in p.files


def test_same_seed_gives_identical_results(tmp_path):
    a = run_predictive("SYN", None, 3, tmp_path / "a", systems=("lightgbm", "triage_only"), **FAST)
    b = run_predictive("SYN", None, 3, tmp_path / "b", systems=("lightgbm", "triage_only"), **FAST)
    assert a["systems"] == b["systems"]


def test_thresholds_are_frozen_from_validation_not_test(syn_runs):
    r = json.loads((syn_runs / "SYN" / "seed1" / "lightgbm.json").read_text())
    c = r["test_counts"]
    assert c["tp"] + c["fn"] == r["n_test_positive"] and c["tp"] + c["fn"] + c["fp"] + c["tn"] == r["n_test"]
    assert r["reference"]["val_select_auprc"] > 0


def test_tables_refuse_fixture_runs_unless_asked_and_then_watermark(syn_runs):
    with pytest.raises(ProtocolError):
        T.build_table5(syn_runs, "SYN", seeds=(1, 2), systems=["lightgbm"])
    t = T.build_table5(syn_runs, "SYN", seeds=(1, 2), systems=["lightgbm", "vr_fraudnet"], allow_fixture=True)
    assert t["fixture"] and t["rows"][0]["AUPRC"]["n"] == 2 and t["rows"][0]["AUPRC"]["sd"] is not None
    md = T.to_markdown(t)
    assert "SYNTHETIC FIXTURE RUN" in md
    row = next(r for r in t["rows"] if r["method"] == "LightGBM")
    vals = [json.loads((syn_runs / "SYN" / f"seed{s}" / "lightgbm.json").read_text())["metrics"]["AUPRC"] for s in (1, 2)]
    assert row["AUPRC"]["mean"] == pytest.approx(np.mean(vals)) and row["AUPRC"]["sd"] == pytest.approx(np.std(vals, ddof=1))   # values come from run files


def test_missing_seeds_or_systems_are_errors_naming_the_files(syn_runs):
    with pytest.raises(MissingInputError) as e:
        T.build_table5(syn_runs, "SYN", seeds=(1, 2, 3), systems=["lightgbm"], allow_fixture=True)
    assert "seed3/lightgbm.json" in str(e.value)
    with pytest.raises(MissingInputError):
        T.build_table6(syn_runs, datasets=("SYN",), seeds=(1, 2), allow_fixture=True)                 # the other baselines were not run
    with pytest.raises(MissingInputError):
        T.build_table7(syn_runs, "SYN", (1, 2), allow_fixture=True)                                    # no ablation runs exist


def test_conformal_and_cost_tables_come_from_run_files(syn_runs):
    t = T.build_s4a(syn_runs, "SYN", (1, 2), allow_fixture=True)
    assert set(t["alphas"]) == {"0.05", "0.1"} and t["alphas"]["0.05"]["n_cal"] > 100
    from vrfraudnet.evaluation.cost import CostModel
    s1 = T.build_s1(syn_runs, "SYN", CostModel(300, 5, 5), (1, 2), ["lightgbm"], allow_fixture=True)
    assert s1["rows"][0]["total_loss"]["n"] == 2 and "ASSUMPTION" in s1["note"]


def test_no_provider_means_vr_fraudnet_is_blocked_not_faked(tmp_path):
    s = run_predictive("SYN", None, 5, tmp_path, systems=("lightgbm", "vr_fraudnet"), **FAST)
    assert s["systems"]["vr_fraudnet"]["status"] == "blocked"
    r = json.loads((tmp_path / "SYN" / "seed5" / "summary.json").read_text())
    assert "lightgbm" in r["systems"] and not (tmp_path / "SYN" / "seed5" / "vr_fraudnet.json").exists()


def test_ablation_runs_are_marked_and_isolated(tmp_path):
    run_predictive("SYN", None, 6, tmp_path, systems=("lightgbm", "vr_fraudnet"), provider=RuleFixtureProvider(), ablation=Ablation("A6", use_llm=False), **FAST)
    f = tmp_path / "SYN" / "seed6" / "vr_fraudnet_A6.json"
    assert f.is_file() and json.loads(f.read_text())["provenance"]["config"]["ablation"] == "A6"
    assert not (tmp_path / "SYN" / "seed6" / "vr_fraudnet.json").exists()                  # an ablation never overwrites the full model


def test_d1_windows_use_the_real_split_and_are_in_the_run_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    PF.write_d1(raw, n_per_month=150)
    run_predictive("D1", raw, 0, tmp_path / "runs", systems=("lightgbm", "logistic_regression"), **FAST)
    r = json.loads((tmp_path / "runs" / "D1" / "seed0" / "lightgbm.json").read_text())
    assert set(r["windows"]) == {"month_6", "month_7"} and r["provenance"]["kind"] == "benchmark"      # tiny provider-format files count as the D1 adapter path
    assert r["windows"]["month_6"].get("n") == 150


def test_runner_plan_names_exactly_what_is_missing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    PF.write_d1(raw, n_per_month=40)
    ctx = Context(tmp_path / "runs", {"D1": raw}, None, None, None, (0,))
    rows = {r["id"]: r for r in plan(ctx)}
    assert rows["predictive-D1"]["baselines"] == "ready" and rows["predictive-D1"]["vr_fraudnet"] == "blocked"
    assert any("Llama" in m for m in rows["predictive-D1"]["missing_for_vr_fraudnet"]) and any("adapter" in m for m in rows["predictive-D1"]["missing_for_vr_fraudnet"])
    assert rows["predictive-D2"]["baselines"] == "blocked" and any("--raw-dir for D2" in m for m in rows["predictive-D2"]["missing_for_baselines"])
    log = sweep(ctx, ["predictive-D2", "predictive-D1"], dry_run=True)
    assert {"id": "predictive-D1", "seed": 0, "status": "would run", "vr_fraudnet": "blocked"} in log
    assert next(x for x in log if x["id"] == "predictive-D2")["status"] == "blocked"


def test_sweep_resumes_by_skipping_finished_seeds(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    PF.write_d1(raw, n_per_month=40)
    ctx = Context(tmp_path / "runs", {"D1": raw}, None, None, None, (0,))
    first = sweep(ctx, ["predictive-D1"], **FAST)
    assert [x["status"] for x in first] == ["done"] and "lightgbm" in first[0]["summary"]
    again = sweep(ctx, ["predictive-D1"], dry_run=True)
    assert again[0]["status"].startswith("skipped")


# ------------------------------------------------------------------------------------------ the stage workflow on SYN, via the CLI
def test_stage_workflow_on_synthetic_via_cli(tmp_path, capsys):
    w = ["--dataset", "SYN", "--work", str(tmp_path / "ws"), "--seed", "0"]
    assert cli.main(["prepare", *w]) == 0
    assert cli.main(["train-encoder", *w, "--epochs", "1"]) == 0
    assert cli.main(["train-triage", *w]) == 0
    assert cli.main(["build-index", *w, "--fixture"]) == 0
    assert cli.main(["fit-mixer", *w, "--fixture"]) == 0
    assert cli.main(["calibrate", *w, "--fixture"]) == 0
    assert cli.main(["predict", *w, "--fixture", "--limit", "60"]) == 0
    assert cli.main(["evaluate", *w]) == 0
    ws = tmp_path / "ws"
    ev = json.loads((ws / "evaluation.json").read_text())
    assert ev["n"] == 60 and ev["provenance"]["kind"] == "synthetic_fixture" and "AUPRC" in ev["metrics"]
    lines = (ws / "predictions_test.jsonl").read_text().splitlines()
    assert len(lines) == 60 and all(json.loads(l)["transaction_id"].startswith("TX_SYN_") for l in lines)
    idx = json.loads((ws / "index.json").read_text())
    assert idx["n_indexed"] > 0 and idx["index"]["is_fixture_embedder"] is True
    prep = json.loads((ws / "prepare.json").read_text())
    assert len(prep["fingerprint"]) == 64 and prep["realised_n_features"] > 0
    capsys.readouterr()
    assert cli.main(["build-index", "--dataset", "D1", "--work", str(tmp_path / "ws2"), "--fixture"]) == 2    # fixtures are SYN-only


def test_stage_order_is_enforced(tmp_path, capsys):
    w = ["--dataset", "SYN", "--work", str(tmp_path / "ws")]
    assert cli.main(["train-triage", *w]) == 2
    assert "run `prepare` first" in capsys.readouterr().err
    assert cli.main(["prepare", *w]) == 0
    assert cli.main(["fit-mixer", *w, "--fixture"]) == 2                                                       # triage artifacts missing
    assert "train-triage" in capsys.readouterr().err
