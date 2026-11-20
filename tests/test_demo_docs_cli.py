import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vrfraudnet import cli
from vrfraudnet import docs as D
from vrfraudnet.demo import BANNER, explain, run_demo
from vrfraudnet.errors import ProtocolError
from vrfraudnet.evaluation.latency import decode_plausibility, hardware_info, measure
from vrfraudnet.evaluation.robustness import applicable, score_variants, verifier_rejection_rate, wilson
from vrfraudnet.experiments import status as ST
from vrfraudnet.types import Route, TransactionResult, VerifierState

EX = Path(__file__).resolve().parents[1] / "examples"


def test_demo_shows_all_three_outcomes_and_labels_itself_a_fixture(tmp_path, capsys):
    out = tmp_path / "demo.json"
    payload = run_demo(out, seed=0)
    res = [TransactionResult.from_dict(d) for d in payload["results"]]
    assert [r.route for r in res] == [Route.ACCEPT, Route.VERIFIED_AUTO, Route.HUMAN_REVIEW, Route.HUMAN_REVIEW]
    assert res[0].verifier is VerifierState.NOT_INVOKED and res[1].verifier is VerifierState.ACCEPTED and res[2].verifier is VerifierState.REJECTED
    assert res[2].rationale_probability_used is None and res[2].verification.reasons()
    assert json.loads(out.read_text())["banner"] == BANNER and "SYNTHETIC FIXTURE" in capsys.readouterr().out
    assert all(r.transaction_id.startswith("TX_SYN_") for r in res)
    assert "claim audit" in explain(res[2]) and "NOT USED" in explain(res[2])


def test_demo_is_deterministic():
    a, b = run_demo(None, seed=1, quiet=True), run_demo(None, seed=1, quiet=True)
    assert a == b


def test_verify_cli_examples_accept_and_reject(capsys):
    assert cli.main(["verify", str(EX / "rationale_accepted.json"), str(EX / "evidence_wire.json")]) == 0
    assert "ACCEPTED" in capsys.readouterr().out
    assert cli.main(["verify", str(EX / "rationale_rejected.json"), str(EX / "evidence_wire.json")]) == 1
    out = capsys.readouterr().out
    assert "REJECTED" in out and "contradicted" in out


def test_cli_data_list_and_check_and_early_failures(tmp_path, capsys):
    assert cli.main(["data", "list"]) == 0
    out = capsys.readouterr().out
    assert "kaggle.com/datasets/sgpjesus" in out and "dgraph.xinye.com" in out and "NOTE:" in out
    assert cli.main(["data", "check", "--dataset", "D1", "--raw-dir", str(tmp_path)]) == 1          # reports problems, exit 1
    assert "missing file Base.csv" in capsys.readouterr().out
    assert cli.main(["data", "check"]) == 2                                                              # missing arguments: named error
    assert cli.main(["prepare", "--dataset", "D3", "--raw-dir", str(tmp_path), "--work", str(tmp_path / "w")]) == 2
    assert "train_transaction.csv" in capsys.readouterr().err
    assert cli.main(["run", "--dataset", "D1", "--fixture", "--runs", str(tmp_path)]) == 2             # the fixture provider is never allowed on a benchmark
    assert "only allowed for dataset SYN" in capsys.readouterr().err
    assert cli.main(["predict", "--dataset", "SYN", "--work", str(tmp_path / "empty"), "--fixture"]) == 2
    assert "prepare" in capsys.readouterr().err


def test_cli_audit_and_status_and_docs(tmp_path, capsys):
    assert cli.main(["audit", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "AUDIT.md").is_file() and "A-01D3" in (tmp_path / "audit.json").read_text()
    assert cli.main(["audit", "--strict"]) == 1                                                          # a blocking inconsistency exists: strict mode fails
    capsys.readouterr()
    assert cli.main(["status", "--out", str(tmp_path / "STATUS.md")]) == 0
    assert "blocked" in (tmp_path / "STATUS.md").read_text()
    assert cli.main(["docs", "--docs-dir", str(tmp_path)]) == 0
    ET.fromstring((tmp_path / "figures" / "pipeline.svg").read_text())                                 # well-formed SVG
    with pytest.raises(SystemExit):
        cli.main(["--version"])


def test_docs_stage_table_is_the_single_source_of_truth():
    assert [s["id"] for s in D.STAGES] == ["0", "1", "R", "2", "3", "4"]
    assert "Stage 4" in D.pipeline_svg() and "Fixed deterministic verifier" in D.stage_table_markdown()


def test_status_ledger_is_honest_about_what_is_blocked():
    ids = [i.id for i in ST.ITEMS]
    assert len(ids) == len(set(ids))
    assert all(i.blocked_by for i in ST.ITEMS if i.state == "blocked")                                  # a blocked item must name its blocker
    assert next(i for i in ST.ITEMS if i.id == "tab5c").blocked_by.count("impossible") == 1             # the audit finding travels with the item
    assert all(i.state != "benchmark_run" for i in ST.ITEMS if i.id.startswith("tab"))                  # nothing claims a benchmark run without run files
    root = Path(__file__).resolve().parents[1]
    import re
    for it in ST.ITEMS:                                                                                  # every test file the ledger names must exist
        for t in re.findall(r"tests/[A-Za-z0-9_]+\.py", it.test):
            assert (root / t).is_file(), (it.id, t)
    assert ST.detect_benchmark_runs(Path("does-not-exist")) == {}
    md = ST.render(None)
    assert md.count("\n| ") >= len(ST.ITEMS)


# ------------------------------------------------------------------------------------------ robustness scorer
def _variants(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fam in ("e1_split", "e2_delay", "e3_mule", "e5_feat"):
        for b in (1, 2):
            orig = rng.random(n)
            rows += [dict(original_id=f"T{i}", family=fam, budget=b, original_score=orig[i], variant_score=orig[i] - rng.random() * 0.3 * b) for i in range(n)]
    return pd.DataFrame(rows)


def test_score_variants_reports_denominator_and_inapplicable_cells():
    df = _variants()
    r = score_variants(df, 0.5, "D2", "detected_originals")
    c = r["cells"]["e1_split|B=1"]
    base = df[(df.family == "e1_split") & (df.budget == 1) & (df.original_score >= 0.5)]
    assert c["n"] == len(base) and c["evaded"] == int((base.variant_score < 0.5).sum()) and c["ci95"][0] <= c["rate"] <= c["ci95"][1]
    allf = score_variants(df, 0.5, "D2", "all_frauds")
    assert allf["cells"]["e1_split|B=1"]["n"] == 200 and allf["cells"]["e1_split|B=1"]["rate"] >= c["rate"] - 1e-9     # misses that predate the edit inflate it
    d1 = score_variants(df, 0.5, "D1")
    assert d1["cells"]["e3_mule|B=1"]["status"] == "not_applicable" and "no entity graph" in d1["cells"]["e3_mule|B=1"]["reason"]
    assert d1["families_scored_per_budget"]["B=1"] == 1 and "B=1" in d1["mean_over_applicable_families"]     # the mean is over applicable cells only
    assert applicable("D2", "e3_mule") == (True, None) and applicable("D3", "e3_mule")[0] is False


def test_robustness_input_validation_and_wilson_and_rejection_rate():
    with pytest.raises(ProtocolError):
        score_variants(pd.DataFrame({"a": [1]}), 0.5, "D2")
    with pytest.raises(ValueError):
        score_variants(_variants(5), 0.5, "D2", "weird")
    lo, hi = wilson(0, 100)
    assert lo == 0.0 and 0 < hi < 0.05 and np.isnan(wilson(0, 0)[0])
    r = verifier_rejection_rate([True] * 6 + [False] * 94, target=0.05)
    assert r["rate"] == 0.06 and r["meets_target"] is False and verifier_rejection_rate([False] * 100)["meets_target"] is True


def test_robustness_cli_scores_a_supplied_file(tmp_path, capsys):
    p = tmp_path / "v.csv"
    _variants(50).to_csv(p, index=False)
    assert cli.main(["robustness", str(p), "--dataset", "D2", "--threshold", "0.5"]) == 0
    assert "e1_split|B=1" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------ latency
def test_latency_records_hardware_and_never_claims_paper_comparability_on_a_laptop():
    r = measure(lambda i: 5, path="common", warmup=5, timed=40)
    assert r["n_timed"] == 40 and r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"] and r["generated_tokens_mean"] == 5.0
    assert r["protocol_match"]["batch_size_1"] and not r["protocol_match"]["warmup_ge_1000"] and not r["protocol_match"]["timed_ge_10000"]
    assert r["comparable_to_paper"] is False and "platform" in hardware_info()


def test_decode_plausibility_needs_token_counts():
    assert decode_plausibility(175.23, None) == {"checked": False, "reason": "generated-token count not reported"}
    r = decode_plausibility(175.23, 384)
    assert r["checked"] and r["min_plausible_ms"] == pytest.approx(3200.0) and r["plausible"] is False         # 384 tokens in 175 ms would need >2,000 tok/s
    assert decode_plausibility(175.23, 12)["plausible"] is True


def test_latency_cli_is_labelled_a_fixture_timing(capsys):
    assert cli.main(["latency", "--path", "common", "--warmup", "2", "--timed", "10"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "FIXTURE timing" in out["WARNING"] and out["comparable_to_paper"] is False
