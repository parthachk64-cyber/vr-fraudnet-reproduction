import copy
import json

import numpy as np
import pytest

from vrfraudnet import reported as R
from vrfraudnet.errors import MissingInputError, ProtocolError, SchemaError
from vrfraudnet.experiments import tables as T
from vrfraudnet.published import ABLATIONS, BASELINES
from vrfraudnet.schemas import load_schema, require_valid, validate_rationale
from vrfraudnet.training.manifest import git_revision, package_versions, sha256_file

OK = {"verdict": "fraud", "probability": 0.9, "claims": [{"type": "numeric_comparison", "field": "amount", "op": ">", "value": 1, "evidence_id": "E_AMT"}],
      "evidence_refs": ["E_AMT"]}


def mut(f):
    d = copy.deepcopy(OK)
    f(d)
    return d


BAD = {
    "not an object": [1, 2],
    "missing keys": {"verdict": "fraud"},
    "extra top key": mut(lambda d: d.update(x=1)),
    "bad verdict": mut(lambda d: d.update(verdict="maybe")),
    "prob > 1": mut(lambda d: d.update(probability=1.01)),
    "prob bool": mut(lambda d: d.update(probability=True)),
    "claims not list": mut(lambda d: d.update(claims="x")),
    "too many claims": mut(lambda d: d.update(claims=[OK["claims"][0]] * 9)),
    "refs not unique": mut(lambda d: d.update(evidence_refs=["E_AMT", "E_AMT"])),
    "ref bad chars": mut(lambda d: d.update(evidence_refs=["E AMT!"])),
    "claim not object": mut(lambda d: d.update(claims=[5])),
    "claim extra key": mut(lambda d: d["claims"][0].update(why="x")),
    "unknown type": mut(lambda d: d["claims"][0].update(type="magic")),
    "op for type": mut(lambda d: d["claims"][0].update(op="in")),
    "field uppercase": mut(lambda d: d["claims"][0].update(field="Amount")),
    "numeric string value": mut(lambda d: d["claims"][0].update(value="5")),
    "set value scalar": mut(lambda d: d["claims"][0].update(type="set_membership", op="in", value="x")),
    "graph value scalar": mut(lambda d: d["claims"][0].update(type="graph_path", op="path_exists", value="x")),
    "entity numeric": mut(lambda d: d["claims"][0].update(type="entity_match", op="==", value=5)),
    "list too long": mut(lambda d: d["claims"][0].update(type="set_membership", op="in", value=[1, 2, 3, 4, 5])),
    "list nested": mut(lambda d: d["claims"][0].update(type="set_membership", op="in", value=[[1]])),
    "long string": mut(lambda d: d["claims"][0].update(type="entity_match", op="==", value="x" * 70)),
}


def test_schema_file_and_validator_agree_on_every_case():
    jsonschema = pytest.importorskip("jsonschema")
    schema = load_schema()
    assert validate_rationale(OK) == [] and jsonschema.Draft202012Validator(schema).is_valid(OK)
    for name, obj in BAD.items():
        mine = validate_rationale(obj)
        theirs = not jsonschema.Draft202012Validator(schema).is_valid(obj)
        assert mine, f"validator accepted {name}"
        assert theirs, f"schema file accepted {name}"


def test_case_specific_constraints_and_require_valid():
    assert validate_rationale(OK, allowed_fields={"amount"}, allowed_evidence_ids={"E_AMT"}) == []
    assert any("not in the current" in e for e in validate_rationale(OK, allowed_fields={"other"}))
    assert any("not in the current" in e for e in validate_rationale(OK, allowed_evidence_ids={"E_OTHER"}))
    require_valid(OK)
    with pytest.raises(SchemaError):
        require_valid(BAD["bad verdict"])


# ------------------------------------------------------------------------------------------ builders on hand-written run files
def write_run(root, ds, seed, system, auprc, *, windows=None, ref=0.5, kind="synthetic_fixture", tag="", conformal=None, counts=None):
    d = root / ds / f"seed{seed}{tag}"
    d.mkdir(parents=True, exist_ok=True)
    r = {"system": system, "dataset": ds, "seed": seed, "provenance": {"kind": kind}, "metrics": {m: auprc for m in T.METRICS}, "reference": {"val_select_auprc": ref},
         "test_counts": counts or {"tp": 80, "fn": 20, "fp": 400, "tn": 9500}, "n_test": 10000}
    if windows:
        r["windows"] = windows
    if conformal:
        r["conformal"] = conformal
    (d / f"{system}.json").write_text(json.dumps(r), encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    rng = np.random.default_rng(0)
    for ds in ("D1", "D2", "D3"):
        for seed in (1, 2, 3):
            for b in BASELINES:
                write_run(tmp_path, ds, seed, b, 0.40 + 0.01 * BASELINES.index(b) + rng.normal(0, 0.002))
            write_run(tmp_path, ds, seed, "vr_fraudnet", 0.60 + rng.normal(0, 0.002),
                      conformal={"0.05": {"empirical_miscoverage": 0.04 + 0.001 * seed, "n_cal": 500}, "0.1": {"empirical_miscoverage": 0.09, "n_cal": 500}})
    for a in ABLATIONS:
        if a != "A8":
            for seed in (1, 2, 3):
                write_run(tmp_path, "D1", seed, f"vr_fraudnet_{a}", 0.55 + 0.001 * seed)
    return tmp_path


def test_table6_pairs_by_seed_and_applies_holm(runs):
    t = T.build_table6(runs, seeds=(1, 2, 3), allow_fixture=True)
    assert t["n_pairs"] == 9 and len(t["rows"]) == 9 and t["fixture"]
    assert all(r["holm_adjusted_p"] >= r["p_one_sided"] for r in t["rows"])
    assert t["rows"][0]["mean_delta_auprc_pp"] > 15                       # VR-FraudNet (~0.60) vs logistic regression (~0.40)


def test_table7_deltas_are_relative_to_the_full_model(runs):
    t = T.build_table7(runs, "D1", (1, 2, 3), allow_fixture=True)
    full = next(r for r in t["rows"] if r["id"] == "Full")
    assert full["delta_vs_full_pp"] == pytest.approx(0.0)
    a1 = next(r for r in t["rows"] if r["id"] == "A1")
    assert a1["delta_vs_full_pp"] == pytest.approx(100 * (a1["AUPRC"]["mean"] - full["AUPRC"]["mean"])) and a1["delta_vs_full_pp"] < -3


def test_table8_windows_state_their_reference_and_refuse_empty_windows(runs):
    win = lambda a6, a7: {"month_6": {"AUPRC": a6, "n": 100}, "month_7": {"AUPRC": a7, "n": 100}}
    for s in ("lightgbm", "tabtransformer", "vr_fraudnet"):
        for seed in (1, 2):
            write_run(runs, "D1", seed, s, 0.50, windows=win(0.48, 0.45), ref=0.52)
    t = T.build_table8_windows(runs, "D1", ["lightgbm"], (1, 2), reference="primary_test", allow_fixture=True)
    assert t["reference_definition"] == "primary_test" and t["rows"][0]["delta_month_6_pp"] == pytest.approx(-2.0) and t["rows"][0]["delta_month_7_pp"] == pytest.approx(-5.0)
    t2 = T.build_table8_windows(runs, "D1", ["lightgbm"], (1, 2), reference="val_select", allow_fixture=True)
    assert t2["rows"][0]["delta_month_6_pp"] == pytest.approx(-4.0)       # the same run, a different STATED reference
    write_run(runs, "D1", 1, "lightgbm", 0.5, windows={"month_6": {"status": "no_positives_in_window", "n": 3}, "month_7": {"AUPRC": 0.4, "n": 9}})
    with pytest.raises(ProtocolError):
        T.build_table8_windows(runs, "D1", ["lightgbm"], (1, 2), allow_fixture=True)
    with pytest.raises(ValueError):
        T.build_table8_windows(runs, "D1", ["lightgbm"], (1,), reference="whatever")


def test_table8b_expanding_windows(runs):
    qs = (0.60, 0.70, 0.80, 0.85, 0.90)
    for q, v in zip(qs, (0.60, 0.65, 0.70, 0.69, 0.68)):
        for seed in (1, 2):
            write_run(runs, "D3", seed, "lightgbm", v, tag=f"_q{q:.2f}")
    t = T.build_table8b(runs, ["lightgbm"], (1, 2), allow_fixture=True)
    row = t["rows"][0]
    assert row["mean_across_folds_descriptive"] == pytest.approx(np.mean([0.60, 0.65, 0.70, 0.69, 0.68])) and row["late_delta_f5_minus_f3_pp"] == pytest.approx(-2.0)


def test_s4a_and_benchmark_kind_is_accepted_without_a_watermark(runs):
    t = T.build_s4a(runs, "D1", (1, 2, 3), allow_fixture=True)
    assert t["alphas"]["0.05"]["empirical_miscoverage"]["mean"] == pytest.approx(0.042) and t["alphas"]["0.05"]["n_cal"] == 500
    write_run(runs, "D2", 9, "lightgbm", 0.5, kind="benchmark")
    t5 = T.build_table5(runs, "D2", seeds=(9,), systems=["lightgbm"])
    assert not t5["fixture"] and t5["rows"][0]["AUPRC"]["sd"] is None            # one seed: SD is undefined and says so, instead of 0.0
    assert "SYNTHETIC" not in T.to_markdown(t5)


# ------------------------------------------------------------------------------------------ small utilities
def test_manifest_helpers(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert package_versions()["python"] and git_revision(tmp_path) in (None, git_revision(tmp_path))


def test_reported_loader_error_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("VRFRAUDNET_REFERENCES", str(tmp_path / "nowhere"))
    monkeypatch.setattr(R, "__file__", str(tmp_path / "a" / "b" / "reported.py"))
    monkeypatch.chdir(tmp_path)
    R.references_dir.__wrapped__ if hasattr(R.references_dir, "__wrapped__") else None
    with pytest.raises(MissingInputError) as e:
        R.references_dir()
    assert "VRFRAUDNET_REFERENCES" in str(e.value)
    assert R.mean({"mean": 1.5, "sd": 0.1}) == 1.5 and R.mean({"value": 2.0}) == 2.0 and R.mean(3) == 3.0
    with pytest.raises(KeyError):
        R.row([{"a": 1}], "a", 2)
