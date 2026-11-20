"""Baseline smoke tests and the end-to-end integration test: raw synthetic records -> scoring -> routing -> calibration -> audit -> report."""
import json

import numpy as np
import pytest

from vrfraudnet.baselines.registry import make_baseline
from vrfraudnet.data import splits as SP
from vrfraudnet.data.evidence import EvidenceBuilder
from vrfraudnet.data.providers import preprocessor_for
from vrfraudnet.data.synthetic import make_synthetic
from vrfraudnet.errors import FixtureInResearchModeError, MissingInputError, ProtocolError
from vrfraudnet.evaluation.metrics import auprc, roc_auc
from vrfraudnet.models.mixer import VerifiedMixer
from vrfraudnet.models.pipeline import Ablation, Case, VRFraudNetPipeline
from vrfraudnet.models.rationale import RuleFixtureProvider
from vrfraudnet.published import BASELINES
from vrfraudnet.retrieval.index import Exemplar, ExemplarIndex, HashingEmbedder
from vrfraudnet.types import Route, TransactionResult, VerifierState


@pytest.fixture(scope="module")
def prepared():
    ds = make_synthetic(3000, seed=0)
    m = SP.make_split("SYN", ds.frame)
    f = ds.frame
    pp = preprocessor_for(ds)
    tr = f.iloc[m.partitions["train"]]
    X = {"train": pp.fit_transform_train(tr, tr["__label"].to_numpy())}
    for k in ("val_select", "val_calibration", "test"):
        X[k] = pp.transform(f.iloc[m.partitions[k]])
    y = {k: f["__label"].to_numpy()[m.partitions[k]] for k in ("train", "val_select", "val_calibration", "test")}
    return ds, m, X, y


@pytest.mark.parametrize("name", BASELINES)
def test_every_baseline_runs_and_ranks_better_than_chance(prepared, name):
    if name in ("xgboost", "lightgbm"):
        pytest.importorskip(name)
    if name in ("mlp", "cnn1d", "lstm", "tabtransformer"):
        pytest.importorskip("torch")
    _, _, X, y = prepared
    mdl = make_baseline(name, seed=0, epochs=4).fit(X["train"], y["train"], X["val_select"], y["val_select"])
    s = mdl.score(X["test"])
    assert s.shape == (len(y["test"]),) and np.isfinite(s).all()
    assert roc_auc(y["test"], s) > 0.55, name                           # orientation is fraud-high, including the negated Isolation Forest


def test_isolation_forest_is_negated_into_fraud_orientation(prepared):
    _, _, X, y = prepared
    m = make_baseline("isolation_forest", 0).fit(X["train"], y["train"])
    assert np.allclose(m.score(X["test"]), -m.est.score_samples(X["test"]))
    with pytest.raises(ValueError):
        make_baseline("nope")


def test_baseline_results_are_deterministic_per_seed(prepared):
    _, _, X, y = prepared
    a = make_baseline("random_forest", 7).fit(X["train"], y["train"]).score(X["test"])
    b = make_baseline("random_forest", 7).fit(X["train"], y["train"]).score(X["test"])
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------------------- the pipeline, one transaction at a time
@pytest.fixture(scope="module")
def pipe_parts(prepared):
    pytest.importorskip("lightgbm")
    from vrfraudnet.models.triage import TriageModel

    ds, m, X, y = prepared
    tm = TriageModel(params={"n_estimators": 80, "min_child_samples": 20, "early_stopping_rounds": 15}, seed=0).fit(X["train"], y["train"], X["val_select"], y["val_select"])
    p = {k: tm.predict_proba(X[k]) for k in ("val_select", "val_calibration", "test")}
    return ds, m, X, y, tm, p


def _cases(ds, m, part, p, builder, n=None):
    f = ds.frame
    pos = m.partitions[part][:n]
    return [Case(str(f["__id"].iloc[i]), float(pi), builder.snapshot(int(i))) for i, pi in zip(pos, p[:len(pos)])]


def test_end_to_end_routing_verification_calibration_and_serialisation(pipe_parts):
    ds, m, X, y, tm, p = pipe_parts
    builder = EvidenceBuilder(ds)
    from vrfraudnet.models.triage import select_thresholds
    th = select_thresholds(p["val_select"], y["val_select"], max_escalation=0.35, c_fn=300, c_fp=5, c_review=5)   # fitted on the SELECTION half only
    low, high = th["low"], th["high"]
    prov = RuleFixtureProvider()
    pipe0 = VRFraudNetPipeline(low=low, high=high, provider=prov, mode="demo")
    # fit mixer on the selection half, conformal on the calibration half (disjoint partitions, as in the paper)
    sel_cases = _cases(ds, m, "val_select", p["val_select"], builder)
    sel = pipe0.run_many(sel_cases)
    states = [r.verifier.value for r in sel]
    p_r = [r.rationale_probability_used if r.rationale_probability_used is not None else np.nan for r in sel]
    mixer = VerifiedMixer(min_stratum=30).fit([r.triage_probability for r in sel], states, p_r, y["val_select"])
    from vrfraudnet.calibration.conformal import SplitConformal
    cal_pipe = VRFraudNetPipeline(low=low, high=high, provider=prov, mode="demo", mixer=mixer)
    cal = cal_pipe.run_many(_cases(ds, m, "val_calibration", p["val_calibration"], builder))
    conf = SplitConformal(0.10).fit([r.risk_score for r in cal], y["val_calibration"])
    pipe = VRFraudNetPipeline(low=low, high=high, provider=prov, mode="demo", mixer=mixer, conformal=conf)
    res = pipe.run_many(_cases(ds, m, "test", p["test"], builder))
    routes = {r.route for r in res}
    assert Route.VERIFIED_AUTO in routes and (Route.ACCEPT in routes or Route.REJECT in routes), routes
    for r in res:
        if r.route in (Route.ACCEPT, Route.REJECT):
            assert r.verifier is VerifierState.NOT_INVOKED and r.verification is None and r.rationale is None   # common path: never invoked, not 'failed'
            assert r.rationale_probability_used is None
        if r.route is Route.VERIFIED_AUTO:
            assert r.verifier is VerifierState.ACCEPTED and r.verification.accepted and r.rationale_probability_used is not None
            assert r.score_provenance == "triage+verified_rationale"
        assert 0 <= r.risk_score <= 1 and r.conformal_set is not None
        assert TransactionResult.from_dict(json.loads(r.to_json())) == r                                    # lossless serialisation, states preserved
    score = np.array([r.risk_score for r in res])
    assert auprc(y["test"], score) > y["test"].mean() * 1.5
    assert conf.coverage([r.risk_score for r in res], y["test"])["empirical_coverage"] >= 0.80


def test_rejected_rationale_goes_to_a_human_and_its_probability_is_not_used(pipe_parts):
    ds, m, X, y, tm, p = pipe_parts
    builder = EvidenceBuilder(ds)
    case = Case("TX", 0.5, builder.snapshot(int(m.partitions["test"][3])))
    for beh in ("false_claim", "unknown_field", "bad_json", "truncated"):
        r = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(beh), mode="demo").run(case)
        assert r.route is Route.HUMAN_REVIEW and r.verifier is VerifierState.REJECTED and r.automated is False, beh
        assert r.rationale_probability_used is None and r.rationale is None and r.verification.reasons()
        assert r.score_provenance == "triage_only_uncalibrated"
    ok = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider("faithful"), mode="demo").run(case)
    assert ok.route is Route.VERIFIED_AUTO and ok.automated
    pipe = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="demo")
    assert pipe.decision(ok) in ("fraud", "legitimate")
    hum = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider("false_claim"), mode="demo").run(case)
    assert pipe.decision(hum) is None


def test_inclusive_band_endpoints_are_escalated(pipe_parts):
    ds, m, X, y, tm, p = pipe_parts
    ev = EvidenceBuilder(ds).snapshot(int(m.partitions["test"][0]))
    pipe = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="demo")
    assert pipe.run(Case("a", 0.2, ev)).route in (Route.VERIFIED_AUTO, Route.HUMAN_REVIEW)
    assert pipe.run(Case("b", 0.8, ev)).route in (Route.VERIFIED_AUTO, Route.HUMAN_REVIEW)
    assert pipe.run(Case("c", 0.199999, ev)).route is Route.ACCEPT and pipe.run(Case("d", 0.800001, ev)).route is Route.REJECT


def test_research_mode_refuses_fixtures_and_missing_providers_fail_early(pipe_parts):
    ds, m, X, y, tm, p = pipe_parts
    with pytest.raises(FixtureInResearchModeError):
        VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="research")
    idx = ExemplarIndex(HashingEmbedder()).build([])
    with pytest.raises(FixtureInResearchModeError):
        VRFraudNetPipeline(low=0.2, high=0.8, index=idx, mode="research")
    ev = EvidenceBuilder(ds).snapshot(int(m.partitions["test"][0]))
    with pytest.raises(MissingInputError):
        VRFraudNetPipeline(low=0.2, high=0.8, mode="research").run(Case("x", 0.5, ev))
    with pytest.raises(ProtocolError):
        VRFraudNetPipeline(low=0.8, high=0.2)
    with pytest.raises(ProtocolError):
        VRFraudNetPipeline(low=0.2, high=0.8, provider=None, mode="demo").run(Case("x", float("nan"), ev))
    with pytest.raises(ValueError):
        VRFraudNetPipeline(low=0.2, high=0.8, mode="weird")


def test_ablations_are_isolated_and_marked(pipe_parts):
    ds, m, X, y, tm, p = pipe_parts
    ev = EvidenceBuilder(ds).snapshot(int(m.partitions["test"][2]))
    with pytest.raises(FixtureInResearchModeError):                           # benchmark-grade ablations refuse fixtures just like research mode
        VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="ablation", ablation=Ablation("A1", use_verifier=False))
    with pytest.raises(ProtocolError):                                        # a protective component cannot be switched off in a normal mode
        VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="demo", ablation=Ablation("A1", use_verifier=False))
    a1 = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider("false_claim"), mode="demo_ablation", ablation=Ablation("A1", use_verifier=False))
    r = a1.run(Case("x", 0.5, ev))
    assert r.score_provenance.startswith("ABLATION:A1") and r.route is Route.VERIFIED_AUTO and r.verification is None      # the unsafe path, labelled
    default = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider("false_claim"), mode="demo").run(Case("x", 0.5, ev))
    assert default.route is Route.HUMAN_REVIEW                                  # the default policy never lets that rationale through
    a2 = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="demo_ablation", ablation=Ablation("A2", use_gate=False))
    assert a2.run(Case("x", 0.01, ev)).route in (Route.VERIFIED_AUTO, Route.HUMAN_REVIEW)     # no triage gate: everything goes to Stage 2
    a6 = VRFraudNetPipeline(low=0.2, high=0.8, mode="demo_ablation", ablation=Ablation("A6", use_llm=False))
    assert a6.run(Case("x", 0.5, ev)).rationale is None


def test_retrieval_demonstrations_reach_the_provider_and_never_leak_the_current_case(pipe_parts):
    ds, m, X, y, tm, p = pipe_parts
    builder = EvidenceBuilder(ds)
    seen = []

    class Spy(RuleFixtureProvider):
        def generate(self, ev, demos, p_triage, *, constrained=True):
            seen.append((ev.transaction_id, list(demos)))
            return super().generate(ev, demos, p_triage, constrained=constrained)

    rat = {"verdict": "uncertain", "probability": 0.5, "claims": [], "evidence_refs": []}
    f = ds.frame
    train = m.partitions["train"][:40]
    exs = [Exemplar(str(f["__id"].iloc[i]), float(f["__time"].iloc[i]), frozenset([f["__origin"].iloc[i]]), f"amount={f['amount'].iloc[i]}", rat, "review") for i in train]
    index = ExemplarIndex(HashingEmbedder()).build(exs)
    case = Case("T", 0.5, builder.snapshot(int(m.partitions["test"][0])))
    VRFraudNetPipeline(low=0.2, high=0.8, provider=Spy(), index=index, mode="demo").run(case)
    tid, demos = seen[0]
    assert demos and len(demos) <= 4
    assert all(d.timestamp < case.evidence.prediction_time and d.id != tid and not (d.entity_ids & set(case.evidence.entities.values())) for d in demos)
