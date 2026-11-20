import copy
import json

import pytest

from conftest import T0, claim
from vrfraudnet.errors import LeakageError
from vrfraudnet.types import VerifierState
from vrfraudnet.verifier import EvidenceSnapshot, GraphEvidence, verify, verify_text


def test_accepts_fully_grounded_rationale(evidence, good_rationale):
    r = verify(good_rationale, evidence)
    assert r.accepted and r.state is VerifierState.ACCEPTED
    assert r.reasons() == [] and all(c.passed for c in r.claims) and len(r.claims) == 5


def test_numeric_claim_failures(evidence, good_rationale):
    bad = copy.deepcopy(good_rationale)
    bad["claims"][0]["value"] = 20000          # amount >= 20000 is false for amount 9800
    r = verify(bad, evidence)
    assert not r.accepted and r.claims[0].status == "contradicted"
    missing = copy.deepcopy(good_rationale)
    missing["claims"][0].update(field="account_age_days", evidence_id="E_AGE", op="<", value=10)
    missing["evidence_refs"].append("E_AGE")
    assert verify(missing, evidence).claims[0].passed
    only = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": ["E_PAY"],
            "claims": [claim("numeric_comparison", "payment_type", ">", 1, "E_PAY")]}
    assert verify(only, evidence).claims[0].status == "malformed"      # categorical evidence is not numeric


def test_set_membership_not_in(evidence):
    ok = {"verdict": "uncertain", "probability": 0.4, "evidence_refs": ["E_PAY"],
          "claims": [claim("set_membership", "payment_type", "not_in", ["cash"], "E_PAY")]}
    assert verify(ok, evidence).accepted
    wrong = copy.deepcopy(ok)
    wrong["claims"][0]["op"] = "in"
    assert verify(wrong, evidence).claims[0].status == "contradicted"


def test_temporal_within_before_after(evidence):
    def run(op, val, field="last_transfer_time", eid="E_LAST"):
        r = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": [eid], "claims": [claim("temporal_relation", field, op, val, eid)]}
        return verify(r, evidence).claims[0]
    assert run("within", 3600).passed and not run("within", 60).passed
    assert run("after", "account_opened_time").passed
    assert not run("before", "account_opened_time").passed
    assert run("after", "no_such_event").status == "unverifiable"
    assert run("within", -5).status == "malformed"


def test_graph_path_exists_absent_and_unknown(evidence):
    def run(op, tgt, hops, ev=evidence):
        r = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": ["E_ORG"],
             "claims": [claim("graph_path", "origin", op, [tgt, hops], "E_ORG")]}
        return verify(r, ev).claims[0]
    assert run("path_exists", "ACC_SYN_900", 2).passed
    assert run("path_exists", "ACC_SYN_900", 1).status == "contradicted"
    assert run("path_absent", "ACC_SYN_555", 3).passed
    incomplete = EvidenceSnapshot(**{**evidence.__dict__, "graph": GraphEvidence(evidence.graph.edges, complete=False, prediction_time=T0)})
    assert run("path_absent", "ACC_SYN_555", 3, incomplete).status == "unverifiable"      # absence is NOT proven
    no_graph = EvidenceSnapshot(**{**evidence.__dict__, "graph": None})
    assert run("path_exists", "ACC_SYN_900", 2, no_graph).status == "unverifiable"
    assert run("path_absent", "ACC_SYN_900", 2, no_graph).status == "unverifiable"
    assert run("path_exists", "ACC_SYN_900", 9).status == "malformed"                     # hop bound


def test_graph_budget_exhaustion_is_unknown_not_absent():
    edges = tuple(("N0", f"C{i}", 1.0) for i in range(50)) + (("C3", "TARGET", 1.0),)
    g = GraphEvidence(edges, complete=True, prediction_time=2.0)
    assert g.path("N0", "NOWHERE", 3, node_budget=5) is None          # wide graph: budget hit before the search finished
    assert g.path("N0", "NOWHERE", 3) is False                        # unbounded budget on a complete graph: provably absent
    assert g.path("N0", "TARGET", 3, node_budget=100) is True


def test_entity_match(evidence):
    r = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": ["E_BEN"],
         "claims": [claim("entity_match", "beneficiary", "!=", "ACC_SYN_900", "E_BEN")]}
    assert verify(r, evidence).claims[0].status == "contradicted"


def test_rejects_unknown_field_and_foreign_evidence_id(evidence, good_rationale):
    bad = copy.deepcopy(good_rationale)
    bad["claims"][0]["field"] = "invented_field"
    rep = verify(bad, evidence)
    assert not rep.accepted and any("not in the current transaction" in r for r in rep.rule_failures)
    bad2 = copy.deepcopy(good_rationale)
    bad2["claims"][0]["evidence_id"] = "E_RETRIEVED_EXEMPLAR"
    assert not verify(bad2, evidence).accepted


def test_evidence_id_must_document_the_claim_field(evidence, good_rationale):
    bad = copy.deepcopy(good_rationale)
    bad["claims"][0]["evidence_id"] = "E_PAY"
    assert any("does not document field" in r for r in verify(bad, evidence).rule_failures)


def test_claim_evidence_must_be_listed_in_refs(evidence, good_rationale):
    bad = copy.deepcopy(good_rationale)
    bad["evidence_refs"].remove("E_AMT")
    assert any("not listed in evidence_refs" in r for r in verify(bad, evidence).rule_failures)


def test_contradictory_and_duplicate_claims(evidence):
    base = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": ["E_AMT"]}
    c1 = claim("numeric_comparison", "amount", ">", 5000, "E_AMT")
    c2 = claim("numeric_comparison", "amount", "<", 100, "E_AMT")
    assert any("contradictory numeric" in r for r in verify({**base, "claims": [c1, c2]}, evidence).rule_failures)
    eq_ne = [claim("numeric_comparison", "amount", "==", 9800, "E_AMT"), claim("numeric_comparison", "amount", "!=", 9800, "E_AMT")]
    assert any("contradictory numeric" in r for r in verify({**base, "claims": eq_ne}, evidence).rule_failures)
    assert any("duplicate" in r for r in verify({**base, "claims": [c1, c1]}, evidence).rule_failures)
    e1 = claim("entity_match", "beneficiary", "==", "A", "E_BEN")
    e2 = claim("entity_match", "beneficiary", "!=", "A", "E_BEN")
    assert any("contradictory entity" in r for r in verify({**base, "evidence_refs": ["E_BEN"], "claims": [e1, e2]}, evidence).rule_failures)
    s1 = claim("set_membership", "payment_type", "in", ["wire"], "E_PAY")
    s2 = claim("set_membership", "payment_type", "not_in", ["wire"], "E_PAY")
    assert any("set-membership" in r for r in verify({**base, "evidence_refs": ["E_PAY"], "claims": [s1, s2]}, evidence).rule_failures)
    g1 = claim("graph_path", "origin", "path_exists", ["X", 2], "E_ORG")
    g2 = claim("graph_path", "origin", "path_absent", ["X", 2], "E_ORG")
    assert any("graph claims" in r for r in verify({**base, "evidence_refs": ["E_ORG"], "claims": [g1, g2]}, evidence).rule_failures)
    t1 = claim("temporal_relation", "last_transfer_time", "before", "account_opened_time", "E_LAST")
    t2 = claim("temporal_relation", "last_transfer_time", "after", "account_opened_time", "E_LAST")
    assert any("temporal claims" in r for r in verify({**base, "evidence_refs": ["E_LAST"], "claims": [t1, t2]}, evidence).rule_failures)


def test_verdict_consistency(evidence, good_rationale):
    no_claims = {"verdict": "fraud", "probability": 0.9, "claims": [], "evidence_refs": []}
    assert any("no claims support" in r for r in verify(no_claims, evidence).rule_failures)
    low_p = copy.deepcopy(good_rationale)
    low_p["probability"] = 0.2
    assert any("contradicts probability" in r for r in verify(low_p, evidence).rule_failures)
    legit_high = copy.deepcopy(good_rationale)
    legit_high["verdict"] = "legitimate"
    assert any("contradicts probability" in r for r in verify(legit_high, evidence).rule_failures)
    hints = {"amount": {"above": "legitimate"}}
    assert any("points to" in r for r in verify(good_rationale, evidence, direction_hints=hints).rule_failures)


def test_table4_style_rejection_reasons(evidence):
    """Synthetic analogue of the paper's rejected example E2."""
    rej = {"verdict": "legitimate", "probability": 0.1, "evidence_refs": ["E_AMT", "E_CNT", "E_ORG"],
           "claims": [claim("numeric_comparison", "amount", "<", 100, "E_AMT"),
                      claim("numeric_comparison", "recent_transfer_count", "<", 2, "E_CNT"),
                      claim("graph_path", "origin", "path_absent", ["ACC_SYN_900", 2], "E_ORG")]}
    r = verify(rej, evidence)
    assert not r.accepted
    assert [c.status for c in r.claims] == ["contradicted", "contradicted", "contradicted"]
    assert len(r.reasons()) >= 3


def test_unparseable_and_truncated_output_is_rejected_not_raised(evidence):
    assert not verify_text("{not json", evidence).accepted
    assert not verify_text('{"verdict": "fraud", "probab', evidence).accepted
    assert not verify_text("null", evidence).accepted
    assert verify_text(json.dumps({"verdict": "uncertain", "probability": 0.5, "claims": [], "evidence_refs": []}), evidence).accepted


def test_schema_violations_are_rejected(evidence, good_rationale):
    for mut in (lambda r: r.update(extra=1), lambda r: r.update(probability=1.5), lambda r: r.update(probability=float("nan")),
                lambda r: r.update(verdict="maybe"), lambda r: r["claims"][0].update(op="in"),
                lambda r: r["claims"][0].update(value="5000")):
        bad = copy.deepcopy(good_rationale)
        mut(bad)
        assert not verify(bad, evidence).accepted


def test_nine_claims_rejected(evidence, good_rationale):
    bad = copy.deepcopy(good_rationale)
    bad["claims"] = [claim("numeric_comparison", "amount", ">", i, "E_AMT") for i in range(9)]
    assert any("at most 8" in r for r in verify(bad, evidence).rule_failures)


def test_snapshot_refuses_future_events_and_future_edges():
    with pytest.raises(LeakageError):
        EvidenceSnapshot(transaction_id="T", prediction_time=10.0, event_times={"x": 11.0})
    with pytest.raises(LeakageError):
        GraphEvidence((("a", "b", 10.0),), prediction_time=10.0)     # an edge AT the prediction time is not 'before' it


def test_verifier_never_evaluates_claim_text(evidence):
    evil = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": ["E_AMT"],
            "claims": [claim("numeric_comparison", "amount", ">", "__import__('os').system('echo hi')", "E_AMT")]}
    assert not verify(evil, evidence).accepted


def test_absence_is_not_provable_beyond_the_covered_radius(evidence):
    g = GraphEvidence(evidence.graph.edges, complete=True, prediction_time=T0, radius=2)
    assert g.path("ACC_SYN_001", "NOWHERE", 2) is False        # proven absent within the covered radius
    assert g.path("ACC_SYN_001", "NOWHERE", 3) is None         # three hops were never expanded: unknown, not absent
