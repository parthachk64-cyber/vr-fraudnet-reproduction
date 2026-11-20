import json

import numpy as np
import pytest

from vrfraudnet.errors import FixtureInResearchModeError, LeakageError, SchemaError
from vrfraudnet.retrieval.index import Exemplar, ExemplarIndex, HashingEmbedder, Query, require_research_embedder, summarise

RAT = {"verdict": "uncertain", "probability": 0.5, "claims": [], "evidence_refs": []}


def ex(i, text, ts=1.0, ents=(), **kw):
    return Exemplar(id=f"X{i}", timestamp=ts, entity_ids=frozenset(ents), summary=text, rationale=RAT, band="review", **kw)


@pytest.fixture
def idx():
    items = [ex(0, "wire amount=9000 new account", 10), ex(1, "wire amount=9100 new account", 20), ex(2, "ach amount=20 old account", 30),
             ex(3, "wire amount=9050 new account", 40, ents=["ACC_A"]), ex(4, "card amount=5 old", 50)]
    return ExemplarIndex(HashingEmbedder(), "flat").build(items)


def test_only_verified_schema_valid_train_exemplars_can_be_indexed():
    emb = HashingEmbedder()
    with pytest.raises(LeakageError):
        ExemplarIndex(emb).build([ex(0, "a", partition="validation")])
    with pytest.raises(LeakageError):
        ExemplarIndex(emb).build([ex(0, "a", partition="test")])
    with pytest.raises(LeakageError):
        ExemplarIndex(emb).build([ex(0, "a", verified=False)])
    bad = Exemplar("X", 1.0, frozenset(), "a", {"verdict": "fraud"}, "review")
    with pytest.raises(SchemaError):
        ExemplarIndex(emb).build([bad])


def test_nearest_eligible_not_just_nearest(idx):
    q = Query("Q", "wire amount=9050 new account", timestamp=100, entity_ids=frozenset(["ACC_A"]))
    got = [e.id for e in idx.retrieve(q, k=2)]
    assert "X3" not in got                         # shared entity with the query: excluded even though it is the closest text
    assert len(got) == 2 and got[0] in ("X0", "X1")


def test_eligibility_is_applied_before_top_k_so_the_result_is_not_shortened(idx):
    q = Query("Q", "wire amount=9050 new account", timestamp=100, entity_ids=frozenset(["ACC_A"]))
    assert len(idx.retrieve(q, k=4)) == 4          # 5 exemplars, 1 ineligible: still four returned


def test_timestamp_rule_is_strict(idx):
    q = Query("Q", "wire amount=9000 new account", timestamp=30)
    ids = {e.id for e in idx.retrieve(q, k=5)}
    assert ids == {"X0", "X1"}                     # X2 is AT the query time (not before it); X3, X4 are later: all ineligible


def test_same_transaction_excluded_and_zero_or_few_results_are_valid(idx):
    q = Query("X0", "wire amount=9000 new account", timestamp=1000)
    assert "X0" not in {e.id for e in idx.retrieve(q, 5)}
    assert idx.retrieve(Query("Q", "x", timestamp=0.5), 4) == []          # nothing precedes the query: empty, not back-filled
    assert ExemplarIndex(HashingEmbedder()).build([]).retrieve(Query("Q", "x"), 4) == []


def test_missing_exemplar_timestamp_is_ineligible_when_query_has_one():
    i = ExemplarIndex(HashingEmbedder()).build([ex(0, "a", ts=None)])
    assert i.retrieve(Query("Q", "a", timestamp=5), 3) == []
    assert len(i.retrieve(Query("Q", "a", timestamp=None), 3)) == 1        # a dataset without clocks: timestamp rule does not apply


def test_retrieval_is_deterministic_and_ranked(idx):
    q = Query("Q", "wire amount=9000 new account", timestamp=100)
    a = [e.id for e in idx.retrieve(q, 4)]
    assert a == [e.id for e in idx.retrieve(q, 4)]
    assert a[0] == "X0"                                                     # an identical summary is the nearest


def test_flat_and_faiss_agree():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    items = [ex(i, f"wire amount={int(rng.integers(1, 99999))} acct={i % 7}", ts=float(i), ents=[f"E{i % 11}"]) for i in range(120)]
    flat = ExemplarIndex(HashingEmbedder(), "flat").build(items)
    fa = ExemplarIndex(HashingEmbedder(), "faiss").build(items)
    for j in range(15):
        q = Query(f"Q{j}", f"wire amount={int(rng.integers(1, 99999))} acct={j % 7}", timestamp=float(60 + j), entity_ids=frozenset([f"E{j % 11}"]))
        assert [e.id for e in flat.retrieve(q, 4)] == [e.id for e in fa.retrieve(q, 4)]


def test_save_load_roundtrip_and_tamper_detection(idx, tmp_path):
    man = idx.save(tmp_path / "ix")
    assert man["n"] == 5 and man["is_fixture_embedder"] is True
    again = ExemplarIndex.load(tmp_path / "ix", HashingEmbedder())
    q = Query("Q", "wire amount=9000 new account", timestamp=100)
    assert [e.id for e in again.retrieve(q, 3)] == [e.id for e in idx.retrieve(q, 3)]
    (tmp_path / "ix" / "exemplars.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(LeakageError):
        ExemplarIndex.load(tmp_path / "ix", HashingEmbedder())


def test_embedder_mismatch_and_research_refusal(idx, tmp_path):
    idx.save(tmp_path / "ix")
    with pytest.raises(ValueError):
        ExemplarIndex.load(tmp_path / "ix", HashingEmbedder(128))
    with pytest.raises(FixtureInResearchModeError):
        require_research_embedder(HashingEmbedder())
    with pytest.raises(ValueError):
        ExemplarIndex(HashingEmbedder(), "nope")


def test_summaries_use_one_text_space():
    assert summarise({"amount": 5, "x": "y"}, "review", 0.5) == summarise({"x": "y", "amount": 5}, "review", 0.5)
