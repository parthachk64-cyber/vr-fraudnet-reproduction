import json
import random

import pytest

from vrfraudnet.models.grammar import (Alt, Cls, Grammar, Lit, Rep, TokenTrie, build_rationale_grammar, greedy_decode, seq)
from vrfraudnet.schemas import load_schema, validate_rationale
from vrfraudnet.verifier import verify_text


def test_regex_engine_basics():
    g = Grammar(seq("ab", Rep(Cls(frozenset("xy")), 1, 3), "!"))
    assert g.accepts("abx!") and g.accepts("abxyx!") and not g.accepts("ab!") and not g.accepts("abxyxy!")
    assert g.viable("abx") and not g.accepts("abx") and not g.viable("ba")
    g2 = Grammar(Rep(Lit("a"), 0, 2))
    assert g2.accepts("") and g2.accepts("aa") and not g2.accepts("aaa")


def test_known_good_rationale_is_in_the_language(evidence, good_rationale):
    g = build_rationale_grammar(evidence)
    from vrfraudnet.models.grammar import canonicalise
    canon = canonicalise(good_rationale)
    assert g.accepts(canon), canon
    assert not g.accepts(canon.replace('"fraud"', '"maybe"'))
    assert not g.accepts(canon.replace("amount", "invented"))
    assert not g.accepts(canon[:-1])                                   # truncated
    assert not g.accepts(canon.replace(",", ", "))                       # whitespace is outside the language
    dup_refs = canon.replace('"evidence_refs":[', '"evidence_refs":["E_AGE","E_AGE",')
    assert not g.accepts(dup_refs)                                         # repeated evidence ids are outside the language


def test_probability_range_enforced(evidence):
    g = build_rationale_grammar(evidence)
    pre = '{"verdict":"uncertain","probability":'
    suf = ',"claims":[],"evidence_refs":[]}'
    for ok in ("0", "0.5", "0.9999", "1", "1.0"):
        assert g.accepts(pre + ok + suf), ok
    for bad in ("1.5", "2", "-0.1", "01", ".5", "0.12345"):
        assert not g.accepts(pre + bad + suf), bad


def test_operator_type_coupling_and_claim_limit(evidence):
    g = build_rationale_grammar(evidence)
    pre = '{"verdict":"uncertain","probability":0.5,"claims":['
    num = '{"type":"numeric_comparison","field":"amount","op":">","value":1,"evidence_id":"E_AMT"}'
    bad = '{"type":"numeric_comparison","field":"amount","op":"in","value":1,"evidence_id":"E_AMT"}'
    assert g.viable(pre + num) and not g.viable(pre + bad)
    eight = pre + ",".join([num] * 8) + '],"evidence_refs":["E_AMT"]}'
    nine = pre + ",".join([num] * 9) + '],"evidence_refs":["E_AMT"]}'
    assert g.accepts(eight) and not g.viable(nine)


def test_evidence_id_must_match_field_in_the_grammar(evidence):
    g = build_rationale_grammar(evidence)
    wrong = '{"verdict":"uncertain","probability":0.5,"claims":[{"type":"numeric_comparison","field":"amount","op":">","value":1,"evidence_id":"E_PAY"'
    assert not g.viable(wrong)


class ToyVocab:
    """A byte-ish vocabulary: single characters plus a few multi-character pieces, as in a BPE tokenizer."""

    def __init__(self):
        chars = list('{}[]",:.0123456789-_ ') + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
        pieces = ['{"verdict":"', '","probability":', ',"claims":', ',"evidence_refs":', '"type":"', "numeric_comparison", "fraud", "legit", "uncertain",
                  "amount", "E_AMT", '"evidence_id":"', '":"', "[]", "0.", ">=", "path_exists"]
        self.id_to_text = {i: t for i, t in enumerate(chars + pieces)}
        self.eos = len(self.id_to_text)
        self.id_to_text[self.eos] = ""
        self.trie = TokenTrie(self.id_to_text)


def test_random_masked_decoding_always_yields_valid_rationales(evidence):
    """Whatever a (random) model prefers, masked greedy decoding ends in a parseable, schema-valid object."""
    g, v = build_rationale_grammar(evidence), ToyVocab()
    for seed in range(12):
        rng = random.Random(seed)
        logits = lambda text, rng=rng: {i: rng.random() for i in v.id_to_text}
        text, complete = greedy_decode(g, v.trie, v.id_to_text, v.eos, logits, max_new_tokens=400)
        if not complete:
            # an unlucky random walk can hit the token limit; it must still be a viable prefix, and the verifier must reject it
            assert g.viable(text)
            assert not verify_text(text, evidence).accepted
            continue
        obj = json.loads(text)
        assert validate_rationale(obj, allowed_fields=evidence.allowed_fields, allowed_evidence_ids=evidence.evidence_ids) == []


def test_masked_decoding_is_deterministic_and_greedy(evidence):
    g, v = build_rationale_grammar(evidence), ToyVocab()
    prefer = lambda text: {i: (1.0 if v.id_to_text[i] in ("0.", "fraud", "[]") else 0.0) for i in v.id_to_text}
    a = greedy_decode(g, v.trie, v.id_to_text, v.eos, prefer, 300)
    b = greedy_decode(g, v.trie, v.id_to_text, v.eos, prefer, 300)
    assert a == b


def test_eos_only_allowed_when_complete(evidence):
    g, v = build_rationale_grammar(evidence), ToyVocab()
    assert not g.accepting(g.start)
    allowed = v.trie.allowed(g, g.start)
    assert v.eos not in allowed and allowed
    text = '{"verdict":"uncertain","probability":0.5,"claims":[],"evidence_refs":[]}'
    done = g.advance(g.start, text)
    assert g.accepting(done) and v.trie.allowed(g, done) == []      # nothing may follow a complete object


def test_token_limit_leaves_incomplete_output_which_is_rejected(evidence):
    g, v = build_rationale_grammar(evidence), ToyVocab()
    text, complete = greedy_decode(g, v.trie, v.id_to_text, v.eos, lambda t: {i: 0.0 for i in v.id_to_text}, max_new_tokens=5)
    assert not complete
    assert not verify_text(text, evidence).accepted


def test_graph_claims_only_when_graph_evidence_exists(evidence):
    from vrfraudnet.verifier import EvidenceSnapshot
    no_graph = EvidenceSnapshot(**{**evidence.__dict__, "graph": None})
    g = build_rationale_grammar(no_graph)
    gp = '{"verdict":"uncertain","probability":0.5,"claims":[{"type":"graph_path"'
    assert not g.viable(gp)
    assert build_rationale_grammar(evidence).viable(gp)


def test_schema_file_and_grammar_agree_on_structure(evidence):
    jsonschema = pytest.importorskip("jsonschema")
    schema = load_schema()
    g = build_rationale_grammar(evidence)
    v = ToyVocab()
    rng = random.Random(7)
    for _ in range(6):
        text, complete = greedy_decode(g, v.trie, v.id_to_text, v.eos, lambda t: {i: rng.random() for i in v.id_to_text}, 400)
        if complete:
            jsonschema.validate(json.loads(text), schema)
