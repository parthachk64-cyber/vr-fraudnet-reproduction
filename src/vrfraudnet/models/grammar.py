"""Grammar-constrained decoding for the Stage 2 rationale (Sec. 3.3.3).

The permitted output is a REGULAR language: compact JSON, at most 8 claims, field names and evidence IDs drawn from
the CURRENT transaction's evidence, operators coupled to claim types. It is compiled once per case into an NFA and
advanced character by character. For a language-model vocabulary a trie walk yields the exact set of tokens that
keep the output viable, so a completed generation always parses and always satisfies the schema's structure.

What the grammar does NOT do (and the paper says the same): it does not make a claim true. Whether a claim is
supported by the evidence is the verifier's job (Stage 3). Greedy decoding with this mask is what the paper
specifies; an output cut off by the token limit is incomplete and is rejected downstream, never repaired.
Whitespace is excluded from the language (compact JSON), a deliberate simplification (docs/DECISIONS.md D-10).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Union

from ..verifier.evidence import EvidenceSnapshot

# ----------------------------------------------------------------------------- regex AST


@dataclass(frozen=True)
class Lit:
    s: str


@dataclass(frozen=True)
class Cls:
    chars: frozenset


@dataclass(frozen=True)
class Seq:
    parts: tuple


@dataclass(frozen=True)
class Alt:
    options: tuple


@dataclass(frozen=True)
class Rep:
    node: object
    lo: int
    hi: int


@dataclass(frozen=True)
class OrderedSubset:
    """A comma-separated strictly-increasing selection of `items` (each already a quoted literal), at most `max_items`.
    Strict ordering makes the items unique without leaving the regular languages."""

    items: tuple
    max_items: int = 8


Node = Union[Lit, Cls, Seq, Alt, Rep, OrderedSubset]
DIGITS = frozenset("0123456789")
NONZERO = frozenset("123456789")
IDCHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:.-")


def seq(*p) -> Seq:
    return Seq(tuple(Lit(x) if isinstance(x, str) else x for x in p))


def alt(*o) -> Alt:
    return Alt(tuple(Lit(x) if isinstance(x, str) else x for x in o))


def opt(n) -> Rep:
    return Rep(n, 0, 1)


# ----------------------------------------------------------------------------- NFA


class _NFA:
    def __init__(self):
        self.eps: list[list[int]] = []
        self.edges: list[dict[str, list[int]]] = []

    def new(self) -> int:
        self.eps.append([])
        self.edges.append({})
        return len(self.eps) - 1

    def compile(self, n: Node) -> tuple[int, int]:
        if isinstance(n, str):
            n = Lit(n)
        s, e = self.new(), self.new()
        if isinstance(n, Lit):
            cur = s
            for ch in n.s:
                nxt = self.new()
                self.edges[cur].setdefault(ch, []).append(nxt)
                cur = nxt
            self.eps[cur].append(e)
        elif isinstance(n, Cls):
            for ch in n.chars:
                self.edges[s].setdefault(ch, []).append(e)
        elif isinstance(n, Seq):
            cur = s
            for p in n.parts:
                ps, pe = self.compile(p)
                self.eps[cur].append(ps)
                cur = pe
            self.eps[cur].append(e)
        elif isinstance(n, Alt):
            for o in n.options:
                os_, oe = self.compile(o)
                self.eps[s].append(os_)
                self.eps[oe].append(e)
        elif isinstance(n, Rep):
            cur = s
            if n.lo == 0:
                self.eps[s].append(e)                      # zero copies are allowed
            for i in range(n.hi):
                ps, pe = self.compile(n.node)
                self.eps[cur].append(ps)
                cur = pe
                if i + 1 >= n.lo:
                    self.eps[cur].append(e)                # may stop after this copy
        elif isinstance(n, OrderedSubset):
            m = len(n.items)
            q = {(j, c): self.new() for j in range(m + 1) for c in range(n.max_items + 1)}   # j = last item index+1 (0 = none), c = count
            self.eps[s].append(q[(0, 0)])
            for (j, c), st in q.items():
                self.eps[st].append(e)                                                      # may stop here
                if c >= n.max_items:
                    continue
                for k in range(j + 1, m + 1):
                    text = ("," if c > 0 else "") + n.items[k - 1]
                    cur = st
                    for ch in text[:-1]:
                        nxt = self.new()
                        self.edges[cur].setdefault(ch, []).append(nxt)
                        cur = nxt
                    self.edges[cur].setdefault(text[-1], []).append(q[(k, c + 1)])
        else:  # pragma: no cover
            raise TypeError(n)
        return s, e


class Grammar:
    """Compiled language with character-level incremental recognition."""

    def __init__(self, root: Node):
        self.nfa = _NFA()
        start, self.final = self.nfa.compile(root)
        self._closure_cache: dict[frozenset, frozenset] = {}
        self.start: frozenset = self._closure(frozenset([start]))

    def _closure(self, states: frozenset) -> frozenset:
        hit = self._closure_cache.get(states)
        if hit is not None:
            return hit
        out, stack = set(states), list(states)
        while stack:
            for t in self.nfa.eps[stack.pop()]:
                if t not in out:
                    out.add(t)
                    stack.append(t)
        res = frozenset(out)
        self._closure_cache[states] = res
        return res

    def step(self, states: frozenset, ch: str) -> frozenset:
        nxt = set()
        for s in states:
            nxt.update(self.nfa.edges[s].get(ch, ()))
        return self._closure(frozenset(nxt)) if nxt else frozenset()

    def advance(self, states: frozenset, text: str) -> frozenset:
        for ch in text:
            states = self.step(states, ch)
            if not states:
                return states
        return states

    def accepting(self, states: frozenset) -> bool:
        return self.final in states

    def viable(self, text: str) -> bool:
        return bool(self.advance(self.start, text))

    def accepts(self, text: str) -> bool:
        s = self.advance(self.start, text)
        return bool(s) and self.accepting(s)


# ----------------------------------------------------------------------------- vocabulary trie


class TokenTrie:
    """Trie over token strings. `allowed(grammar, states)` returns every token id whose text keeps the output viable."""

    def __init__(self, id_to_text: Mapping[int, str]):
        self.root: dict = {}
        for tid, text in id_to_text.items():
            if not text:
                continue
            node = self.root
            for ch in text:
                node = node.setdefault(ch, {})
            node.setdefault("\0ids", []).append(tid)

    def allowed(self, g: Grammar, states: frozenset) -> list[int]:
        out: list[int] = []
        stack = [(self.root, states)]
        while stack:
            node, st = stack.pop()
            for ch, child in node.items():
                if ch == "\0ids":
                    continue
                nst = g.step(st, ch)
                if not nst:
                    continue
                out += child.get("\0ids", [])
                stack.append((child, nst))
        return out


# ----------------------------------------------------------------------------- the rationale language


def _num() -> Node:
    return seq(opt("-"), alt("0", seq(Cls(NONZERO), Rep(Cls(DIGITS), 0, 8))), opt(seq(".", Rep(Cls(DIGITS), 1, 6))))


def _nonneg_num() -> Node:
    return seq(alt("0", seq(Cls(NONZERO), Rep(Cls(DIGITS), 0, 8))), opt(seq(".", Rep(Cls(DIGITS), 1, 6))))


def _prob() -> Node:
    return alt(seq("0", opt(seq(".", Rep(Cls(DIGITS), 1, 4)))), seq("1", opt(seq(".", Rep(Cls(frozenset("0")), 1, 4)))))


def _str() -> Node:
    return seq('"', Rep(Cls(IDCHARS), 1, 64), '"')


def _scalar() -> Node:
    return alt(_num(), _str())


def build_rationale_grammar(ev: EvidenceSnapshot, *, max_claims: int = 8) -> Grammar:
    """The language of valid rationales for THIS evidence snapshot."""
    by_field: dict[str, list[str]] = {}
    for eid, f in ev.evidence_index.items():
        by_field.setdefault(f, []).append(eid)
    numeric = [f for f, v in ev.fields.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    categorical = [f for f, v in ev.fields.items() if f not in numeric]
    events, roles = sorted(ev.event_times), sorted(ev.entities)

    def head(t: str, fields: Iterable[str], ops: Sequence[str], value: Node) -> list[Node]:
        opts = []
        for f in sorted(fields):
            for eid in sorted(by_field.get(f, [])):
                opts.append(seq(f'{{"type":"{t}","field":"{f}","op":"', alt(*ops), '","value":', value, f',"evidence_id":"{eid}"}}'))
        return opts

    claim_opts: list[Node] = []
    claim_opts += head("numeric_comparison", numeric, ["<", "<=", ">", ">=", "==", "!="], _num())
    claim_opts += head("set_membership", categorical, ["in", "not_in"], seq("[", _scalar(), Rep(seq(",", _scalar()), 0, 3), "]"))
    for f in sorted(events):
        for eid in sorted(by_field.get(f, [])):
            ref = alt(*[f'"{e}"' for e in events if e != f]) if len(events) > 1 else None
            parts = [seq(f'{{"type":"temporal_relation","field":"{f}","op":"within","value":', _nonneg_num(), f',"evidence_id":"{eid}"}}')]
            if ref is not None:
                parts.append(seq(f'{{"type":"temporal_relation","field":"{f}","op":"', alt("before", "after"), '","value":', ref, f',"evidence_id":"{eid}"}}'))
            claim_opts += parts
    if ev.graph is not None:
        claim_opts += head("graph_path", roles, ["path_exists", "path_absent"], seq("[", _str(), opt(seq(",", alt("1", "2", "3"))), "]"))
    claim_opts += head("entity_match", roles, ["==", "!="], _str())
    if not claim_opts:                                   # evidence with nothing claimable: only an empty claim list is valid
        claims: Node = Lit("[]")
    else:
        claim = Alt(tuple(claim_opts))
        claims = alt("[]", seq("[", claim, Rep(seq(",", claim), 0, max_claims - 1), "]"))
    ids = sorted(ev.evidence_index)
    refs: Node = seq("[", OrderedSubset(tuple(f'"{i}"' for i in ids)), "]") if ids else Lit("[]")
    return Grammar(seq('{"verdict":"', alt("fraud", "legitimate", "uncertain"), '","probability":', _prob(), ',"claims":', claims, ',"evidence_refs":', refs, "}"))


def canonicalise(rationale: dict) -> str:
    """Serialise a rationale in the grammar's canonical form: fixed key order, compact separators, evidence ids sorted."""
    claims = [{k: c[k] for k in ("type", "field", "op", "value", "evidence_id")} for c in rationale["claims"]]
    obj = {"verdict": rationale["verdict"], "probability": rationale["probability"], "claims": claims,
           "evidence_refs": sorted(rationale["evidence_refs"])}
    return json.dumps(obj, separators=(",", ":"))


# ----------------------------------------------------------------------------- generation helpers


def greedy_decode(g: Grammar, trie: TokenTrie, id_to_text: Mapping[int, str], eos_id: int, logits_fn, max_new_tokens: int = 384) -> tuple[str, bool]:
    """Greedy decoding with the mask. `logits_fn(text_so_far) -> {token_id: logit}` (any scorer). Returns (text, complete).

    Stops at EOS (permitted only in an accepting state) or after `max_new_tokens`; an unfinished output is returned
    with complete=False and must be rejected by the caller."""
    states, text = g.start, ""
    for _ in range(max_new_tokens):
        allowed = set(trie.allowed(g, states))
        if g.accepting(states):
            allowed.add(eos_id)
        if not allowed:
            return text, False
        scores = logits_fn(text)
        tid = max(allowed, key=lambda t: (scores.get(t, float("-inf")), -t))
        if tid == eos_id:
            return text, True
        states = g.advance(states, id_to_text[tid])
        text += id_to_text[tid]
    return text, g.accepting(states)


class GrammarLogitsProcessor:
    """Hugging Face `LogitsProcessor`: masks every token that would leave the language.

    Tracks one grammar state per batch row and advances it by the tokens generated since `prompt_len`.
    """

    def __init__(self, grammar: Grammar, trie: TokenTrie, id_to_text: Mapping[int, str], eos_id: int, prompt_len: int):
        self.g, self.trie, self.text, self.eos, self.prompt_len = grammar, trie, id_to_text, eos_id, prompt_len
        self._state: dict[int, tuple[frozenset, int]] = {}

    def __call__(self, input_ids, scores):
        import torch

        mask = torch.full_like(scores, float("-inf"))
        for b in range(input_ids.shape[0]):
            states, done = self._state.get(b, (self.g.start, 0))
            gen = input_ids[b, self.prompt_len:].tolist()
            for tid in gen[done:]:
                states = self.g.advance(states, self.text.get(int(tid), ""))
            self._state[b] = (states, len(gen))
            allowed = self.trie.allowed(self.g, states) if states else []
            if self.g.accepting(states):
                allowed = allowed + [self.eos]
            if not allowed:
                allowed = [self.eos]                     # dead end: end the sequence; the verifier rejects the fragment
            idx = torch.tensor(sorted(set(allowed)), device=scores.device)
            mask[b, idx] = 0.0
        return scores + mask


def vocab_texts(tokenizer) -> dict[int, str]:
    """Decode every token id to the text it appends. Special tokens map to ''."""
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    out = {}
    for tid in range(len(tokenizer)):
        out[tid] = "" if tid in special else tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return out
