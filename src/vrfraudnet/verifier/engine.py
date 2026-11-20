"""Stage 3: the fixed, non-trainable verifier V(r, x, G) in {0, 1}.

Order of checks (every failure is recorded; the decision is the conjunction):
  1. parse + schema (including: fields and evidence ids must come from THIS transaction's evidence);
  2. each claim against its typed primitive;
  3. evidence-ID bookkeeping: a claim's id must document the claim's field, and be listed in evidence_refs;
  4. cross-claim consistency (duplicates, contradictions);
  5. verdict consistency (decisive verdicts need claims; verdict and probability must agree).
V = 1 only if there are no failures at all. The rationale-side probability is never consulted for the decision.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from ..schemas import validate_rationale
from ..types import ClaimVerdict, VerificationReport, VerifierState
from .evidence import EvidenceSnapshot
from .primitives import DISPATCH

# Verdict/probability agreement: decisive verdicts must lie on their side of 0.5 (decision D-05).
FRAUD_MIN_P = 0.5


def verify_text(raw: str, ev: EvidenceSnapshot, **kw) -> VerificationReport:
    """Verify raw model output. Anything that does not parse is a rejection, not a crash."""
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError) as e:
        return _reject(f"unparseable_output: {e}")
    return verify(obj, ev, **kw)


def _reject(*reasons: str, claims: tuple = ()) -> VerificationReport:
    return VerificationReport(state=VerifierState.REJECTED, claims=claims, rule_failures=tuple(reasons))


def verify(rationale: Any, ev: EvidenceSnapshot, *, direction_hints: Optional[Mapping[str, Mapping[str, str]]] = None) -> VerificationReport:
    """`direction_hints` optionally maps field -> {"above": "fraud"|"legitimate", ...}; see `_direction_conflicts`."""
    errs = validate_rationale(rationale, allowed_fields=ev.allowed_fields, allowed_evidence_ids=ev.evidence_ids)
    if errs:
        return _reject(*[f"schema: {e}" for e in errs])
    claims = rationale["claims"]
    verdicts: list[ClaimVerdict] = [DISPATCH[c["type"]](i, c, ev) for i, c in enumerate(claims)]
    rules: list[str] = []

    refs = set(rationale["evidence_refs"])
    for i, c in enumerate(claims):
        if ev.evidence_index.get(c["evidence_id"]) != c["field"]:
            rules.append(f"claim[{i}] evidence_id {c['evidence_id']!r} does not document field {c['field']!r}")
        if c["evidence_id"] not in refs:
            rules.append(f"claim[{i}] evidence_id {c['evidence_id']!r} is not listed in evidence_refs")
    keys = [json.dumps(c, sort_keys=True) for c in claims]
    if len(set(keys)) != len(keys):
        rules.append("duplicate claims")
    rules += _contradictions(claims)

    verdict, p = rationale["verdict"], float(rationale["probability"])
    if verdict != "uncertain":
        if not claims:
            rules.append(f"verdict {verdict!r} is decisive but no claims support it")
        if verdict == "fraud" and p < FRAUD_MIN_P:
            rules.append(f"verdict 'fraud' contradicts probability {p} < {FRAUD_MIN_P}")
        if verdict == "legitimate" and p >= FRAUD_MIN_P:
            rules.append(f"verdict 'legitimate' contradicts probability {p} >= {FRAUD_MIN_P}")
        if direction_hints:
            rules += _direction_conflicts(claims, verdicts, verdict, direction_hints)

    ok = all(v.passed for v in verdicts) and not rules
    return VerificationReport(state=VerifierState.ACCEPTED if ok else VerifierState.REJECTED,
                              claims=tuple(verdicts), rule_failures=tuple(rules))


def _direction_conflicts(claims, verdicts, verdict, hints) -> list[str]:
    out = []
    for i, c in enumerate(claims):
        h = hints.get(c["field"])
        if not h or not verdicts[i].passed or c["type"] != "numeric_comparison":
            continue
        side = "above" if c["op"] in (">", ">=") else "below" if c["op"] in ("<", "<=") else None
        if side and h.get(side) not in (None, verdict):
            out.append(f"claim[{i}] ({c['field']} {c['op']} {c['value']}) points to {h.get(side)!r}, not the stated verdict {verdict!r}")
    return out


def _contradictions(claims: list[dict]) -> list[str]:
    out: list[str] = []
    by: dict[tuple, list[dict]] = {}
    for c in claims:
        by.setdefault((c["type"], c["field"]), []).append(c)
    for (t, f), group in by.items():
        if len(group) < 2:
            continue
        if t == "numeric_comparison":
            lo, hi, lo_inc, hi_inc, eq, ne = -float("inf"), float("inf"), False, False, set(), set()
            for c in group:
                v, op = float(c["value"]), c["op"]
                if op in (">", ">="):
                    if v > lo or (v == lo and op == ">"):
                        lo, lo_inc = v, op == ">="
                elif op in ("<", "<="):
                    if v < hi or (v == hi and op == "<"):
                        hi, hi_inc = v, op == "<="
                elif op == "==":
                    eq.add(v)
                else:
                    ne.add(v)
            empty = lo > hi or (lo == hi and not (lo_inc and hi_inc))
            if empty or len(eq) > 1 or (eq & ne) or any(not (lo < e < hi or (e == lo and lo_inc) or (e == hi and hi_inc)) for e in eq):
                out.append(f"contradictory numeric claims on {f!r}")
        elif t == "set_membership":
            ins = [set(map(str, c["value"])) for c in group if c["op"] == "in"]
            nots = set().union(*[set(map(str, c["value"])) for c in group if c["op"] == "not_in"]) if any(c["op"] == "not_in" for c in group) else set()
            if ins and not (set.intersection(*ins) - nots):
                out.append(f"contradictory set-membership claims on {f!r}")
        elif t == "entity_match":
            eqs = {c["value"] for c in group if c["op"] == "=="}
            nes = {c["value"] for c in group if c["op"] == "!="}
            if len(eqs) > 1 or (eqs & nes):
                out.append(f"contradictory entity claims on {f!r}")
        elif t == "graph_path":
            ex = {str(c["value"][0]) for c in group if c["op"] == "path_exists"}
            ab = {str(c["value"][0]) for c in group if c["op"] == "path_absent"}
            if ex & ab:
                out.append(f"contradictory graph claims on {f!r}")
        elif t == "temporal_relation":
            ops = {c["op"] for c in group}
            if {"before", "after"} <= ops or {"within", "after"} <= ops:
                out.append(f"contradictory temporal claims on {f!r}")
    return out
