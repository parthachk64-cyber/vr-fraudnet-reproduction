"""The five typed claim checks. Each is a pure, deterministic function of (claim, evidence snapshot).

Status vocabulary: "supported", "contradicted", "unverifiable" (evidence missing or insufficient), "malformed".
A claim passes only when it is supported. "Unverifiable" is NOT "false": for a graph the absence of a stored
edge is not proof of absence unless the snapshot declares itself complete.
No claim text is ever evaluated as code.
"""
from __future__ import annotations

import math
import operator
from typing import Any, Callable

from ..types import ClaimVerdict
from .evidence import EvidenceSnapshot

_CMP: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
    "==": lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), "!=": lambda a, b: not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9),
}
GRAPH_MAX_HOPS = 3
GRAPH_NODE_BUDGET = 10_000
GRAPH_TIME_BUDGET_S = 1.0


def _v(i: int, c: dict, passed: bool, status: str, reason: str, observed: Any = None) -> ClaimVerdict:
    return ClaimVerdict(index=i, claim_type=str(c.get("type")), passed=passed, status=status, reason=reason, observed=observed)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_numeric(i: int, c: dict, ev: EvidenceSnapshot) -> ClaimVerdict:
    obs = ev.fields.get(c["field"])
    if obs is None:
        return _v(i, c, False, "unverifiable", f"field {c['field']!r} is absent from the evidence")
    if not _num(obs):
        return _v(i, c, False, "malformed", f"evidence for {c['field']!r} is not numeric", obs)
    ok = _CMP[c["op"]](float(obs), float(c["value"]))
    return _v(i, c, ok, "supported" if ok else "contradicted", f"observed {obs} {c['op']} {c['value']} is {ok}", obs)


def check_set_membership(i: int, c: dict, ev: EvidenceSnapshot) -> ClaimVerdict:
    obs = ev.fields.get(c["field"])
    if obs is None:
        return _v(i, c, False, "unverifiable", f"field {c['field']!r} is absent from the evidence")
    inside = obs in list(c["value"])
    ok = inside if c["op"] == "in" else not inside
    return _v(i, c, ok, "supported" if ok else "contradicted", f"observed {obs!r} {c['op']} {list(c['value'])} is {ok}", obs)


def check_temporal(i: int, c: dict, ev: EvidenceSnapshot) -> ClaimVerdict:
    e = ev.event_times.get(c["field"])
    if e is None:
        return _v(i, c, False, "unverifiable", f"event {c['field']!r} is absent from the evidence")
    t = ev.prediction_time
    op, val = c["op"], c["value"]
    if op == "within":
        if not _num(val) or val < 0:
            return _v(i, c, False, "malformed", "within needs a non-negative number of seconds")
        age = t - e
        ok = 0 <= age <= val
        return _v(i, c, ok, "supported" if ok else "contradicted", f"event age {age} s vs window {val} s", age)
    if not isinstance(val, str) or val not in ev.event_times:
        return _v(i, c, False, "unverifiable", f"reference event {val!r} is absent from the evidence")
    ref = ev.event_times[val]
    ok = e < ref if op == "before" else e > ref
    return _v(i, c, ok, "supported" if ok else "contradicted", f"{c['field']}={e} {op} {val}={ref} is {ok}", (e, ref))


def check_graph(i: int, c: dict, ev: EvidenceSnapshot) -> ClaimVerdict:
    if ev.graph is None:
        return _v(i, c, False, "unverifiable", "no graph evidence is available; absence cannot be inferred")
    val = c["value"]
    target = val[0]
    hops = val[1] if len(val) > 1 else 2
    if not isinstance(target, str) or not _num(hops) or not 1 <= int(hops) <= GRAPH_MAX_HOPS:
        return _v(i, c, False, "malformed", f"graph_path value must be [target, hops in 1..{GRAPH_MAX_HOPS}]")
    src = ev.entities.get(c["field"])
    if src is None:
        return _v(i, c, False, "unverifiable", f"entity role {c['field']!r} is absent from the evidence")
    found = ev.graph.path(src, target, int(hops), node_budget=GRAPH_NODE_BUDGET, time_budget_s=GRAPH_TIME_BUDGET_S)
    if found is None:
        return _v(i, c, False, "unverifiable", "no path found, but the stored neighbourhood is incomplete or the search budget was exhausted")
    ok = found if c["op"] == "path_exists" else not found
    return _v(i, c, ok, "supported" if ok else "contradicted", f"path {src}->{target} within {int(hops)} hops: {found}", found)


def check_entity(i: int, c: dict, ev: EvidenceSnapshot) -> ClaimVerdict:
    obs = ev.entities.get(c["field"])
    if obs is None:
        return _v(i, c, False, "unverifiable", f"entity role {c['field']!r} is absent from the evidence")
    same = obs == c["value"]
    ok = same if c["op"] == "==" else not same
    return _v(i, c, ok, "supported" if ok else "contradicted", f"observed {obs!r} {c['op']} {c['value']!r} is {ok}", obs)


DISPATCH: dict[str, Callable[[int, dict, EvidenceSnapshot], ClaimVerdict]] = {
    "numeric_comparison": check_numeric, "set_membership": check_set_membership, "temporal_relation": check_temporal,
    "graph_path": check_graph, "entity_match": check_entity,
}
