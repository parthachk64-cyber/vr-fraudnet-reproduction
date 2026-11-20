"""Versioned rationale schema and a dependency-free validator that mirrors it.

The JSON Schema file (`rationale_v1.json`) is the contract. `validate_rationale` enforces the same rules
without requiring `jsonschema`, and additionally checks the case-specific constraints a static schema
cannot express: fields and evidence IDs must come from the CURRENT transaction's evidence snapshot.
Tests cross-check the two against each other.
"""
from __future__ import annotations

import json
import math
import re
from importlib import resources
from typing import Any, Iterable, Optional

from ..errors import SchemaError

SCHEMA_VERSION = "rationale_v1"
MAX_CLAIMS = 8
CLAIM_TYPES = ("numeric_comparison", "set_membership", "temporal_relation", "graph_path", "entity_match")
OPS = {
    "numeric_comparison": ("<", "<=", ">", ">=", "==", "!="),
    "set_membership": ("in", "not_in"),
    "temporal_relation": ("before", "after", "within"),
    "graph_path": ("path_exists", "path_absent"),
    "entity_match": ("==", "!="),
}
VERDICTS = ("fraud", "legitimate", "uncertain")
_FIELD = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EID = re.compile(r"^[A-Za-z0-9_:.-]{1,64}$")
_CLAIM_KEYS = {"type", "field", "op", "value", "evidence_id"}
_TOP_KEYS = {"verdict", "probability", "claims", "evidence_refs"}


def load_schema() -> dict:
    with resources.files("vrfraudnet.schemas").joinpath("rationale_v1.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _isnum(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_rationale(obj: Any, *, allowed_fields: Optional[Iterable[str]] = None,
                       allowed_evidence_ids: Optional[Iterable[str]] = None) -> list[str]:
    """Return a list of human-readable violations (empty list = valid)."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["rationale must be a JSON object"]
    extra = set(obj) - _TOP_KEYS
    missing = _TOP_KEYS - set(obj)
    if extra:
        errs.append(f"unexpected top-level keys: {sorted(extra)}")
    if missing:
        errs.append(f"missing required keys: {sorted(missing)}")
        return errs
    if obj["verdict"] not in VERDICTS:
        errs.append(f"verdict must be one of {VERDICTS}")
    p = obj["probability"]
    if not _isnum(p) or not 0.0 <= p <= 1.0:
        errs.append("probability must be a finite number in [0, 1]")
    claims = obj["claims"]
    if not isinstance(claims, list):
        errs.append("claims must be a list")
        claims = []
    elif len(claims) > MAX_CLAIMS:
        errs.append(f"at most {MAX_CLAIMS} claims are allowed, got {len(claims)}")
    refs = obj["evidence_refs"]
    if not isinstance(refs, list) or len(refs) > MAX_CLAIMS or len(set(map(str, refs))) != len(refs):
        errs.append("evidence_refs must be a list of up to 8 unique ids")
        refs = refs if isinstance(refs, list) else []
    for r in refs:
        if not isinstance(r, str) or not _EID.match(r):
            errs.append(f"invalid evidence id {r!r}")
        elif allowed_evidence_ids is not None and r not in set(allowed_evidence_ids):
            errs.append(f"evidence id {r!r} is not in the current transaction's evidence")
    fields = set(allowed_fields) if allowed_fields is not None else None
    for i, c in enumerate(claims):
        errs += [f"claims[{i}]: {e}" for e in _validate_claim(c, fields)]
    return errs


def _validate_claim(c: Any, allowed_fields: Optional[set]) -> list[str]:
    if not isinstance(c, dict):
        return ["claim must be an object"]
    errs = []
    if set(c) != _CLAIM_KEYS:
        return [f"claim keys must be exactly {sorted(_CLAIM_KEYS)}, got {sorted(c)}"]
    t, op, v = c["type"], c["op"], c["value"]
    if t not in CLAIM_TYPES:
        return [f"unknown claim type {t!r}"]
    if op not in OPS[t]:
        errs.append(f"operator {op!r} is not allowed for {t}")
    if not isinstance(c["field"], str) or not _FIELD.match(c["field"]):
        errs.append(f"invalid field name {c['field']!r}")
    elif allowed_fields is not None and c["field"] not in allowed_fields:
        errs.append(f"field {c['field']!r} is not in the current transaction's evidence")
    if not isinstance(c["evidence_id"], str) or not _EID.match(c["evidence_id"]):
        errs.append("invalid evidence_id")
    ok_value = _isnum(v) or (isinstance(v, str) and len(v) <= 64) or (
        isinstance(v, list) and 1 <= len(v) <= 4 and all(_isnum(x) or isinstance(x, str) for x in v))
    if not ok_value:
        errs.append("value must be a number, a short string, or a list of 1-4 scalars")
    elif t == "numeric_comparison" and not _isnum(v):
        errs.append("numeric_comparison value must be a number")
    elif t == "set_membership" and not isinstance(v, list):
        errs.append("set_membership value must be a list")
    elif t == "graph_path" and not isinstance(v, list):
        errs.append("graph_path value must be [target_entity, optional_max_hops]")
    elif t == "entity_match" and not isinstance(v, str):
        errs.append("entity_match value must be an entity id string")
    return errs


def require_valid(obj: Any, **kw) -> None:
    errs = validate_rationale(obj, **kw)
    if errs:
        raise SchemaError("; ".join(errs))
