"""Training signals beyond the plain predictive loss (Sec. 3.6) and the corpora that feed them.

  * cost-sensitive weights and class-balanced focal loss live in `models.triage` (they train the LightGBM triage model);
  * `make_counterfactual`: label-preserving, evidence-consistent training pairs for the rationale model (Sec. 3.6.3);
  * `RuleTargetBuilder`: verified review-band TARGET rationales built from training evidence only (decision D-15);
  * `load_frozen_variants` / `augment_with_variants`: defensive training on OFFLINE, frozen edited variants that someone
    else generated. This repository does not generate edited fraud transactions and has no search for variants that evade a detector.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..errors import LeakageError, ProtocolError
from ..verifier import EvidenceSnapshot, verify
from ..verifier.evidence import EvidenceSnapshot as Snap


# ------------------------------------------------------------------------------------------ counterfactual
def make_counterfactual(ev: EvidenceSnapshot, rationale: dict, rng: np.random.Generator):
    """Change ONE admissible numeric evidence value so that a claim built on it turns false, then reconstruct the target:
    claims the change contradicts are removed, claims backed by unchanged evidence are kept verbatim, and the label-side fields
    (verdict, probability) are unchanged. Returns (new_evidence, new_rationale) or None if no claim can be flipped.
    The result is verified; an unverifiable pair is never returned."""
    cands = [i for i, c in enumerate(rationale["claims"]) if c["type"] == "numeric_comparison" and isinstance(ev.fields.get(c["field"]), (int, float))]
    if not cands:
        return None
    i = int(rng.choice(cands))
    c = rationale["claims"][i]
    v = float(ev.fields[c["field"]])
    new_v = (v * 0.25 - 1.0) if c["op"] in (">", ">=", "==") else (v * 4.0 + 1.0)
    fields = dict(ev.fields)
    fields[c["field"]] = type(ev.fields[c["field"]])(new_v) if isinstance(ev.fields[c["field"]], int) else new_v
    ev2 = Snap(transaction_id=ev.transaction_id + "#cf", prediction_time=ev.prediction_time, fields=fields, event_times=dict(ev.event_times),
               entities=dict(ev.entities), evidence_index=dict(ev.evidence_index), graph=ev.graph)
    kept = []
    for j, cl in enumerate(rationale["claims"]):
        probe = {"verdict": "uncertain", "probability": 0.5, "claims": [cl], "evidence_refs": [cl["evidence_id"]]}
        if verify(probe, ev2).accepted:
            kept.append(cl)
    if len(kept) == len(rationale["claims"]):
        return None
    verdict = rationale["verdict"] if kept else "uncertain"
    out = {"verdict": verdict, "probability": rationale["probability"], "claims": kept, "evidence_refs": sorted({k["evidence_id"] for k in kept})}
    if not verify(out, ev2).accepted:
        return None
    return ev2, out


# ------------------------------------------------------------------------------------------ rule-built targets
@dataclass
class TargetStats:
    """Per-field mean/std from TRAINING rows only."""

    mean: dict
    std: dict

    @classmethod
    def fit(cls, snapshots: list[EvidenceSnapshot]) -> "TargetStats":
        cols: dict[str, list[float]] = {}
        for s in snapshots:
            for f, v in s.fields.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cols.setdefault(f, []).append(float(v))
        return cls({f: float(np.mean(v)) for f, v in cols.items()}, {f: float(np.std(v)) or 1.0 for f, v in cols.items()})


class RuleTargetBuilder:
    """Builds a target rationale for a TRAINING review-band row from its own evidence and label (decision D-15).

    The paper does not say how its training rationales were authored. This builder is deterministic and conservative: the most
    atypical numeric fields (by training z-score) become numeric claims with loosened-but-true thresholds, the categorical value becomes
    a set-membership claim, a recent event becomes a temporal claim, and a visible two-hop path becomes a graph claim. The verdict follows
    the TRAINING label; the probability is the triage probability pulled toward the label. Targets the verifier rejects are dropped.
    """

    def __init__(self, stats: TargetStats, max_numeric: int = 3):
        self.stats, self.max_numeric = stats, max_numeric

    def build(self, ev: EvidenceSnapshot, p_triage: float, label: int) -> Optional[dict]:
        rev = {f: e for e, f in ev.evidence_index.items()}
        claims = []
        nums = [(abs((float(v) - self.stats.mean.get(f, 0.0)) / self.stats.std.get(f, 1.0)), f, float(v)) for f, v in ev.fields.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and f in rev]
        for z, f, v in sorted(nums, reverse=True)[: self.max_numeric]:
            above = v >= self.stats.mean.get(f, 0.0)
            claims.append({"type": "numeric_comparison", "field": f, "op": ">=" if above else "<=", "value": int(np.floor(v)) if above else int(np.ceil(v)), "evidence_id": rev[f]})
        for f, v in sorted(ev.fields.items()):
            if isinstance(v, str) and f in rev:
                claims.append({"type": "set_membership", "field": f, "op": "in", "value": [v], "evidence_id": rev[f]})
                break
        if "last_transfer_time" in ev.event_times and "last_transfer_time" in rev:
            age = ev.prediction_time - ev.event_times["last_transfer_time"]
            claims.append({"type": "temporal_relation", "field": "last_transfer_time", "op": "within", "value": int(np.ceil(age)), "evidence_id": rev["last_transfer_time"]})
        if ev.graph is not None and "origin" in ev.entities and "origin" in rev:
            reach = [d for s, d, _ in ev.graph.edges if s == ev.entities["origin"]]
            if reach:
                claims.append({"type": "graph_path", "field": "origin", "op": "path_exists", "value": [sorted(reach)[0], 1], "evidence_id": rev["origin"]})
        claims = claims[:8]
        p = float(np.clip(0.5 * p_triage + 0.5 * label, 0.02, 0.98))
        rat = {"verdict": "fraud" if label == 1 else "legitimate", "probability": p, "claims": claims, "evidence_refs": sorted({c["evidence_id"] for c in claims})}
        if not claims or not verify(rat, ev).accepted:
            return None
        return rat


# ------------------------------------------------------------------------------------------ frozen offline variants
REQUIRED_VARIANT_COLUMNS = ("original_id", "family", "budget")


def load_frozen_variants(path: Path, *, expected_sha256: Optional[str] = None) -> tuple[pd.DataFrame, str]:
    """Load a FROZEN file of edited variants produced elsewhere. Records the checksum; refuses a file whose checksum differs from the one pinned."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise LeakageError(f"variant file checksum {digest[:12]} differs from the pinned {expected_sha256[:12]}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in REQUIRED_VARIANT_COLUMNS if c not in df]
    if missing:
        raise ProtocolError(f"variant file lacks columns {missing}")
    return df, digest


def augment_with_variants(X, y, w, train_ids, variants: pd.DataFrame, variant_features: np.ndarray, weight: float = 0.5):
    """Append frozen variants of TRAINING fraud rows (label 1) with sample weight `weight` (the adversarial weight 0.50). Originals stay."""
    allowed = set(map(str, train_ids))
    bad = [o for o in variants["original_id"].astype(str) if o not in allowed]
    if bad:
        raise LeakageError(f"{len(bad)} variant(s) derive from rows outside the training partition (e.g. {bad[0]!r})")
    if len(variants) != len(variant_features):
        raise ProtocolError("variant features and variant table differ in length")
    return (np.vstack([X, variant_features]), np.r_[np.asarray(y), np.ones(len(variants), int)],
            np.r_[np.ones(len(X)) if w is None else np.asarray(w, float), np.full(len(variants), float(weight))])
