"""Result types. The verifier state keeps 'never invoked' distinct from 'invoked and failed' everywhere."""
from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


class VerifierState(str, enum.Enum):
    NOT_INVOKED = "not_invoked"  # common path: the review band was never entered
    ACCEPTED = "accepted"  # V = 1
    REJECTED = "rejected"  # V = 0


class Route(str, enum.Enum):
    ACCEPT = "accept"  # score below the lower threshold, common path
    REJECT = "reject"  # score above the upper threshold, common path
    VERIFIED_AUTO = "verified_auto_decision"  # review band, rationale verified
    HUMAN_REVIEW = "human_review"  # review band, verifier rejected (or generation failed)


@dataclass(frozen=True)
class ClaimVerdict:
    index: int
    claim_type: str
    passed: bool
    status: str  # "supported" | "contradicted" | "unverifiable" | "malformed"
    reason: str
    observed: Any = None


@dataclass(frozen=True)
class VerificationReport:
    state: VerifierState
    claims: tuple[ClaimVerdict, ...] = ()
    rule_failures: tuple[str, ...] = ()  # cross-claim / verdict / schema-level reasons

    @property
    def accepted(self) -> bool:
        return self.state is VerifierState.ACCEPTED

    def reasons(self) -> list[str]:
        out = [f"claim[{c.index}] {c.claim_type}: {c.status} ({c.reason})" for c in self.claims if not c.passed]
        out += [f"rule: {r}" for r in self.rule_failures]
        return out


@dataclass(frozen=True)
class TransactionResult:
    """Everything the pipeline reports for one transaction."""

    transaction_id: str
    route: Route
    triage_probability: float
    risk_score: float  # calibrated; see `score_provenance`
    score_provenance: str  # "triage_only" | "triage+verified_rationale"
    verifier: VerifierState
    rationale: Optional[dict] = None
    rationale_probability_used: Optional[float] = None  # None when not eligible: never a stand-in 0.0
    verification: Optional[VerificationReport] = None
    conformal_set: Optional[tuple[int, ...]] = None
    automated: bool = True  # False when the case is for a human

    def to_dict(self) -> dict:
        d = asdict(self)
        d["route"] = self.route.value
        d["verifier"] = self.verifier.value
        if self.verification is not None:
            d["verification"]["state"] = self.verification.state.value
        if self.conformal_set is not None:
            d["conformal_set"] = list(self.conformal_set)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TransactionResult":
        v = d.get("verification")
        report = None
        if v is not None:
            report = VerificationReport(
                state=VerifierState(v["state"]),
                claims=tuple(ClaimVerdict(**c) for c in v.get("claims", ())),
                rule_failures=tuple(v.get("rule_failures", ())),
            )
        cs = d.get("conformal_set")
        return cls(
            transaction_id=d["transaction_id"], route=Route(d["route"]),
            triage_probability=d["triage_probability"], risk_score=d["risk_score"],
            score_provenance=d["score_provenance"], verifier=VerifierState(d["verifier"]),
            rationale=d.get("rationale"), rationale_probability_used=d.get("rationale_probability_used"),
            verification=report, conformal_set=tuple(cs) if cs is not None else None,
            automated=d.get("automated", True),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class Provenance:
    """Attached to every run output so a number can be traced to its inputs."""

    kind: str  # "benchmark" | "synthetic_fixture" | "transcribed"
    seed: Optional[int] = None
    split_fingerprint: Optional[str] = None
    config_hash: Optional[str] = None
    notes: list[str] = field(default_factory=list)
