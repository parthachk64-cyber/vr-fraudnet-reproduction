"""The trusted evidence snapshot a rationale is checked against.

A snapshot is frozen at prediction time. It is the ONLY thing the verifier reads: the language model's
text, retrieved exemplars and anything produced after the prediction time are never evidence.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..errors import LeakageError


@dataclass(frozen=True)
class GraphEvidence:
    """Directed, timestamped edges visible at prediction time.

    `complete` states whether the stored neighbourhood is exhaustive for the hop radius the verifier may be
    asked about. If it is not (or the asked hop count exceeds `radius`), the verifier can prove a path exists but can NEVER prove one is absent.
    """

    edges: tuple[tuple[str, str, float], ...]
    complete: bool = False
    prediction_time: float = float("inf")
    radius: int = 99  # hop radius for which `complete` holds; absence is provable only up to this many hops

    def __post_init__(self):
        late = [e for e in self.edges if e[2] >= self.prediction_time]
        if late:
            raise LeakageError(f"{len(late)} graph edge(s) are not before the prediction time {self.prediction_time}")

    def adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {}
        for s, d, _ in self.edges:
            adj.setdefault(s, []).append(d)
        return adj

    def path(self, src: str, dst: str, max_hops: int, *, node_budget: int = 10_000, time_budget_s: float = 1.0) -> Optional[bool]:
        """True: a path of at most `max_hops` exists. False: proven absent. None: unknown (budget hit, or graph incomplete)."""
        adj = self.adjacency()
        if src == dst:
            return True
        deadline = time.monotonic() + time_budget_s
        seen, q, visited = {src}, deque([(src, 0)]), 0
        while q:
            node, depth = q.popleft()
            visited += 1
            if visited > node_budget or time.monotonic() > deadline:
                return None
            if depth >= max_hops:
                continue
            for nxt in adj.get(node, ()):
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, depth + 1))
        return False if (self.complete and max_hops <= self.radius) else None


@dataclass(frozen=True)
class EvidenceSnapshot:
    transaction_id: str
    prediction_time: float
    fields: Mapping[str, Any] = field(default_factory=dict)  # numeric or categorical values for this record
    event_times: Mapping[str, float] = field(default_factory=dict)  # named past events (seconds, same clock as prediction_time)
    entities: Mapping[str, str] = field(default_factory=dict)  # role -> entity id ("origin", "beneficiary", ...)
    evidence_index: Mapping[str, str] = field(default_factory=dict)  # evidence id -> field/role/event name it documents
    graph: Optional[GraphEvidence] = None

    def __post_init__(self):
        future = {k: t for k, t in self.event_times.items() if t > self.prediction_time}
        if future:
            raise LeakageError(f"events after the prediction time in evidence: {sorted(future)}")

    @property
    def allowed_fields(self) -> set[str]:
        return set(self.fields) | set(self.event_times) | set(self.entities)

    @property
    def evidence_ids(self) -> set[str]:
        return set(self.evidence_index)

    def to_prompt_dict(self) -> dict:
        """What the rationale model is shown (never includes anything absent from the snapshot)."""
        return {"transaction_id": self.transaction_id, "prediction_time": self.prediction_time, "fields": dict(self.fields),
                "event_times": dict(self.event_times), "entities": dict(self.entities),
                "evidence_ids": sorted(self.evidence_index), "graph_available": self.graph is not None}
