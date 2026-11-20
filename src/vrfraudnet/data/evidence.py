"""Evidence construction: the trusted snapshot a rationale about ONE transaction is checked against.

Everything is built from rows and edges strictly before the prediction time. A snapshot holds the record's own
fields, a handful of past-only aggregates, named past events, entity roles, and (when the dataset has a graph) the
ego neighbourhood of the entities involved. The ego graph is marked `complete` only if the breadth-first expansion
finished inside its node cap, which is what lets the verifier prove a path ABSENT; otherwise absence stays unknown.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ..verifier.evidence import EvidenceSnapshot, GraphEvidence
from .graph import ego_edges
from .providers import LoadedDataset


def sanitize_field(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]", "_", str(name).lower())
    return s if re.match(r"^[a-z]", s) else f"f_{s}"


class EvidenceBuilder:
    def __init__(self, ds: LoadedDataset, *, window_s: float = 86400.0, hops: int = 3, node_cap: int = 2000):
        self.ds, self.window, self.hops, self.node_cap = ds, window_s, hops, node_cap
        f = ds.frame
        self.t = f["__time"].to_numpy()
        self.fields = [c for c in ds.meta.get("evidence_fields", []) if c in f]
        self.has_entities = "__origin" in f
        self._by_entity: dict[str, np.ndarray] = {}
        if self.has_entities:
            for k, grp in f.groupby("__origin").indices.items():
                self._by_entity[k] = np.sort(self.t[grp])
        names = ds.meta.get("node_names")
        self.node_index = {n: i for i, n in enumerate(names)} if names is not None else {}
        self.node_names = names

    def snapshot(self, pos: int) -> EvidenceSnapshot:
        row = self.ds.frame.iloc[pos]
        t = float(row["__time"])
        fields, index = {}, {}
        for c in self.fields:
            v = row[c]
            fields[sanitize_field(c)] = v.item() if hasattr(v, "item") else v
        events, entities = {}, {}
        if self.has_entities:
            past = self._by_entity.get(row["__origin"], np.array([]))
            lo = np.searchsorted(past, t - self.window, side="left")
            hi = np.searchsorted(past, t, side="left")                      # strictly before t
            fields["recent_transfer_count"] = int(hi - lo)
            if hi > 0:
                events["last_transfer_time"] = float(past[hi - 1])
            entities["origin"] = str(row["__origin"])
            entities["beneficiary"] = str(row["__beneficiary"])
        for f in list(fields) + list(events) + list(entities):
            index[f"E_{re.sub(r'[^A-Za-z0-9_]', '_', f).upper()}"] = f
        graph: Optional[GraphEvidence] = None
        if self.ds.graph is not None and self.has_entities and self.node_names is not None:
            edges, complete = [], True
            for role in ("__origin", "__beneficiary"):
                e, c = ego_edges(self.ds.graph, self.node_index[str(row[role])], t, self.hops, self.node_cap)
                edges += e
                complete = complete and c
            named = tuple(sorted({(self.node_names[s], self.node_names[d], ts) for s, d, ts in edges}))
            graph = GraphEvidence(named, complete=complete, prediction_time=t, radius=self.hops)
        return EvidenceSnapshot(transaction_id=str(row["__id"]), prediction_time=t, fields=fields, event_times=events, entities=entities,
                                evidence_index=index, graph=graph)
