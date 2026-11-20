"""Sparse temporal graph snapshots. Nothing here ever builds a dense N x N matrix.

Future-edge rule: a snapshot at prediction time t contains only edges with timestamp STRICTLY BEFORE t
(optionally within a trailing window), so the transaction being scored cannot appear in its own graph context.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..errors import LeakageError


class TemporalGraph:
    def __init__(self, src, dst, ts, n_nodes: int):
        src, dst, ts = (np.asarray(a) for a in (src, dst, ts))
        if not (len(src) == len(dst) == len(ts)):
            raise ValueError("src, dst, ts differ in length")
        if len(src) and (src.max() >= n_nodes or dst.max() >= n_nodes or src.min() < 0 or dst.min() < 0):
            raise ValueError("node index out of range")
        order = np.argsort(ts, kind="mergesort")
        self.src, self.dst, self.ts = src[order].astype(np.int64), dst[order].astype(np.int64), ts[order].astype(float)
        self.n_nodes = int(n_nodes)

    def edge_slice(self, t: float, window: Optional[float] = None) -> slice:
        hi = int(np.searchsorted(self.ts, t, side="left"))            # strictly before t
        lo = 0 if window is None else int(np.searchsorted(self.ts, t - window, side="left"))
        return slice(lo, hi)

    def snapshot(self, t: float, window: Optional[float] = None, *, symmetric: bool = True, weighted: bool = False) -> sp.csr_matrix:
        sl = self.edge_slice(t, window)
        s, d = self.src[sl], self.dst[sl]
        if symmetric:
            s, d = np.r_[s, d], np.r_[d, s]
        a = sp.coo_matrix((np.ones(len(s), dtype=np.float32), (s, d)), shape=(self.n_nodes, self.n_nodes)).tocsr()
        if not weighted:
            a.data[:] = 1.0                                              # collapse duplicate edges to a binary adjacency
        a.setdiag(0)
        a.eliminate_zeros()
        return a

    def assert_no_future(self, adjacency_t: float, used_ts) -> None:
        if len(used_ts) and np.max(used_ts) >= adjacency_t:
            raise LeakageError(f"edge at {np.max(used_ts)} is not before the prediction time {adjacency_t}")


def normalized_laplacian(adj: sp.csr_matrix) -> sp.csr_matrix:
    """L = I - D^{-1/2} A D^{-1/2} (paper Eq. 2); isolated nodes get a zero row/column in the A term."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.zeros_like(deg, dtype=np.float32)
    nz = deg > 0
    inv[nz] = deg[nz] ** -0.5
    d = sp.diags(inv)
    return (sp.identity(adj.shape[0], dtype=np.float32, format="csr") - d @ adj @ d).tocsr()


def sample_neighbors(adj: sp.csr_matrix, nodes: np.ndarray, fanout: int, rng: np.random.Generator) -> np.ndarray:
    """Up to `fanout` neighbours (without replacement) of each node, returned as a flat array of unique node ids."""
    out = []
    for n in nodes:
        nb = adj.indices[adj.indptr[n]:adj.indptr[n + 1]]
        out.append(nb if len(nb) <= fanout else rng.choice(nb, fanout, replace=False))
    return np.unique(np.concatenate(out)) if out else np.array([], dtype=np.int64)


def induced_subgraph(adj: sp.csr_matrix, nodes: np.ndarray) -> sp.csr_matrix:
    return adj[nodes][:, nodes]


def ego_edges(g: TemporalGraph, center: int, t: float, hops: int, node_cap: int = 2000, window: Optional[float] = None):
    """Directed edges (src, dst, ts) reachable from `center` within `hops` hops using only edges before t.

    Returns (edges, complete): `complete` is True only if the breadth-first expansion finished without hitting
    `node_cap`, which is what licenses the verifier to claim a path is ABSENT within `hops`.
    """
    sl = g.edge_slice(t, window)
    s, d, ts = g.src[sl], g.dst[sl], g.ts[sl]
    order = np.argsort(s, kind="mergesort")
    s_sorted, d_sorted, ts_sorted = s[order], d[order], ts[order]
    starts = np.searchsorted(s_sorted, np.arange(g.n_nodes + 1))
    frontier, seen, edges, complete = [center], {center}, [], True
    for _ in range(hops):
        nxt = []
        for node in frontier:
            lo, hi = starts[node], starts[node + 1]
            for j in range(lo, hi):
                edges.append((int(node), int(d_sorted[j]), float(ts_sorted[j])))
                if d_sorted[j] not in seen:
                    seen.add(int(d_sorted[j]))
                    nxt.append(int(d_sorted[j]))
            if len(seen) > node_cap:
                complete = False
                return edges, complete
        frontier = nxt
    return edges, complete
