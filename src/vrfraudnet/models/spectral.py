"""Stage 0: time-conditioned spectral graph encoder (Sec. 3.1, Eqs. 1-2, Fig. 3).

Shapes (N nodes in the sampled subgraph, F input features, H = 64 hidden, K = 4 filters):

    x            [N, F]    node (or transaction-endpoint) features, train-fitted preprocessing already applied
    L            [N, N]    SPARSE normalized Laplacian I - D^-1/2 A D^-1/2 of the snapshot visible before time t
    t            scalar    evaluation time (normalised by the caller)
    emb(t)       [32]      learned sinusoidal time embedding
    theta(t)     [K]       2-layer MLP (GELU) of emb(t): time-conditioned filter coefficients
    B_k(L) h     [N, H]    Beta-kernel band-pass filter k applied to h = W_in x
    z_g          [N, 64]   sum_k theta_k(t) B_k(L) h, then Linear -> GELU

Beta kernels (Tang et al., 2022 style): with C = K - 1 and (p, q) = (k, C - k),
    B_k(L) = (L/2)^p (I - L/2)^q / Beta(p + 1, q + 1),
applied by repeated SPARSE products (never a dense matrix). The spectrum of L lies in [0, 2], so L/2 is in [0, 1].
D2 edge alignment (account embeddings -> transaction targets) is NOT specified by the paper; see `align_edges`.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

from ..errors import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise OptionalDependencyError("torch", "graph", "the spectral graph encoder") from e


def sparse_from_scipy(m, device=None):
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape, device=device).coalesce()


class TimeEmbedding(nn.Module):
    """emb(t) = [sin(w_i t), cos(w_i t)] with 16 learnable log-spaced frequencies -> 32 dimensions."""

    def __init__(self, dim: int = 32):
        super().__init__()
        assert dim % 2 == 0
        self.freq = nn.Parameter(torch.logspace(-2, 1, dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t.reshape(-1, 1) * self.freq
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1).squeeze(0)


class SpectralEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_filters: int = 4, time_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.K, self.C = n_filters, n_filters - 1
        self.inp = nn.Linear(in_dim, hidden)
        self.time = TimeEmbedding(time_dim)
        self.coef = nn.Sequential(nn.Linear(time_dim, 64), nn.GELU(), nn.Linear(64, n_filters))     # g_eta: 2-layer MLP, GELU
        self.out = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim
        self._norm = [1.0 / math.exp(math.lgamma(p + 1) + math.lgamma(self.C - p + 1) - math.lgamma(self.C + 2)) for p in range(n_filters)]  # 1/Beta(p+1, q+1)

    def filters(self, lap, h: torch.Tensor) -> list[torch.Tensor]:
        """[B_k(L) h for k in 0..K-1], each [N, H], using only sparse-dense products."""
        half = lambda v: torch.sparse.mm(lap, v) * 0.5                       # (L/2) v
        outs = []
        for p in range(self.K):
            q = self.C - p
            v = h
            for _ in range(q):
                v = v - half(v)                                               # (I - L/2) v
            for _ in range(p):
                v = half(v)                                                   # (L/2) v
            outs.append(v * self._norm[p])
        return outs

    def forward(self, x: torch.Tensor, lap, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        theta = self.coef(self.time(t))                                       # [K]
        mix = sum(theta[k] * f for k, f in enumerate(self.filters(lap, h)))   # [N, H]
        return torch.nn.functional.gelu(self.out(mix))                        # z_g: [N, 64]


AlignPolicy = Literal["mean", "src", "dst", "hadamard"]


def align_edges(z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, policy: AlignPolicy = "mean") -> torch.Tensor:
    """Map ACCOUNT embeddings to TRANSACTION (edge) targets. The paper says Stage 0 yields a 64-dim representation 'associated with the target
    transaction' but not how for D2 (accounts = nodes, transactions = edges). Every policy here keeps 64 dimensions; the default ('mean') and the
    alternatives are a RECONSTRUCTION recorded in the run manifest (docs/DECISIONS.md D-11)."""
    zs, zd = z[src], z[dst]
    if policy == "mean":
        return 0.5 * (zs + zd)
    if policy == "src":
        return zs
    if policy == "dst":
        return zd
    if policy == "hadamard":
        return zs * zd
    raise ValueError(f"unknown alignment policy {policy!r}")
