"""Stage 0 training (Sec. 3.8.1): AdamW, lr 1e-3, weight decay 1e-4, up to 100 epochs, early stopping patience 15 on validation AUPRC,
ReduceLROnPlateau x0.5 after 5 stale epochs (floor 1e-5), gradient clipping 1.0, node mini-batches of 2,048, two-hop neighbour sampling 25/10.

The paper says the encoder is trained before the tree model but not with what signal. Here a small LINEAR HEAD on the 64-dim embedding is trained with
a class-balanced focal loss on the labelled targets; the head is discarded afterwards and the 64-dim embedding is what Stage 1 receives. (The tree
model's gradients do not reach the encoder: LightGBM is not differentiable. docs/DECISIONS.md D-11.)

Leakage control: samples are processed in TIME-ORDERED batches and each batch sees the graph snapshot at the batch's EARLIEST sample time, so no sample's
own transaction, or any later edge, is in its graph context.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..data.graph import TemporalGraph, induced_subgraph, normalized_laplacian, sample_neighbors
from ..errors import OptionalDependencyError
from ..evaluation.metrics import auprc
from ..published import ENCODER_PARAMS

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise OptionalDependencyError("torch", "graph", "encoder training") from e

from ..models.spectral import SpectralEncoder, align_edges, sparse_from_scipy


@dataclass
class Samples:
    """Targets for Stage 0: node targets have b == -1; edge targets (D2) give both endpoints. `t` is the sample's own time."""

    a: np.ndarray
    b: np.ndarray
    t: np.ndarray
    y: np.ndarray


def _batches(t: np.ndarray, size: int):
    order = np.argsort(t, kind="mergesort")
    return [order[i:i + size] for i in range(0, len(order), size)]


def embed_batch(model: SpectralEncoder, x_nodes: np.ndarray, graph: TemporalGraph, s: Samples, idx: np.ndarray, fanouts, rng, align: str, t_scale: float, device):
    t_batch = float(s.t[idx].min())                              # snapshot strictly before every sample in the batch
    adj = graph.snapshot(t_batch)
    seeds = np.unique(np.r_[s.a[idx], s.b[idx][s.b[idx] >= 0]])
    nodes, frontier = set(seeds.tolist()), seeds
    for fo in fanouts:
        frontier = sample_neighbors(adj, frontier, fo, rng)
        nodes.update(frontier.tolist())
    nodes = np.array(sorted(nodes))
    pos = {n: i for i, n in enumerate(nodes)}
    lap = sparse_from_scipy(normalized_laplacian(induced_subgraph(adj, nodes)), device)
    x = torch.tensor(x_nodes[nodes], dtype=torch.float32, device=device)
    z = model(x, lap, torch.tensor(t_batch / t_scale, dtype=torch.float32, device=device))
    ia = torch.tensor([pos[n] for n in s.a[idx]], device=device)
    if (s.b[idx] >= 0).any():
        ib = torch.tensor([pos[n] if n >= 0 else pos[s.a[idx][k]] for k, n in enumerate(s.b[idx])], device=device)
        return align_edges(z, ia, ib, align)
    return z[ia]


def focal_bce(logits, y, gamma: float = 2.0, a_pos: float = 1.0, a_neg: float = 1.0):
    p = torch.sigmoid(logits)
    loss_pos = -a_pos * (1 - p).pow(gamma) * torch.log(p.clamp_min(1e-8))
    loss_neg = -a_neg * p.pow(gamma) * torch.log((1 - p).clamp_min(1e-8))
    return torch.where(y > 0.5, loss_pos, loss_neg).mean()


def train_encoder(x_nodes: np.ndarray, graph: TemporalGraph, train: Samples, val: Samples, *, params: Optional[dict] = None, max_epochs: Optional[int] = None,
                  align: str = "mean", seed: int = 0, device: str = "cpu", batch_size: Optional[int] = None, verbose: bool = False, deterministic: bool = True):
    p = {**ENCODER_PARAMS, **(params or {})}
    if deterministic:                                   # multithreaded sparse products sum in a varying order; one thread makes runs bit-reproducible
        torch.set_num_threads(1)
    max_epochs = max_epochs or p["max_epochs"]
    batch_size = batch_size or p["node_batch_size"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpectralEncoder(x_nodes.shape[1], 64, p["n_filters"], p["time_embedding_dim"], p["output_dim"]).to(device)
    head = nn.Linear(p["output_dim"], 1).to(device)
    params_ = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params_, lr=p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=p["lr_decay"], patience=p["lr_decay_patience"], min_lr=p["min_lr"])
    t_scale = float(max(train.t.max(), val.t.max(), 1.0))
    n1, n0 = max(int((train.y == 1).sum()), 1), max(int((train.y == 0).sum()), 1)
    a_pos, a_neg = n0 / (n0 + n1) * 2 * 0.5 + 0.5, 1.0                      # mild positive up-weighting on top of the focal term
    best, best_state, stale, hist = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        batches = _batches(train.t, batch_size)
        rng.shuffle(batches)
        tot = 0.0
        for idx in batches:
            z = embed_batch(model, x_nodes, graph, train, idx, p["fanouts"], rng, align, t_scale, device)
            loss = focal_bce(head(z).squeeze(-1), torch.tensor(train.y[idx], dtype=torch.float32, device=device), 2.0, a_pos, a_neg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params_, p["grad_clip"])
            opt.step()
            tot += float(loss.detach()) * len(idx)
        v = predict_scores(model, head, x_nodes, graph, val, align, t_scale, device, p["fanouts"], seed)
        score = auprc(val.y, v) if val.y.sum() > 0 else 0.0
        sched.step(score)
        hist.append({"epoch": epoch, "train_loss": tot / len(train.y), "val_auprc": score, "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(hist[-1])
        if score > best + 1e-9:
            best, stale, best_state = score, 0, (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
        else:
            stale += 1
            if stale >= p["patience"]:
                break
    model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    return model, head, {"history": hist, "best_val_auprc": best, "t_scale": t_scale, "align": align, "epochs_run": len(hist)}


@torch.no_grad()
def predict_scores(model, head, x_nodes, graph, s: Samples, align, t_scale, device, fanouts, seed=0) -> np.ndarray:
    model.eval()
    rng = np.random.default_rng(seed)
    out = np.zeros(len(s.y))
    for idx in _batches(s.t, 2048):
        z = embed_batch(model, x_nodes, graph, s, idx, fanouts, rng, align, t_scale, device)
        out[idx] = torch.sigmoid(head(z).squeeze(-1)).cpu().numpy()
    return out


@torch.no_grad()
def encode(model, x_nodes, graph, s: Samples, align, t_scale, device="cpu", fanouts=(25, 10), seed=0) -> np.ndarray:
    """64-dim Stage-0 representation for each sample, in the samples' original order (this is what Stage 1 concatenates to the record features)."""
    model.eval()
    rng = np.random.default_rng(seed)
    out = np.zeros((len(s.y), model.out_dim), dtype=np.float32)
    for idx in _batches(s.t, 2048):
        out[idx] = embed_batch(model, x_nodes, graph, s, idx, fanouts, rng, align, t_scale, device).cpu().numpy()
    return out
