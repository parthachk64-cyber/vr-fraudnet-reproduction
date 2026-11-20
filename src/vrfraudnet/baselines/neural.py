"""MLP, 1D-CNN, LSTM and TabTransformer baselines (PyTorch). Small, deterministic given the seed, trained with a positive-class weight and early
stopping on validation AUPRC. Architectures are reconstructions (the paper names the families only)."""
from __future__ import annotations

import copy

import numpy as np

from ..errors import OptionalDependencyError
from ..evaluation.metrics import auprc

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise OptionalDependencyError("torch", "graph", "the neural baselines") from e


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    """Convolution along the feature axis (features treated as a 1-D signal)."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(8))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)


class LSTM(nn.Module):
    """LSTM over the feature sequence (one scalar per step)."""

    def __init__(self, d):
        super().__init__()
        self.rnn = nn.LSTM(1, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.rnn(x.unsqueeze(-1))
        return self.head(h[-1]).squeeze(-1)


class TabTransformer(nn.Module):
    """Each feature value becomes a token (value * w_i + b_i), a CLS token is prepended, 2 transformer layers, CLS -> logit."""

    def __init__(self, d, dim=32):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.randn(d, dim) * 0.1), nn.Parameter(torch.zeros(d, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, 64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        tok = x.unsqueeze(-1) * self.w + self.b
        h = self.enc(torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], 1))
        return self.head(h[:, 0]).squeeze(-1)


ARCH = {"mlp": MLP, "cnn1d": CNN1D, "lstm": LSTM, "tabtransformer": TabTransformer}


class NeuralBaseline:
    def __init__(self, name: str, seed: int = 0, epochs: int = 30, batch_size: int = 512, lr: float = 1e-3, patience: int = 5):
        self.name, self.seed, self.epochs, self.bs, self.lr, self.patience = name, seed, epochs, batch_size, lr, patience
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.model = ARCH[self.name](X.shape[1])
        pos_w = torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1)).__pow__(0.5))
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, stale = -1.0, copy.deepcopy(self.model.state_dict()), 0
        for _ in range(self.epochs):
            self.model.train()
            for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // self.bs)):
                opt.zero_grad()
                lossf(self.model(X[idx]), yt[idx]).backward()
                opt.step()
            if X_val is not None and np.asarray(y_val).sum() > 0:
                v = auprc(y_val, self.score(X_val))
                if v > best + 1e-9:
                    best, best_state, stale = v, copy.deepcopy(self.model.state_dict()), 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if X_val is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        out = [torch.sigmoid(self.model(torch.tensor(np.asarray(X[i:i + 4096]), dtype=torch.float32))).numpy() for i in range(0, len(X), 4096)]
        return np.concatenate(out)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
