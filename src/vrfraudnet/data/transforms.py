"""Train-only transformers. `fit` sees training rows only; `transform` is applied unchanged to everything else.

Every statistic (scaling, imputation medians, category priors, PCA axes, per-currency moments) is estimated from the
training partition and stored; none is re-estimated on validation, calibration or test rows.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler

from ..errors import ProtocolError


class SmoothedTargetEncoder:
    """enc(c) = (n_c * mean_c + m * prior) / (n_c + m). Unseen categories map to the prior.

    `fit_transform_train` returns OUT-OF-FOLD encodings for the training rows themselves (contiguous folds), so a row's
    own label never leaks into its own feature; `transform` uses the full-train mapping for every other partition.
    """

    def __init__(self, smoothing: float = 20.0, n_folds: int = 5):
        self.m, self.n_folds = smoothing, n_folds
        self.prior_: float = 0.0
        self.table_: dict = {}

    def _table(self, x: pd.Series, y: np.ndarray, prior: float) -> dict:
        df = pd.DataFrame({"c": x.astype(object).fillna("__nan__").to_numpy(), "y": y})
        g = df.groupby("c")["y"].agg(["sum", "count"])
        return ((g["sum"] + self.m * prior) / (g["count"] + self.m)).to_dict()

    def fit(self, x: pd.Series, y) -> "SmoothedTargetEncoder":
        y = np.asarray(y, float)
        self.prior_ = float(y.mean())
        self.table_ = self._table(x, y, self.prior_)
        return self

    def transform(self, x: pd.Series) -> np.ndarray:
        return x.astype(object).fillna("__nan__").map(self.table_).fillna(self.prior_).to_numpy(dtype=float)

    def fit_transform_train(self, x: pd.Series, y) -> np.ndarray:
        y = np.asarray(y, float)
        self.fit(x, y)
        out = np.empty(len(x))
        folds = np.array_split(np.arange(len(x)), self.n_folds)
        for f in folds:
            rest = np.setdiff1d(np.arange(len(x)), f)
            prior = float(y[rest].mean())
            tbl = self._table(x.iloc[rest], y[rest], prior)
            out[f] = x.iloc[f].astype(object).fillna("__nan__").map(tbl).fillna(prior).to_numpy(dtype=float)
        return out


def hash64_bucket(value, n_buckets: int) -> int:
    """64-bit hash (blake2b) reduced to a bucket: stable across runs and platforms (unlike Python's hash())."""
    return int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=8).digest(), "little") % n_buckets


class CategoricalEncoder:
    """One-hot for categories seen in training (up to `max_onehot`), 64-bit hashing into `n_buckets` for the rest/high cardinality."""

    def __init__(self, max_onehot: int = 20, n_buckets: int = 16):
        self.max_onehot, self.n_buckets = max_onehot, n_buckets
        self.levels_: Optional[list] = None

    def fit(self, x: pd.Series) -> "CategoricalEncoder":
        vc = x.astype(object).fillna("__nan__").value_counts()
        self.levels_ = list(vc.index) if len(vc) <= self.max_onehot else None
        return self

    @property
    def width(self) -> int:
        return len(self.levels_) + 1 if self.levels_ is not None else self.n_buckets

    def transform(self, x: pd.Series) -> np.ndarray:
        v = x.astype(object).fillna("__nan__").to_numpy()
        out = np.zeros((len(v), self.width), dtype=np.float32)
        if self.levels_ is not None:
            idx = {l: i for i, l in enumerate(self.levels_)}
            for r, val in enumerate(v):
                out[r, idx.get(val, len(self.levels_))] = 1.0         # last column = unseen in training
        else:
            for r, val in enumerate(v):
                out[r, hash64_bucket(val, self.n_buckets)] = 1.0
        return out


class PerGroupScaler:
    """Standardise within a group (currency, time step). Groups never seen in training use POOLED TRAINING statistics (D-12):
    estimating them from the unseen group's own rows would fit on evaluation data."""

    def __init__(self):
        self.stats_: dict = {}
        self.pooled_: tuple = (0.0, 1.0)

    def fit(self, x: np.ndarray, groups) -> "PerGroupScaler":
        x = np.asarray(x, float)
        self.pooled_ = (float(np.nanmean(x)), float(np.nanstd(x)) or 1.0)
        df = pd.DataFrame({"g": np.asarray(groups), "x": x}).groupby("g")["x"].agg(["mean", "std"])
        self.stats_ = {g: (float(r["mean"]), float(r["std"]) if r["std"] and np.isfinite(r["std"]) else 1.0) for g, r in df.iterrows()}
        return self

    def transform(self, x: np.ndarray, groups) -> np.ndarray:
        x = np.asarray(x, float)
        mu = np.array([self.stats_.get(g, self.pooled_)[0] for g in np.asarray(groups)])
        sd = np.array([self.stats_.get(g, self.pooled_)[1] for g in np.asarray(groups)])
        return (x - mu) / np.where(sd == 0, 1.0, sd)


class TabularPreprocessor:
    """numeric (median-impute then robust/standard scale) + target-encoded + one-hot/hashed categorical columns + optional PCA block.

    `fit(frame, y)` must be given TRAINING rows only (the caller passes `frame.iloc[train]`).
    """

    def __init__(self, numeric: Sequence[str], target_encoded: Sequence[str] = (), categorical: Sequence[str] = (), pca_columns: Sequence[str] = (),
                 pca_components: int = 50, scaler: str = "robust", log1p_columns: Sequence[str] = (), seed: int = 0,
                 group_scale: Optional[dict] = None):
        self.numeric, self.te_cols, self.cat_cols = list(numeric), list(target_encoded), list(categorical)
        self.pca_cols, self.pca_k, self.scaler_name, self.log1p, self.seed = list(pca_columns), pca_components, scaler, list(log1p_columns), seed
        self.group_scale = dict(group_scale or {})                 # numeric column -> grouping column (per-currency, per-time-step)
        self.numeric = [c for c in self.numeric if c not in self.group_scale]
        self.fitted_ = False

    def _num(self, f: pd.DataFrame) -> np.ndarray:
        x = f[self.numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) if self.numeric else np.zeros((len(f), 0))
        for c in self.log1p:
            if c in self.numeric:
                i = self.numeric.index(c)
                x[:, i] = np.log1p(np.clip(x[:, i], 0, None))
        return x

    def _gs_values(self, frame: pd.DataFrame, col: str) -> np.ndarray:
        x = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        return np.log1p(np.clip(x, 0, None)) if col in self.log1p else x

    def fit(self, frame: pd.DataFrame, y) -> "TabularPreprocessor":
        y = np.asarray(y)
        self.gs_ = {}
        for c, g in self.group_scale.items():
            v = self._gs_values(frame, c)
            med = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
            self.gs_[c] = (med, PerGroupScaler().fit(np.where(np.isnan(v), med, v), frame[g].to_numpy()))
        x = self._num(frame)
        self.medians_ = np.nanmedian(x, axis=0) if x.shape[1] else np.array([])
        xi = np.where(np.isnan(x), self.medians_, x)
        self.scaler_ = (RobustScaler() if self.scaler_name == "robust" else StandardScaler()).fit(xi) if x.shape[1] else None
        self.te_ = {c: SmoothedTargetEncoder().fit(frame[c], y) for c in self.te_cols}
        self.cat_ = {c: CategoricalEncoder().fit(frame[c]) for c in self.cat_cols}
        if self.pca_cols:
            v = frame[self.pca_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            self.pca_med_ = np.nanmedian(v, axis=0)
            self.pca_med_ = np.where(np.isnan(self.pca_med_), 0.0, self.pca_med_)
            vi = np.where(np.isnan(v), self.pca_med_, v)
            self.pca_scaler_ = StandardScaler().fit(vi)
            self.pca_ = PCA(n_components=min(self.pca_k, vi.shape[1], max(1, vi.shape[0] - 1)), random_state=self.seed).fit(self.pca_scaler_.transform(vi))
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame, y_train_oof: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.fitted_:
            raise ProtocolError("TabularPreprocessor.fit must be called on the training rows first")
        blocks = []
        if self.numeric:
            x = self._num(frame)
            blocks.append(self.scaler_.transform(np.where(np.isnan(x), self.medians_, x)))
        for c, g in self.group_scale.items():
            med, sc = self.gs_[c]
            v = self._gs_values(frame, c)
            blocks.append(sc.transform(np.where(np.isnan(v), med, v), frame[g].to_numpy())[:, None].astype(np.float32))
        for c in self.te_cols:
            blocks.append(self.te_[c].transform(frame[c])[:, None])
        for c in self.cat_cols:
            blocks.append(self.cat_[c].transform(frame[c]))
        if self.pca_cols:
            v = frame[self.pca_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            blocks.append(self.pca_.transform(self.pca_scaler_.transform(np.where(np.isnan(v), self.pca_med_, v))))
        return np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(frame), 0), np.float32)

    def fit_transform_train(self, frame: pd.DataFrame, y) -> np.ndarray:
        """Fit on training rows and return their features, using OUT-OF-FOLD target encodings for the training rows."""
        self.fit(frame, y)
        out = self.transform(frame)
        if self.te_cols:
            start = (len(self.numeric) if self.numeric else 0) + len(self.group_scale)
            for j, c in enumerate(self.te_cols):
                out[:, start + j] = self.te_[c].fit_transform_train(frame[c], y)
                self.te_[c].fit(frame[c], y)                     # restore the full-train mapping for later transform() calls
        return out

    @property
    def n_features_(self) -> int:
        n = len(self.numeric) + len(self.group_scale) + len(self.te_cols) + sum(e.width for e in self.cat_.values())
        return n + (self.pca_.n_components_ if self.pca_cols else 0)
