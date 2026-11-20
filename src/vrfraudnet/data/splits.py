"""Chronological split contracts. A split is a set of row POSITIONS per named partition, with a fingerprint.

Standard frame columns (set by the provider adapters): `__id` (str), `__time` (float, strictly comparable within a
dataset), `__label` (1 fraud, 0 legitimate, -1 unlabeled/background), and for some datasets `__group`
(D1 month, D4 time step).

Validation is divided into two equal chronological halves with distinct purposes (Sec. 3.8.1): the FIRST half is for
hyperparameters, early stopping, routing thresholds and the isotonic mixer; the SECOND half is held out exclusively
for split-conformal calibration. Test, shock, strict-inductive and robustness partitions are never used for selection.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from ..errors import LeakageError, ProtocolError


@dataclass
class SplitManifest:
    dataset: str
    partitions: dict[str, np.ndarray]
    notes: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def assert_disjoint(self) -> None:
        for (a, pa), (b, pb) in combinations(self.partitions.items(), 2):
            if a == "context" or b == "context":
                continue                                     # D5 background nodes may be context for message passing only
            inter = np.intersect1d(pa, pb)
            if len(inter):
                raise LeakageError(f"partitions {a!r} and {b!r} share {len(inter)} row(s)")

    def assert_chronological(self, time: np.ndarray, order=("train", "val_select", "val_calibration", "test")) -> None:
        prev_name, prev_max = None, -np.inf
        for name in order:
            if name not in self.partitions or len(self.partitions[name]) == 0:
                continue
            tmin, tmax = time[self.partitions[name]].min(), time[self.partitions[name]].max()
            if tmin < prev_max:
                raise LeakageError(f"partition {name!r} starts at {tmin}, before {prev_name!r} ends at {prev_max}")
            prev_name, prev_max = name, tmax

    def fingerprint(self, ids) -> str:
        h = hashlib.sha256()
        for name in sorted(self.partitions):
            h.update(name.encode())
            h.update(hashlib.sha256("\n".join(map(str, np.asarray(ids)[np.sort(self.partitions[name])])).encode()).digest())
        h.update(repr(sorted(self.params.items())).encode())
        return h.hexdigest()

    def prevalence(self, labels) -> dict:
        y = np.asarray(labels)
        out = {}
        for name, pos in self.partitions.items():
            yy = y[pos]
            lab = yy[yy >= 0]
            out[name] = {"n": int(len(pos)), "n_labeled": int(len(lab)), "n_positive": int((lab == 1).sum()),
                         "prevalence": float((lab == 1).mean()) if len(lab) else None}
        return out


def _order(frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """Positions sorted by (__time, position): a stable chronological order."""
    t = frame["__time"].to_numpy()[positions]
    return positions[np.lexsort((positions, t))]


def halve_validation(val_positions: np.ndarray, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(selection, conformal_calibration): the first and second chronological halves of validation."""
    v = _order(frame, np.asarray(val_positions))
    h = len(v) // 2
    return v[:h], v[h:]


def _finish(name: str, frame, train, val, test, notes, params, extra: Optional[dict] = None) -> SplitManifest:
    sel, cal = halve_validation(val, frame)
    parts = {"train": _order(frame, train), "val_select": sel, "val_calibration": cal, "test": _order(frame, test)}
    if extra:
        parts.update(extra)
    m = SplitManifest(name, parts, notes, params)
    m.assert_disjoint()
    return m


def split_d1(frame: pd.DataFrame, *, dev_max_month: int = 5, validation_fraction: float = 0.10) -> SplitManifest:
    g = frame["__group"].to_numpy()
    pos = np.arange(len(frame))
    dev = _order(frame, pos[g <= dev_max_month])
    test = pos[g > dev_max_month]
    nv = int(round(validation_fraction * len(dev)))
    notes = [f"D1: months <= {dev_max_month} development; last {validation_fraction:.0%} of that period (stable order by month, then file row) is validation",
             "Order within a month is the file's row order: the dataset has no finer timestamp (docs/DECISIONS.md D-13)."]
    return _finish("D1", frame, dev[: len(dev) - nv], dev[len(dev) - nv:], test, notes, {"dev_max_month": dev_max_month, "validation_fraction": validation_fraction})


def split_d2(frame: pd.DataFrame, fractions=(0.6, 0.2, 0.2)) -> SplitManifest:
    order = _order(frame, np.arange(len(frame)))
    n = len(order)
    a, b = int(round(fractions[0] * n)), int(round((fractions[0] + fractions[1]) * n))
    return _finish("D2", frame, order[:a], order[a:b], order[b:], ["D2: time-sorted 60/20/20 by transaction timestamp"], {"fractions": tuple(fractions)})


def split_d3(frame: pd.DataFrame, *, q: float = 0.80, validation_fraction: float = 0.10) -> SplitManifest:
    t = frame["__time"].to_numpy()
    cut = np.quantile(t, q)
    pos = np.arange(len(frame))
    dev = _order(frame, pos[t < cut])
    test = pos[t >= cut]
    nv = int(round(validation_fraction * len(dev)))
    notes = [f"D3: TransactionDT < {q}-quantile ({cut:.0f}) development, >= test; validation fraction {validation_fraction:.0%} of development is an ASSUMPTION (not stated)"]
    return _finish("D3", frame, dev[: len(dev) - nv], dev[len(dev) - nv:], test, notes, {"q": q, "cut": float(cut), "validation_fraction": validation_fraction})


def split_d4(frame: pd.DataFrame, *, dev_max_step: int = 34, validation_fraction: float = 0.10) -> SplitManifest:
    g, y = frame["__group"].to_numpy(), frame["__label"].to_numpy()
    pos = np.arange(len(frame))
    lab = y >= 0
    dev = _order(frame, pos[(g <= dev_max_step) & lab])
    test = pos[(g > dev_max_step) & lab]
    nv = int(round(validation_fraction * len(dev)))
    ctx = pos[~lab]
    notes = [f"D4: strict-inductive; steps <= {dev_max_step} development, later steps test; unknown-label nodes are context only (never supervised, never scored)"]
    return _finish("D4", frame, dev[: len(dev) - nv], dev[len(dev) - nv:], test, notes, {"dev_max_step": dev_max_step, "validation_fraction": validation_fraction}, {"context": ctx})


def split_d5(frame: pd.DataFrame, *, fractions=(0.70, 0.15, 0.15)) -> SplitManifest:
    y = frame["__label"].to_numpy()
    pos = np.arange(len(frame))
    lab = _order(frame, pos[y >= 0])
    n = len(lab)
    a, b = int(round(fractions[0] * n)), int(round((fractions[0] + fractions[1]) * n))
    ctx = pos[y < 0]
    notes = ["D5: temporal node-arrival split over labelled nodes; cut points are NOT stated by the paper (70/15/15 assumed, recorded)"]
    return _finish("D5", frame, lab[:a], lab[a:b], lab[b:], notes, {"fractions": tuple(fractions)}, {"context": ctx})


def split_syn(frame: pd.DataFrame, fractions=(0.6, 0.2, 0.2)) -> SplitManifest:
    m = split_d2(frame, fractions)
    m.dataset = "SYN"
    m.notes = ["SYN: synthetic fixture, 60/20/20 by time"]
    return m


SPLITTERS = {"D1": split_d1, "D2": split_d2, "D3": split_d3, "D4": split_d4, "D5": split_d5, "SYN": split_syn}


def make_split(dataset: str, frame: pd.DataFrame, **kw) -> SplitManifest:
    for c in ("__id", "__time", "__label"):
        if c not in frame:
            raise ProtocolError(f"frame lacks the standard column {c!r}")
    m = SPLITTERS[dataset.upper()](frame, **kw)
    m.assert_chronological(frame["__time"].to_numpy()) if dataset.upper() in ("D2", "D3", "SYN") else None
    return m


# ------------------------------------------------------------------- evaluation windows (not used for selection)
def d1_month_window(frame: pd.DataFrame, month: int) -> np.ndarray:
    return np.nonzero(frame["__group"].to_numpy() == month)[0]


def d3_expanding_fold(frame: pd.DataFrame, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-origin, moving-cut, test-to-end: train = TransactionDT < q-quantile, test = TransactionDT >= it (Table 8d)."""
    t = frame["__time"].to_numpy()
    cut = np.quantile(t, q)
    return np.nonzero(t < cut)[0], np.nonzero(t >= cut)[0]


def d4_shock_windows(frame: pd.DataFrame, shock_step: int = 43, test_start: int = 35) -> dict[str, np.ndarray]:
    """Pre-shock = labelled test steps in [test_start, shock_step); post-shock = steps >= shock_step. `shock_step` is configurable and
    NOT stated by the paper (default 43 follows the Elliptic literature; recorded in the run manifest)."""
    g, y = frame["__group"].to_numpy(), frame["__label"].to_numpy()
    lab = y >= 0
    return {"pre_shock": np.nonzero(lab & (g >= test_start) & (g < shock_step))[0], "post_shock": np.nonzero(lab & (g >= shock_step))[0]}
