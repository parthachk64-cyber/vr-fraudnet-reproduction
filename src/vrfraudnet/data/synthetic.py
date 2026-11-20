"""A small, obviously invented transaction dataset for the demo, the tests and CI.

Every identifier says SYN. It has the properties the pipeline exercises: chronological order, a transaction graph in
which mule accounts matter, categorical and numeric features, and a label produced by a known stochastic rule. It is
NOT a benchmark: results on it are never comparable with the paper, and run manifests record `synthetic_fixture`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import TemporalGraph
from .providers import LoadedDataset
from .registry import SYN


def make_synthetic(n: int = 6000, n_accounts: int = 300, seed: int = 0, days: float = 10.0) -> LoadedDataset:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, days * 86400.0, n))
    mule = rng.random(n_accounts) < 0.06
    opened = -rng.exponential(120.0, n_accounts) * 86400.0                     # account opening times (days before t=0)
    org = rng.integers(0, n_accounts, n)
    # mule rings: mules send to mules more often
    ben = np.where(mule[org] & (rng.random(n) < 0.5), rng.choice(np.nonzero(mule)[0], n), rng.integers(0, n_accounts, n))
    pay = rng.choice(["wire", "ach", "card", "cash"], n, p=[0.2, 0.35, 0.35, 0.1])
    amount = np.exp(rng.normal(5.0, 1.2, n)) * np.where(pay == "wire", 3.0, 1.0)
    age_days = (t - opened[org]) / 86400.0
    # past-only count of this origin's transactions in the previous 24h
    recent = np.zeros(n)
    by: dict[int, list[float]] = {}
    for i in range(n):
        ts = by.setdefault(int(org[i]), [])
        recent[i] = sum(1 for x in ts[-50:] if t[i] - x <= 86400.0)
        ts.append(t[i])
    z = -5.2 + 0.9 * (np.log(amount) - 5.0) + 1.3 * (age_days < 20) + 1.6 * mule[ben] + 0.9 * (pay == "wire") + 0.25 * np.minimum(recent, 8) + 0.9 * mule[org]
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    names = np.array([f"ACC_SYN_{i:04d}" for i in range(n_accounts)])
    frame = pd.DataFrame({"__id": [f"TX_SYN_{i:06d}" for i in range(n)], "__label": y, "__time": t, "__origin": names[org], "__beneficiary": names[ben],
                          "amount": amount.round(2), "account_age_days": age_days.round(2), "payment_type": pay, "recent_transfer_count": recent,
                          "hour_of_day": ((t % 86400) // 3600).astype(int)})
    g = TemporalGraph(org, ben, t, n_accounts)
    feats = ["amount", "account_age_days", "payment_type", "recent_transfer_count", "hour_of_day"]
    return LoadedDataset(SYN, frame, g, feats, {"synthetic": True, "node_names": list(names), "categorical": ["payment_type"], "edge_src": org, "edge_dst": ben,
                                                "evidence_fields": ["amount", "account_age_days", "payment_type", "recent_transfer_count"],
                                                "realised_n_features": len(feats), "account_opened": opened})
