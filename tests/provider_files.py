"""Writes TINY invented files in each provider's documented format. Not benchmark data; exists to exercise the adapters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrfraudnet.data.registry import BAF_COLUMNS


def write_d1(root: Path, n_per_month: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = 8 * n_per_month
    df = pd.DataFrame({c: rng.random(n) for c in BAF_COLUMNS})
    df["fraud_bool"] = (rng.random(n) < 0.08).astype(int)
    df["month"] = np.repeat(np.arange(8), n_per_month)
    df["customer_age"] = rng.integers(10, 90, n)
    for c, vals in (("payment_type", ["AA", "AB", "AC"]), ("employment_status", ["CA", "CB"]), ("housing_status", ["BA", "BB"]), ("source", ["INTERNET", "TELEAPP"]), ("device_os", ["linux", "windows"])):
        df[c] = rng.choice(vals, n)
    df.to_csv(root / "Base.csv", index=False)
    return root


def write_d2(root: Path, n: int = 400, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    t = pd.Timestamp("2022-09-01") + pd.to_timedelta(np.sort(rng.integers(0, 3 * 24 * 60, n)), unit="m")
    df = pd.DataFrame({"Timestamp": t.strftime("%Y/%m/%d %H:%M"), "From Bank": rng.integers(1, 5, n), "Account": [f"A{i}" for i in rng.integers(0, 40, n)],
                       "To Bank": rng.integers(1, 5, n), "Account.1": [f"A{i}" for i in rng.integers(0, 40, n)], "Amount Received": rng.lognormal(5, 1, n).round(2),
                       "Receiving Currency": rng.choice(["US Dollar", "Euro"], n), "Amount Paid": rng.lognormal(5, 1, n).round(2),
                       "Payment Currency": rng.choice(["US Dollar", "Euro"], n), "Payment Format": rng.choice(["Wire", "Cheque", "ACH"], n),
                       "Is Laundering": (rng.random(n) < 0.05).astype(int)})
    df.to_csv(root / "HI-Small_Trans.csv", index=False)
    (root / "HI-Small_Patterns.txt").write_text("THIS FILE IS A LABEL ORACLE AND MUST NEVER BE READ\n", encoding="utf-8")
    return root


def write_d3(root: Path, n: int = 300, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    d = {"TransactionID": np.arange(1000, 1000 + n), "isFraud": (rng.random(n) < 0.06).astype(int), "TransactionDT": np.sort(rng.integers(86400, 86400 * 180, n)),
         "TransactionAmt": rng.lognormal(4, 1, n).round(2), "ProductCD": rng.choice(list("WCRHS"), n), "card4": rng.choice(["visa", "mastercard"], n)}
    for i in range(1, 16):
        d[f"D{i}"] = rng.random(n)
    for i in range(1, 340):
        d[f"V{i}"] = rng.random(n)
    pd.DataFrame(d).to_csv(root / "train_transaction.csv", index=False)
    keep = rng.random(n) < 0.4
    pd.DataFrame({"TransactionID": d["TransactionID"][keep], "id_01": rng.random(keep.sum()), "DeviceType": rng.choice(["mobile", "desktop"], keep.sum())}).to_csv(root / "train_identity.csv", index=False)
    return root


def write_d4(root: Path, n: int = 245, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    step = np.repeat(np.arange(1, 50), n // 49)
    n = len(step)
    ids = 9000 + np.arange(n)
    feats = pd.DataFrame(rng.random((n, 183)), columns=[f"f{i}" for i in range(183)])
    feats.insert(0, "Time step", step)
    feats.insert(0, "txId", ids)
    feats.to_csv(root / "txs_features.csv", index=False)
    pd.DataFrame({"txId": ids, "class": rng.choice([1, 2, 3], n, p=[0.1, 0.4, 0.5])}).to_csv(root / "txs_classes.csv", index=False)
    pd.DataFrame({"txId1": ids[rng.integers(0, n, 300)], "txId2": ids[rng.integers(0, n, 300)]}).to_csv(root / "txs_edgelist.csv", index=False)
    return root


def write_d5(root: Path, n: int = 200, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    e = 500
    np.savez(root / "dgraphfin.npz", x=rng.random((n, 17)).astype(np.float32), y=rng.choice([0, 1, 2, 3], n, p=[0.5, 0.05, 0.25, 0.2]),
             edge_index=rng.integers(0, n, (e, 2)), edge_type=rng.integers(1, 12, e), edge_timestamp=rng.integers(1, 800, e))
    return root
