"""Provider adapters: raw download -> standard frame (+ graph). One adapter per benchmark.

The raw files are never modified or copied. A loader fails early, naming the missing file, the provider page and the
access terms. `check_dataset` compares your files with the registry's expectations and records checksums.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..errors import MissingInputError, ProtocolError
from .graph import TemporalGraph
from .registry import REGISTRY, DatasetSpec, get
from .transforms import TabularPreprocessor


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    frame: pd.DataFrame                       # standard columns __id, __time, __label (+ __group, entities) and feature columns
    graph: Optional[TemporalGraph] = None
    features: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _need(spec: DatasetSpec, raw_dir: Path, name: str) -> Path:
    p = Path(raw_dir) / name
    if not p.is_file():
        raise MissingInputError(f"{spec.id} file {name!r} in {raw_dir}", needed_for=f"loading {spec.name}",
                                how_to_fix=f"Download it from {spec.page}. Access: {spec.access}")
    return p


# --------------------------------------------------------------------------------------------- check
def check_dataset(dataset_id: str, raw_dir: Path, *, hash_files: bool = True) -> dict:
    """Report what is present and what differs from expectations. Never raises for a MISSING file: it reports it."""
    spec = get(dataset_id)
    report = {"dataset": spec.id, "name": spec.name, "page": spec.page, "access": spec.access, "files": {}, "ok": True, "problems": []}
    for ef in spec.files:
        p = Path(raw_dir) / ef.name
        entry = {"present": p.is_file(), "note": ef.note}
        if not p.is_file():
            if "OPTIONAL" not in ef.note:
                report["ok"] = False
                report["problems"].append(f"missing file {ef.name}")
            report["files"][ef.name] = entry
            continue
        entry["bytes"] = p.stat().st_size
        if hash_files:
            entry["sha256"] = _sha256(p)
        if p.suffix == ".csv":
            cols = list(pd.read_csv(p, nrows=0).columns)
            entry["n_columns"] = len(cols)
            miss = [c for c in ef.required_columns if c not in cols]
            entry["missing_columns"] = miss
            if miss:
                report["ok"] = False
                report["problems"].append(f"{ef.name}: missing columns {miss}")
            with p.open("rb") as f:
                entry["n_rows"] = max(sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 22), b"")) - 1, 0)
        elif p.suffix == ".npz":
            z = np.load(p, allow_pickle=False)
            entry["arrays"] = {k: list(z[k].shape) for k in z.files}
            miss = [c for c in ef.required_columns if c not in z.files]
            entry["missing_arrays"] = miss
            if miss:
                report["ok"] = False
                report["problems"].append(f"{ef.name}: missing arrays {miss}")
        report["files"][ef.name] = entry
        if spec.stated_rows and "n_rows" in entry and abs(entry["n_rows"] - spec.stated_rows) > 0 and ef is spec.files[0]:
            report["problems"].append(f"{ef.name}: {entry['n_rows']:,} rows, paper states {spec.stated_rows:,}")
    return report


# --------------------------------------------------------------------------------------------- loaders
D1_CATEGORICAL = ("payment_type", "employment_status", "housing_status", "source", "device_os")


def load_d1(raw_dir: Path, *, include_month_feature: bool = False) -> LoadedDataset:
    spec = get("D1")
    df = pd.read_csv(_need(spec, raw_dir, "Base.csv"))
    n = len(df)
    frame = pd.DataFrame({"__id": [f"BAF_{i}" for i in range(n)], "__label": df["fraud_bool"].astype(int).to_numpy(),
                          "__group": df["month"].astype(int).to_numpy(), "__time": df["month"].to_numpy() * 1e7 + np.arange(n),
                          "__protected_age": df["customer_age"].to_numpy()})
    feats = [c for c in df.columns if c not in ("fraud_bool", "customer_age") and (include_month_feature or c != "month")]
    frame = pd.concat([frame, df[feats].reset_index(drop=True)], axis=1)
    return LoadedDataset(spec, frame, None, feats, {"categorical": [c for c in D1_CATEGORICAL if c in feats], "realised_n_features": len(feats),
                                                    "evidence_fields": [c for c in feats if c not in D1_CATEGORICAL][:12] + [c for c in D1_CATEGORICAL if c in feats]})


def load_d2(raw_dir: Path) -> LoadedDataset:
    spec = get("D2")
    df = pd.read_csv(_need(spec, raw_dir, "HI-Small_Trans.csv"))      # the Patterns file is never opened: it is a label oracle
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M").astype("int64") / 1e9
    src_key = df["From Bank"].astype(str) + "|" + df["Account"].astype(str)
    dst_key = df["To Bank"].astype(str) + "|" + df["Account.1"].astype(str)
    codes, uniq = pd.factorize(pd.concat([src_key, dst_key], ignore_index=True))
    n = len(df)
    frame = pd.DataFrame({"__id": [f"AML_{i}" for i in range(n)], "__label": df["Is Laundering"].astype(int).to_numpy(), "__time": ts.to_numpy(),
                          "__origin": src_key.to_numpy(), "__beneficiary": dst_key.to_numpy(),
                          "amount_received": df["Amount Received"].to_numpy(), "amount_paid": df["Amount Paid"].to_numpy(),
                          "receiving_currency": df["Receiving Currency"].to_numpy(), "payment_currency": df["Payment Currency"].to_numpy(),
                          "payment_format": df["Payment Format"].to_numpy(), "from_bank": df["From Bank"].astype(str).to_numpy(),
                          "to_bank": df["To Bank"].astype(str).to_numpy(), "hour_of_day": ((ts.to_numpy() % 86400) // 3600)})
    g = TemporalGraph(codes[:n], codes[n:], ts.to_numpy(), len(uniq))
    feats = ["amount_received", "amount_paid", "receiving_currency", "payment_currency", "payment_format", "from_bank", "to_bank", "hour_of_day"]
    return LoadedDataset(spec, frame, g, feats, {"node_names": list(uniq), "categorical": ["receiving_currency", "payment_currency", "payment_format", "from_bank", "to_bank"],
                                                  "edge_src": codes[:n], "edge_dst": codes[n:], "evidence_fields": ["amount_received", "amount_paid", "payment_format", "receiving_currency"],
                                                  "realised_n_features": len(feats)})


def load_d3(raw_dir: Path) -> LoadedDataset:
    spec = get("D3")
    tr = pd.read_csv(_need(spec, raw_dir, "train_transaction.csv"))
    idn = pd.read_csv(_need(spec, raw_dir, "train_identity.csv"))
    df = tr.merge(idn, on="TransactionID", how="left")              # identity columns missing => NaN within the record
    drop = set(spec.drop_columns) | {"TransactionDT"}
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    base = [c for c in df.columns if c not in drop and c not in v_cols]
    frame = pd.DataFrame({"__id": df["TransactionID"].astype(str).to_numpy(), "__label": df["isFraud"].astype(int).to_numpy(), "__time": df["TransactionDT"].astype(float).to_numpy()})
    frame = pd.concat([frame, df[base + v_cols].reset_index(drop=True)], axis=1)
    cat = [c for c in base if not pd.api.types.is_numeric_dtype(df[c])]
    return LoadedDataset(spec, frame, None, base + v_cols, {"categorical": cat, "v_columns": v_cols, "numeric": [c for c in base if c not in cat],
                                                           "evidence_fields": ["TransactionAmt", "ProductCD", "card4", "card6"], "realised_n_features": len(base) + len(v_cols)})


def load_d4(raw_dir: Path, *, n_local: int = 94) -> LoadedDataset:
    spec = get("D4")
    feats = pd.read_csv(_need(spec, raw_dir, "txs_features.csv"))
    cls = pd.read_csv(_need(spec, raw_dir, "txs_classes.csv"))
    edges = pd.read_csv(_need(spec, raw_dir, "txs_edgelist.csv"))
    idc, stepc = feats.columns[0], feats.columns[1]
    num_cols = list(feats.columns[2:])
    local = num_cols[:n_local]                                         # the remaining 89 are pre-aggregated neighbour features: dropped
    cls = cls.set_index(cls.columns[0])[cls.columns[1]]
    lab = feats[idc].map(cls).astype(float).map({1.0: 1, 2.0: 0, 3.0: -1}).fillna(-1).astype(int)
    n = len(feats)
    frame = pd.DataFrame({"__id": feats[idc].astype(str).to_numpy(), "__label": lab.to_numpy(), "__group": feats[stepc].astype(int).to_numpy(),
                          "__time": feats[stepc].astype(float).to_numpy()})
    frame = pd.concat([frame, feats[local].reset_index(drop=True)], axis=1)
    index = pd.Series(np.arange(n), index=feats[idc].to_numpy())
    s, d = edges.iloc[:, 0].map(index), edges.iloc[:, 1].map(index)
    ok = s.notna() & d.notna()
    ts = feats[stepc].to_numpy(dtype=float)[s[ok].astype(int).to_numpy()]
    g = TemporalGraph(s[ok].astype(int).to_numpy(), d[ok].astype(int).to_numpy(), ts, n)
    return LoadedDataset(spec, frame, g, local, {"n_dropped_aggregated": max(len(num_cols) - n_local, 0), "realised_n_features": len(local),
                                                  "evidence_fields": local[:6], "edge_time_note": "edge time = time step of the source transaction"})


def load_d5(raw_dir: Path) -> LoadedDataset:
    spec = get("D5")
    z = np.load(_need(spec, raw_dir, "dgraphfin.npz"), allow_pickle=False)
    x, y, ei, et, ets = z["x"], z["y"], z["edge_index"], z["edge_type"], z["edge_timestamp"]
    n = len(y)
    arrival = np.full(n, np.inf)
    np.minimum.at(arrival, ei[:, 0], ets)
    np.minimum.at(arrival, ei[:, 1], ets)
    arrival[~np.isfinite(arrival)] = ets.max() + 1                      # nodes with no edge arrive last (derivation recorded below)
    label = np.where(np.isin(y, (0, 1)), y, -1)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame.insert(0, "__time", arrival.astype(float))
    frame.insert(0, "__label", label.astype(int))
    frame.insert(0, "__id", [f"DGF_{i}" for i in range(n)])
    g = TemporalGraph(ei[:, 0], ei[:, 1], ets, n)
    g.etype = et[np.argsort(ets, kind="mergesort")]
    return LoadedDataset(spec, frame, g, [f"x{i}" for i in range(x.shape[1])],
                         {"arrival": "derived: earliest incident edge timestamp; isolated nodes get max+1", "evidence_fields": ["x0", "x1", "x2"], "realised_n_features": x.shape[1]})


LOADERS = {"D1": load_d1, "D2": load_d2, "D3": load_d3, "D4": load_d4, "D5": load_d5}


def load_dataset(dataset_id: str, raw_dir: Optional[Path], **kw) -> LoadedDataset:
    ds = dataset_id.upper()
    if ds == "SYN":
        from .synthetic import make_synthetic
        return make_synthetic(**kw)
    if raw_dir is None:
        raise MissingInputError(f"--raw-dir for {ds}", needed_for="loading a benchmark dataset", how_to_fix=f"Download from {get(ds).page}. {get(ds).access}")
    return LOADERS[ds](Path(raw_dir), **kw)


# --------------------------------------------------------------------------------------------- preprocessing recipes
def preprocessor_for(ds: LoadedDataset, seed: int = 0) -> TabularPreprocessor:
    """The dataset-specific preprocessing recipe of Table 2. Fit it on the TRAINING rows only."""
    i = ds.spec.id
    cat = ds.meta.get("categorical", [])
    num = [c for c in ds.features if c not in cat]
    if i == "D1":
        return TabularPreprocessor(numeric=num, target_encoded=cat, scaler="robust", seed=seed)
    if i == "D2":
        return TabularPreprocessor(numeric=["hour_of_day"], categorical=cat, log1p_columns=["amount_received", "amount_paid"],
                                   group_scale={"amount_received": "receiving_currency", "amount_paid": "payment_currency"}, scaler="standard", seed=seed)
    if i == "D3":
        v = ds.meta["v_columns"]
        return TabularPreprocessor(numeric=[c for c in ds.meta["numeric"]], categorical=cat, pca_columns=v, pca_components=50, scaler="standard", seed=seed)
    if i == "D4":
        return TabularPreprocessor(numeric=[], group_scale={c: "__group" for c in ds.features}, scaler="standard", seed=seed)
    if i == "D5":
        return TabularPreprocessor(numeric=num, scaler="standard", seed=seed)
    return TabularPreprocessor(numeric=num, categorical=cat, scaler="robust", seed=seed)
