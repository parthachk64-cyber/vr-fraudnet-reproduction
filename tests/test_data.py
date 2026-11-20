import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import provider_files as PF  # noqa: E402

from vrfraudnet.data import splits as SP
from vrfraudnet.data.evidence import EvidenceBuilder, sanitize_field
from vrfraudnet.data.graph import TemporalGraph, ego_edges, induced_subgraph, normalized_laplacian, sample_neighbors
from vrfraudnet.data.providers import check_dataset, load_dataset, preprocessor_for
from vrfraudnet.data.registry import REGISTRY, get
from vrfraudnet.data.synthetic import make_synthetic
from vrfraudnet.data.transforms import CategoricalEncoder, PerGroupScaler, SmoothedTargetEncoder, TabularPreprocessor, hash64_bucket
from vrfraudnet.errors import LeakageError, MissingInputError, ProtocolError, VRFraudNetError
from vrfraudnet.models.grammar import build_rationale_grammar
from vrfraudnet.verifier import verify


# ------------------------------------------------------------------------------------------ registry and check
def test_registry_states_source_access_and_ambiguities():
    assert set(REGISTRY) == {"D1", "D2", "D3", "D4", "D5", "SYN"}
    for d in ("D1", "D2", "D3", "D4", "D5"):
        s = get(d)
        assert s.page.startswith("https://") and s.access and s.split_rule and s.files
    assert get("d2").graph and not get("D1").graph and get("SYN").synthetic
    with pytest.raises(VRFraudNetError):
        get("D9")
    assert any("oracle" in f.note.lower() for f in get("D2").files)                      # the label-oracle file is called out


@pytest.mark.parametrize("ds,writer", [("D1", PF.write_d1), ("D2", PF.write_d2), ("D3", PF.write_d3), ("D4", PF.write_d4), ("D5", PF.write_d5)])
def test_check_and_load_each_provider(tmp_path, ds, writer):
    writer(tmp_path)
    rep = check_dataset(ds, tmp_path)
    assert all(f["present"] for n, f in rep["files"].items() if "Patterns" not in n)
    assert all("sha256" in f for f in rep["files"].values() if f["present"])
    assert not any("missing columns" in p or "missing arrays" in p for p in rep["problems"])
    loaded = load_dataset(ds, tmp_path)
    for c in ("__id", "__time", "__label"):
        assert c in loaded.frame
    assert loaded.frame["__id"].is_unique and loaded.meta["realised_n_features"] == len(loaded.features)


def test_check_reports_missing_files_and_columns_and_loader_names_the_input(tmp_path):
    PF.write_d1(tmp_path)
    df = pd.read_csv(tmp_path / "Base.csv").drop(columns=["velocity_6h"])
    df.to_csv(tmp_path / "Base.csv", index=False)
    rep = check_dataset("D1", tmp_path, hash_files=False)
    assert not rep["ok"] and any("velocity_6h" in p for p in rep["problems"])
    rep3 = check_dataset("D3", tmp_path)
    assert not rep3["ok"] and "missing file train_transaction.csv" in rep3["problems"]
    with pytest.raises(MissingInputError) as e:
        load_dataset("D3", tmp_path)
    assert "kaggle.com/competitions/ieee-fraud-detection" in str(e.value) and "train_transaction.csv" in str(e.value)
    with pytest.raises(MissingInputError):
        load_dataset("D5", None)


def test_d2_loader_never_opens_the_label_oracle_file(tmp_path):
    PF.write_d2(tmp_path)
    ds = load_dataset("D2", tmp_path)
    assert not any("ORACLE" in str(v) for v in ds.frame.to_numpy().ravel()[:50])
    assert ds.graph is not None and ds.graph.n_nodes == len(ds.meta["node_names"])


def test_d1_excludes_label_and_protected_attribute_from_features(tmp_path):
    PF.write_d1(tmp_path)
    ds = load_dataset("D1", tmp_path)
    assert "fraud_bool" not in ds.features and "customer_age" not in ds.features and "month" not in ds.features
    assert "__protected_age" in ds.frame and len(ds.features) == 29                    # 32 columns - label - month - protected attribute, as documented
    assert "month" in load_dataset("D1", tmp_path, include_month_feature=True).features


def test_d3_drops_documented_columns_and_joins_identity(tmp_path):
    PF.write_d3(tmp_path)
    ds = load_dataset("D3", tmp_path)
    for c in ("D2", "D3", "D9", "D11", "D14", "isFraud", "TransactionID", "TransactionDT"):
        assert c not in ds.features
    assert "D1" in ds.features and "D10" in ds.features and "D15" in ds.features
    assert "id_01" in ds.features and ds.frame["id_01"].isna().any()                   # records without identity keep NaN, not a dropped row
    assert len(ds.meta["v_columns"]) == 339 and len(ds.frame) == 300


def test_d4_keeps_94_local_features_and_marks_unknown(tmp_path):
    PF.write_d4(tmp_path)
    ds = load_dataset("D4", tmp_path)
    assert len(ds.features) == 94 and ds.meta["n_dropped_aggregated"] == 89
    assert set(ds.frame["__label"]) <= {-1, 0, 1} and (ds.frame["__label"] == -1).any()


def test_d5_background_nodes_are_unlabelled_and_arrival_is_derived(tmp_path):
    PF.write_d5(tmp_path)
    ds = load_dataset("D5", tmp_path)
    z = np.load(tmp_path / "dgraphfin.npz")
    assert ((ds.frame["__label"] == -1) == np.isin(z["y"], (2, 3))).all()
    assert "derived" in ds.meta["arrival"] and (ds.frame["__time"] >= 1).all()


# ------------------------------------------------------------------------------------------ splits
def _check_split(m, frame):
    m.assert_disjoint()
    ids = frame["__id"].to_numpy()
    fp = m.fingerprint(ids)
    assert fp == m.fingerprint(ids) and len(fp) == 64
    prev = m.prevalence(frame["__label"])
    assert prev["train"]["n"] > 0 and prev["test"]["n"] > 0
    return prev


def test_d1_split_follows_the_months_and_validation_is_chronological(tmp_path):
    PF.write_d1(tmp_path)
    ds = load_dataset("D1", tmp_path)
    m = SP.make_split("D1", ds.frame)
    g, t = ds.frame["__group"].to_numpy(), ds.frame["__time"].to_numpy()
    assert g[m.partitions["test"]].min() == 6 and g[m.partitions["test"]].max() == 7
    dev = np.concatenate([m.partitions[k] for k in ("train", "val_select", "val_calibration")])
    assert g[dev].max() == 5 and len(dev) == 360
    assert len(m.partitions["val_select"]) + len(m.partitions["val_calibration"]) == round(0.1 * 360)
    assert t[m.partitions["train"]].max() < t[m.partitions["val_select"]].min() <= t[m.partitions["val_select"]].max() < t[m.partitions["val_calibration"]].min()
    _check_split(m, ds.frame)


def test_d2_d3_d4_d5_splits(tmp_path):
    for ds_id, w in (("D2", PF.write_d2), ("D3", PF.write_d3), ("D4", PF.write_d4), ("D5", PF.write_d5)):
        root = tmp_path / ds_id
        root.mkdir()
        w(root)
        ds = load_dataset(ds_id, root)
        m = SP.make_split(ds_id, ds.frame)
        t = ds.frame["__time"].to_numpy()
        if ds_id == "D2":
            n = len(ds.frame)
            assert (len(m.partitions["train"]), len(m.partitions["test"])) == (round(0.6 * n), n - round(0.8 * n))
        if ds_id == "D3":
            cut = np.quantile(t, 0.8)
            assert t[m.partitions["test"]].min() >= cut and t[np.concatenate([m.partitions[k] for k in ("train", "val_select", "val_calibration")])].max() < cut
        if ds_id == "D4":
            g, y = ds.frame["__group"].to_numpy(), ds.frame["__label"].to_numpy()
            used = np.concatenate([m.partitions[k] for k in ("train", "val_select", "val_calibration", "test")])
            assert (y[used] >= 0).all()                                                  # unknown-label nodes never supervise or score
            assert g[m.partitions["test"]].min() >= 35 and g[m.partitions["train"]].max() <= 34
            assert (y[m.partitions["context"]] == -1).all()
        if ds_id == "D5":
            y = ds.frame["__label"].to_numpy()
            assert (y[m.partitions["context"]] == -1).all() and (y[m.partitions["test"]] >= 0).all()
        _check_split(m, ds.frame)


def test_validation_halving_is_two_chronological_halves():
    f = pd.DataFrame({"__id": [str(i) for i in range(11)], "__time": np.arange(11)[::-1].astype(float), "__label": 0})
    sel, cal = SP.halve_validation(np.arange(11), f)
    assert len(sel) == 5 and len(cal) == 6
    assert f["__time"].to_numpy()[sel].max() < f["__time"].to_numpy()[cal].min()


def test_overlap_and_order_violations_are_detected():
    m = SP.SplitManifest("X", {"train": np.array([0, 1, 2]), "test": np.array([2, 3])})
    with pytest.raises(LeakageError):
        m.assert_disjoint()
    m2 = SP.SplitManifest("X", {"train": np.array([0, 1]), "test": np.array([2, 3])})
    with pytest.raises(LeakageError):
        m2.assert_chronological(np.array([5.0, 6.0, 1.0, 2.0]))
    with pytest.raises(ProtocolError):
        SP.make_split("D2", pd.DataFrame({"a": [1]}))


def test_d3_expanding_folds_and_d4_shock_windows(tmp_path):
    PF.write_d3(tmp_path)
    f = load_dataset("D3", tmp_path).frame
    for q in (0.6, 0.7, 0.8, 0.85, 0.9):
        tr, te = SP.d3_expanding_fold(f, q)
        assert len(tr) + len(te) == len(f) and f["__time"].to_numpy()[tr].max() < f["__time"].to_numpy()[te].min()
    t4 = tmp_path / "d4"
    t4.mkdir()
    PF.write_d4(t4)
    f4 = load_dataset("D4", t4).frame
    w = SP.d4_shock_windows(f4, shock_step=43)
    g = f4["__group"].to_numpy()
    assert g[w["pre_shock"]].max() == 42 and g[w["post_shock"]].min() == 43 and not set(w["pre_shock"]) & set(w["post_shock"])


# ------------------------------------------------------------------------------------------ transformers: train-only
def test_preprocessor_statistics_come_from_training_rows_only():
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"a": rng.normal(10, 1, 400), "b": rng.normal(0, 1, 400), "c": rng.choice(["x", "y"], 400)})
    y = rng.integers(0, 2, 400)
    tr = slice(0, 200)
    pp = TabularPreprocessor(numeric=["a", "b"], target_encoded=["c"]).fit(f.iloc[tr], y[tr])
    med = pp.medians_.copy()
    f2 = f.copy()
    f2.iloc[200:, 0] = 1e6                                            # wreck the evaluation rows
    out_a = pp.transform(f.iloc[tr])
    out_b = TabularPreprocessor(numeric=["a", "b"], target_encoded=["c"]).fit(f2.iloc[tr], y[tr]).transform(f2.iloc[tr])
    assert np.array_equal(out_a, out_b) and np.array_equal(med, pp.medians_)
    assert np.isfinite(pp.transform(f2.iloc[200:])).all()


def test_target_encoding_out_of_fold_does_not_leak_a_rows_own_label():
    x = pd.Series([f"u{i}" for i in range(100)])                      # every row has a unique category
    y = np.random.default_rng(0).integers(0, 2, 100)
    enc = SmoothedTargetEncoder()
    oof = enc.fit_transform_train(x, y)
    full = enc.transform(x)
    assert np.corrcoef(full, y)[0, 1] > 0.5                          # the in-sample encoding leaks the label...
    assert abs(np.corrcoef(oof, y)[0, 1]) < 0.35                     # ...the out-of-fold encoding does not
    assert enc.transform(pd.Series(["never_seen"]))[0] == pytest.approx(enc.prior_)


def test_categorical_encoder_onehot_hash_and_unseen():
    low = CategoricalEncoder().fit(pd.Series(["a", "b", "a"]))
    out = low.transform(pd.Series(["a", "b", "zzz", None]))
    assert out.shape == (4, 3) and out[2, 2] == 1.0 and out.sum() == 4
    high = CategoricalEncoder(max_onehot=2, n_buckets=8).fit(pd.Series([str(i) for i in range(50)]))
    assert high.width == 8 and high.transform(pd.Series(["1", "1"])).sum() == 2
    assert hash64_bucket("abc", 16) == hash64_bucket("abc", 16)


def test_per_group_scaler_uses_pooled_training_stats_for_unseen_groups():
    x = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    sc = PerGroupScaler().fit(x, g)
    assert sc.transform(np.array([2.0]), np.array(["a"]))[0] == pytest.approx(0.0)
    pooled = (x.mean(), x.std() or 1.0)
    assert sc.transform(np.array([5.0]), np.array(["NEVER_SEEN"]))[0] == pytest.approx((5.0 - pooled[0]) / pooled[1])


def test_dataset_recipes_fit_on_train_and_produce_finite_features(tmp_path):
    for ds_id, w in (("D1", PF.write_d1), ("D2", PF.write_d2), ("D3", PF.write_d3), ("D4", PF.write_d4), ("D5", PF.write_d5)):
        root = tmp_path / ds_id
        root.mkdir()
        w(root)
        ds = load_dataset(ds_id, root)
        m = SP.make_split(ds_id, ds.frame)
        pp = preprocessor_for(ds)
        tr = ds.frame.iloc[m.partitions["train"]]
        xtr = pp.fit_transform_train(tr, tr["__label"].to_numpy())
        xte = pp.transform(ds.frame.iloc[m.partitions["test"]])
        assert xtr.shape[1] == xte.shape[1] == pp.n_features_ and np.isfinite(xtr).all() and np.isfinite(xte).all(), ds_id
        if ds_id == "D3":
            assert pp.pca_.n_components_ <= 50


# ------------------------------------------------------------------------------------------ graph
def test_snapshot_excludes_future_and_equal_time_edges_and_stays_sparse():
    g = TemporalGraph([0, 1, 2], [1, 2, 3], [10.0, 20.0, 30.0], 4)
    a = g.snapshot(20.0)
    assert a.nnz == 2 and a[0, 1] == 1 and a[1, 2] == 0                # the edge AT t=20 is not before t=20
    assert g.snapshot(31.0).nnz == 6 and g.snapshot(31.0, window=5.0).nnz == 2   # trailing window
    big = TemporalGraph(np.arange(10000), np.arange(1, 10001), np.arange(10000.0), 10001)
    snap = big.snapshot(5000.0)                                       # 10,001 x 10,001 would be 800 MB dense
    assert snap.nnz == 2 * 5000 and hasattr(snap, "indptr")
    with pytest.raises(LeakageError):
        g.assert_no_future(20.0, np.array([20.0]))
    with pytest.raises(ValueError):
        TemporalGraph([0], [5], [1.0], 3)


def test_normalized_laplacian_is_sparse_with_spectrum_in_0_2():
    g = TemporalGraph([0, 1, 2, 3], [1, 2, 0, 3], [1.0] * 4, 5)       # a triangle, a self-loop (dropped) and an isolated node
    lap = normalized_laplacian(g.snapshot(2.0))
    dense = lap.toarray()
    ev = np.linalg.eigvalsh(dense)
    assert ev.min() >= -1e-6 and ev.max() <= 2 + 1e-6
    assert dense[4, 4] == 1.0 and np.allclose(dense, dense.T)


def test_neighbor_sampling_and_ego_graph_completeness():
    g = TemporalGraph([0] * 30 + [1], list(range(1, 31)) + [99 % 40], [1.0] * 31, 40)
    a = g.snapshot(2.0)
    nb = sample_neighbors(a, np.array([0]), 10, np.random.default_rng(0))
    assert len(nb) == 10 and set(nb) <= set(range(1, 31))
    assert induced_subgraph(a, np.array([0, 1, 2])).shape == (3, 3)
    edges, complete = ego_edges(g, 0, 2.0, 2)
    assert complete and len(edges) >= 30
    edges, complete = ego_edges(g, 0, 2.0, 2, node_cap=5)
    assert not complete                                               # cap hit: absence may NOT be claimed from this neighbourhood
    assert ego_edges(g, 0, 0.5, 2)[0] == []                           # nothing is visible before the first edge exists


# ------------------------------------------------------------------------------------------ evidence on the synthetic dataset
def test_synthetic_dataset_is_labelled_synthetic_and_reproducible():
    a, b = make_synthetic(500, seed=3), make_synthetic(500, seed=3)
    assert a.frame.equals(b.frame) and a.spec.synthetic and a.frame["__id"].str.startswith("TX_SYN_").all()
    assert a.frame["__time"].is_monotonic_increasing and 0.02 < a.frame["__label"].mean() < 0.4
    assert (a.frame["__origin"].str.startswith("ACC_SYN_")).all()


def test_evidence_is_strictly_past_only_and_matches_brute_force():
    ds = make_synthetic(800, seed=1)
    eb = EvidenceBuilder(ds, window_s=86400.0, hops=3)
    f = ds.frame
    for pos in (5, 200, 799):
        ev = eb.snapshot(pos)
        t, o = f["__time"].iloc[pos], f["__origin"].iloc[pos]
        prior = f[(f["__origin"] == o) & (f["__time"] < t) & (f["__time"] >= t - 86400.0)]
        assert ev.fields["recent_transfer_count"] == len(prior)
        assert all(v < t for v in ev.event_times.values())
        if ev.graph is not None:
            assert all(e[2] < t for e in ev.graph.edges)
    assert sanitize_field("TransactionAmt") == "transactionamt" and sanitize_field("1x") == "f_1x"


def test_a_rationale_built_from_the_snapshot_verifies_and_a_false_one_does_not():
    ds = make_synthetic(800, seed=1)
    ev = EvidenceBuilder(ds).snapshot(600)
    amt = ev.fields["amount"]
    ok = {"verdict": "uncertain", "probability": 0.5, "evidence_refs": ["E_AMOUNT"],
          "claims": [{"type": "numeric_comparison", "field": "amount", "op": ">=", "value": amt - 1, "evidence_id": "E_AMOUNT"}]}
    assert verify(ok, ev).accepted
    ok["claims"][0]["value"] = amt + 1
    assert not verify(ok, ev).accepted
    assert build_rationale_grammar(ev).viable('{"verdict":"fraud","probability":0.9,"claims":[{"type":"numeric_comparison","field":"amount"')
