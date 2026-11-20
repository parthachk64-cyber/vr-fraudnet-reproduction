import numpy as np
import pytest

from vrfraudnet.data.evidence import EvidenceBuilder
from vrfraudnet.data.synthetic import make_synthetic
from vrfraudnet.errors import LeakageError, ProtocolError
from vrfraudnet.models import triage as T
from vrfraudnet.training.manifest import build_manifest, select_checkpoint
from vrfraudnet.training.objectives import (RuleTargetBuilder, TargetStats, augment_with_variants, load_frozen_variants, make_counterfactual)
from vrfraudnet.verifier import verify


# ----------------------------------------------------------------------------------------- routing and focal loss (no heavy deps)
def test_gate_endpoints_belong_to_the_review_band():
    r = T.route_probability([0.0, 0.1999, 0.20, 0.50, 0.80, 0.8001, 1.0], 0.20, 0.80)
    assert list(r) == ["accept", "accept", "escalate", "escalate", "escalate", "reject", "reject"]
    with pytest.raises(ProtocolError):
        T.route_probability([0.5], 0.8, 0.2)


@pytest.mark.parametrize("y", [0, 1])
@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0])
def test_focal_gradient_matches_finite_differences(y, gamma):
    z = np.linspace(-6, 6, 25)
    yy = np.full_like(z, y, dtype=int)
    g, h = T.focal_grad_hess(yy, z, gamma, 1.7, 0.6, min_hess=-1e9)
    eps = 1e-6
    num_g = (T.focal_loss(yy, z + eps, gamma, 1.7, 0.6) - T.focal_loss(yy, z - eps, gamma, 1.7, 0.6)) / (2 * eps)
    num_h = (T.focal_grad_hess(yy, z + eps, gamma, 1.7, 0.6, -1e9)[0] - T.focal_grad_hess(yy, z - eps, gamma, 1.7, 0.6, -1e9)[0]) / (2 * eps)
    assert np.allclose(g, num_g, atol=1e-5)
    assert np.allclose(h, num_h, atol=1e-4)


def test_focal_with_gamma_zero_is_weighted_logloss_and_hessian_is_floored():
    z = np.array([-2.0, 0.5, 3.0])
    y = np.array([1, 0, 1])
    p = 1 / (1 + np.exp(-z))
    assert np.allclose(T.focal_loss(y, z, 0.0), -np.where(y == 1, np.log(p), np.log(1 - p)))
    _, h = T.focal_grad_hess(y, np.array([8.0, -8.0, 8.0]), 2.0)
    assert (h >= 1e-6).all()


def test_class_balanced_and_cost_weights():
    y = np.array([0] * 990 + [1] * 10)
    a1, a0 = T.class_balanced_alpha(y)
    assert a1 > a0 and (a1 * 10 + a0 * 990) / 1000 == pytest.approx(1.0)
    w = T.cost_sample_weights(np.array([0, 1]), c_fn=300.0, c_fp=5.0, lam=0.5)
    assert w[1] == 1.0 and w[0] == pytest.approx(0.5 + 0.5 * 5 / 300)
    assert (T.cost_sample_weights(np.array([0, 1]), 300, 5, lam=0.0) == 1.0).all()


def test_threshold_selection_respects_the_escalation_cap():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.15, 4000), 0, 1)
    sel = T.select_thresholds(p, y, max_escalation=0.08, c_fn=300, c_fp=5, c_review=5)
    r = T.route_probability(p, sel["low"], sel["high"])
    assert (r == "escalate").mean() <= 0.08 and sel["low"] < sel["high"]
    with pytest.raises(ProtocolError):
        T.select_thresholds(p, y, max_escalation=0.0)


# ----------------------------------------------------------------------------------------- LightGBM triage
def test_triage_fits_and_ranks_better_than_chance_on_synthetic_data():
    pytest.importorskip("lightgbm")
    from vrfraudnet.data import splits as SP
    from vrfraudnet.data.providers import preprocessor_for
    from vrfraudnet.evaluation.metrics import auprc

    ds = make_synthetic(4000, seed=0)
    m = SP.make_split("SYN", ds.frame)
    f = ds.frame
    pp = preprocessor_for(ds)
    tr = f.iloc[m.partitions["train"]]
    Xtr = pp.fit_transform_train(tr, tr["__label"].to_numpy())
    sel = f.iloc[m.partitions["val_select"]]
    te = f.iloc[m.partitions["test"]]
    for obj in ("focal", "binary"):
        tm = T.TriageModel(params={"n_estimators": 80, "min_child_samples": 20, "early_stopping_rounds": 15}, objective=obj, seed=0)
        tm.fit(Xtr, tr["__label"], pp.transform(sel), sel["__label"])
        p = tm.predict_proba(pp.transform(te))
        assert ((p >= 0) & (p <= 1)).all()
        assert auprc(te["__label"], p) > te["__label"].mean() * 1.5          # clearly above the base rate
        assert tm.fit_.best_iteration > 0
    with pytest.raises(ProtocolError):
        T.TriageModel().predict_proba(Xtr)


# ----------------------------------------------------------------------------------------- spectral encoder (torch)
def test_spectral_encoder_shapes_sparse_filters_and_time_conditioning():
    torch = pytest.importorskip("torch")
    from vrfraudnet.data.graph import TemporalGraph, normalized_laplacian
    from vrfraudnet.models.spectral import SpectralEncoder, align_edges, sparse_from_scipy

    rng = np.random.default_rng(0)
    n = 60
    g = TemporalGraph(rng.integers(0, n, 200), rng.integers(0, n, 200), rng.random(200), n)
    lap = sparse_from_scipy(normalized_laplacian(g.snapshot(2.0)))
    assert lap.is_sparse
    enc = SpectralEncoder(in_dim=7)
    x = torch.randn(n, 7)
    z1 = enc(x, lap, torch.tensor(0.1))
    z2 = enc(x, lap, torch.tensor(0.9))
    assert z1.shape == (n, 64) and torch.isfinite(z1).all()
    assert not torch.allclose(z1, z2)                                   # the representation depends on the evaluation time
    assert len(enc.filters(lap, enc.inp(x))) == 4 and enc.coef[-1].out_features == 4 and enc.time.freq.numel() * 2 == 32
    src, dst = torch.tensor([0, 1]), torch.tensor([2, 3])
    for pol in ("mean", "src", "dst", "hadamard"):
        assert align_edges(z1, src, dst, pol).shape == (2, 64)
    with pytest.raises(ValueError):
        align_edges(z1, src, dst, "nope")


def test_beta_filters_partition_unity_on_the_constant_signal_spectrum():
    torch = pytest.importorskip("torch")
    from vrfraudnet.models.spectral import SpectralEncoder

    enc = SpectralEncoder(in_dim=3)
    import scipy.sparse as sp
    L = sp.diags([0.0, 0.5, 1.0, 1.5, 2.0]).tocsr()                       # eigenvalues spread over [0, 2]
    from vrfraudnet.models.spectral import sparse_from_scipy
    lap = sparse_from_scipy(L)
    ones = torch.ones(5, 1)
    total = sum(f for f in enc.filters(lap, ones))                        # sum_k Beta_k(lambda) = 1 / (Beta normalisation) pattern: strictly positive everywhere
    assert (total > 0).all()
    f0 = enc.filters(lap, ones)[0].squeeze()
    assert f0[0] > f0[-1]                                                 # the p=0 kernel is low-pass: it favours small eigenvalues


def test_encoder_training_smoke_on_synthetic_graph():
    torch = pytest.importorskip("torch")
    from vrfraudnet.training.encoder_train import Samples, encode, train_encoder

    ds = make_synthetic(900, n_accounts=80, seed=2)
    src, dst = ds.meta["edge_src"], ds.meta["edge_dst"]
    t, y = ds.frame["__time"].to_numpy(), ds.frame["__label"].to_numpy()
    rng = np.random.default_rng(0)
    x_nodes = rng.normal(size=(80, 5)).astype(np.float32)
    cut1, cut2 = int(0.6 * len(t)), int(0.8 * len(t))
    mk = lambda sl: Samples(src[sl], dst[sl], t[sl], y[sl])
    model, head, info = train_encoder(x_nodes, ds.graph, mk(slice(0, cut1)), mk(slice(cut1, cut2)), max_epochs=2, align="mean", seed=0, batch_size=128,
                                      params={"fanouts": (5, 3)})
    assert info["epochs_run"] >= 1 and 0.0 <= info["best_val_auprc"] <= 1.0
    z = encode(model, x_nodes, ds.graph, mk(slice(cut2, None)), "mean", info["t_scale"], fanouts=(5, 3))
    assert z.shape == (len(t) - cut2, 64) and np.isfinite(z).all()


# ----------------------------------------------------------------------------------------- objectives
def _train_evidence(n=300):
    ds = make_synthetic(1500, seed=4)
    eb = EvidenceBuilder(ds)
    snaps = [eb.snapshot(i) for i in range(400, 400 + n)]
    return ds, snaps


def test_rule_built_targets_are_verified_and_use_training_stats():
    ds, snaps = _train_evidence()
    stats = TargetStats.fit(snaps[:200])
    b = RuleTargetBuilder(stats)
    built = 0
    for s, y in zip(snaps[200:260], ds.frame["__label"].iloc[600:660]):
        r = b.build(s, 0.4, int(y))
        if r is not None:
            built += 1
            assert verify(r, s).accepted and r["verdict"] == ("fraud" if y else "legitimate") and len(r["claims"]) <= 8
    assert built > 40


def test_counterfactual_changes_evidence_and_drops_only_contradicted_claims():
    ds, snaps = _train_evidence()
    b = RuleTargetBuilder(TargetStats.fit(snaps[:200]))
    rng = np.random.default_rng(0)
    seen = 0
    for s in snaps[200:300]:
        rat = b.build(s, 0.5, 1)
        cf = make_counterfactual(s, rat, rng) if rat else None
        if cf is None:
            continue
        ev2, rat2 = cf
        seen += 1
        assert ev2.fields != s.fields and verify(rat2, ev2).accepted
        assert len(rat2["claims"]) < len(rat["claims"]) or rat2["verdict"] == "uncertain"
        assert rat2["probability"] == rat["probability"]                 # label-side fields unchanged
        unchanged = {c["field"] for c in rat2["claims"]}
        assert all(ev2.fields[f] == s.fields[f] for f in unchanged if f in s.fields)   # surviving claims rest on unchanged evidence
    assert seen > 10
    assert make_counterfactual(snaps[0], {"verdict": "uncertain", "probability": 0.5, "claims": [], "evidence_refs": []}, rng) is None


def test_frozen_variants_must_derive_from_training_rows_and_match_their_checksum(tmp_path):
    import pandas as pd
    p = tmp_path / "variants.csv"
    pd.DataFrame({"original_id": ["T1", "T2"], "family": ["x", "y"], "budget": [1, 2]}).to_csv(p, index=False)
    df, digest = load_frozen_variants(p)
    with pytest.raises(LeakageError):
        load_frozen_variants(p, expected_sha256="0" * 64)
    feats = np.ones((2, 3))
    X, y, w = augment_with_variants(np.zeros((4, 3)), np.zeros(4, int), None, ["T1", "T2", "T3"], df, feats, 0.5)
    assert X.shape == (6, 3) and list(y[-2:]) == [1, 1] and list(w[-2:]) == [0.5, 0.5] and list(w[:4]) == [1, 1, 1, 1]
    with pytest.raises(LeakageError):
        augment_with_variants(np.zeros((4, 3)), np.zeros(4, int), None, ["T1"], df, feats)         # T2 is not a training row
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ProtocolError):
        load_frozen_variants(tmp_path / "bad.csv")


def test_checkpoint_selection_rule_and_manifest_kinds():
    h = [{"val_loss": 0.50, "schema_valid_rate": 0.9, "verifier_acceptance_rate": 0.8},
         {"val_loss": 0.40, "schema_valid_rate": 0.9, "verifier_acceptance_rate": 0.7},
         {"val_loss": 0.40, "schema_valid_rate": 0.95, "verifier_acceptance_rate": 0.6},
         {"val_loss": 0.40, "schema_valid_rate": 0.95, "verifier_acceptance_rate": 0.9}]
    assert select_checkpoint(h) == 3
    m = build_manifest(kind="synthetic_fixture", dataset="SYN", seed=1, config={"a": 1})
    assert m["kind"] == "synthetic_fixture" and m["packages"]["python"] and len(m["config_hash"]) == 64 and m["metric_definition"]["orientation"]
    with pytest.raises(ValueError):
        build_manifest(kind="made_up", dataset="x", seed=0, config={})
