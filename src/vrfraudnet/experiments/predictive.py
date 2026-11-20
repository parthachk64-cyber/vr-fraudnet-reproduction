"""One (dataset, seed) predictive run: preprocessing -> [Stage 0] -> Stage 1 -> routing -> Stage 2-3 -> Stage 4 -> metrics, plus the nine baselines.

Fit order (each step sees only what precedes it):
  1 preprocessing fitted on TRAIN rows            2 Stage-0 encoder trained on TRAIN, embeddings for every partition from time-bounded graph snapshots
  3 baselines + triage trained on TRAIN, early stopping on VAL_SELECT
  4 operating (F1) threshold and routing thresholds from VAL_SELECT
  5 rationale generation + verification on VAL_SELECT -> mixer fitted on VAL_SELECT
  6 conformal quantile from VAL_CALIBRATION (never used above)   7 TEST scored once with everything frozen.

Writes runs/<id>/<dataset>/seed<k>/{<system>.json, predictions.npz}. A `synthetic_fixture` provenance is stamped on SYN runs; table builders for
reproduced results refuse those. Nothing in this module reads references/reported (published numbers).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..baselines.registry import make_baseline
from ..calibration.conformal import SplitConformal
from ..data import splits as SP
from ..data.evidence import EvidenceBuilder
from ..data.providers import load_dataset, preprocessor_for
from ..errors import MissingInputError, OptionalDependencyError
from ..evaluation import metrics as M
from ..models.mixer import VerifiedMixer
from ..models.pipeline import Ablation, Case, VRFraudNetPipeline
from ..models.triage import TriageModel, cost_sample_weights, select_thresholds
from ..published import BASELINES, CONFORMAL, PUBLISHED_ROUTING_THRESHOLDS
from ..training.manifest import build_manifest
from ..types import Route

ALL_SYSTEMS = tuple(BASELINES) + ("triage_only", "vr_fraudnet")


def _node_features(ds, X_rows: np.ndarray, train_end_time: float) -> np.ndarray:
    """Node feature matrix for Stage 0. Node-per-row datasets (D4, D5): the preprocessed row features. Account-node datasets (D2, SYN): log in/out degree
    over TRAIN-period edges only (a train-fitted statistic)."""
    g = ds.graph
    if len(ds.frame) == g.n_nodes:
        return X_rows.astype(np.float32)
    sl = g.edge_slice(train_end_time + 1e-6)
    indeg = np.bincount(g.dst[sl], minlength=g.n_nodes)
    outdeg = np.bincount(g.src[sl], minlength=g.n_nodes)
    return np.log1p(np.stack([indeg, outdeg], 1)).astype(np.float32)


def _stage0(ds, split, Xrows, y, seed, align, max_epochs, device):
    from ..training import encoder_train as E

    f = ds.frame
    t = f["__time"].to_numpy()
    node_per_row = len(f) == ds.graph.n_nodes
    x_nodes = _node_features(ds, Xrows, float(t[split.partitions["train"]].max()))
    if node_per_row:
        mk = lambda pos: E.Samples(pos.astype(np.int64), -np.ones(len(pos), np.int64), t[pos], y[pos])
    else:
        src, dst = ds.meta["edge_src"], ds.meta["edge_dst"]
        mk = lambda pos: E.Samples(src[pos].astype(np.int64), dst[pos].astype(np.int64), t[pos], y[pos])
    model, head, info = E.train_encoder(x_nodes, ds.graph, mk(split.partitions["train"]), mk(split.partitions["val_select"]), align=align, seed=seed,
                                        device=device, max_epochs=max_epochs)
    z = {k: E.encode(model, x_nodes, ds.graph, mk(split.partitions[k]), align, info["t_scale"], device) for k in ("train", "val_select", "val_calibration", "test")}
    return z, {"history_tail": info["history"][-3:], "best_val_auprc": info["best_val_auprc"], "align": align, "epochs_run": info["epochs_run"]}


def run_predictive(dataset_id: str, raw_dir: Optional[Path], seed: int, out_root: Path, *, systems: Sequence[str] = ALL_SYSTEMS, provider=None, index=None,
                   fit_thresholds: Optional[bool] = None, max_escalation: Optional[float] = None, split_kwargs: Optional[dict] = None, use_graph_encoder: bool = True,
                   align: str = "mean", encoder_epochs: Optional[int] = None, device: str = "cpu", baseline_epochs: int = 30, triage_params: Optional[dict] = None,
                   shock_step: int = 43, tag: str = "", ablation: Optional[Ablation] = None, verbose: bool = False) -> dict:
    t0 = time.time()
    ds = load_dataset(dataset_id, raw_dir)
    kind = "synthetic_fixture" if ds.spec.synthetic else "benchmark"
    if max_escalation is None:                          # the paper keeps the LM path to <= 5% of transactions; the tiny fixture needs a looser cap
        max_escalation = 0.35 if ds.spec.synthetic else 0.05
    split = SP.make_split(ds.spec.id, ds.frame, **(split_kwargs or {}))
    f, ids = ds.frame, ds.frame["__id"].to_numpy()
    y_all = f["__label"].to_numpy()
    pp = preprocessor_for(ds, seed)
    tr = f.iloc[split.partitions["train"]]
    X = {"train": pp.fit_transform_train(tr, tr["__label"].to_numpy())}
    for k in ("val_select", "val_calibration", "test"):
        X[k] = pp.transform(f.iloc[split.partitions[k]])
    Y = {k: y_all[split.partitions[k]] for k in X}
    out_dir = Path(out_root) / ds.spec.id / f"seed{seed}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = list(split.notes) + ([f"graph alignment policy {align} is a reconstruction"] if ds.spec.graph else [])
    base_manifest = lambda sys_, cfg: build_manifest(kind=kind, dataset=ds.spec.id, seed=seed, config={"system": sys_, **cfg}, split_fingerprint=split.fingerprint(ids), notes=notes)
    results, preds = {}, {"ids": ids[split.partitions["test"]], "y": Y["test"]}
    if "__protected_age" in f:
        preds["protected_age"] = f["__protected_age"].to_numpy()[split.partitions["test"]]

    def finish(name, s_val, s_test, extra=None, manifest_cfg=None):
        thr = M.select_f1_threshold(Y["val_select"], s_val)               # frozen from VAL_SELECT; never re-selected on test
        res = {"system": name, "dataset": ds.spec.id, "seed": seed, "provenance": base_manifest(name, manifest_cfg or {}), "operating_threshold": thr,
               "metrics": M.all_metrics(Y["test"], s_test, thr), "reference": {"val_select_auprc": M.auprc(Y["val_select"], s_val)}, "test_prevalence": M.prevalence(Y["test"]), "n_test": int(len(Y["test"])), "n_test_positive": int(Y["test"].sum()),
               "test_counts": dict(zip(("tp", "fp", "fn", "tn"), M.confusion(Y["test"], s_test, thr))), **(extra or {})}
        if ds.spec.id == "D1":
            res["windows"] = {f"month_{m}": _window_metrics(f, split, Y, s_test, thr, np.nonzero(f["__group"].to_numpy()[split.partitions["test"]] == m)[0]) for m in (6, 7)}
        if ds.spec.id == "D4":
            w = SP.d4_shock_windows(f, shock_step)
            tp = split.partitions["test"]
            where = {name: np.nonzero(np.isin(tp, pos))[0] for name, pos in w.items()}
            res["windows"] = {k: _window_metrics(f, split, Y, s_test, thr, v) for k, v in where.items()}
            res["shock_step_note"] = f"shock step {shock_step} is a RECONSTRUCTION (not stated by the paper)"
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        preds[name] = s_test
        results[name] = res
        if verbose:
            print(f"[{ds.spec.id} seed{seed}] {name}: AUPRC {res['metrics']['AUPRC']:.4f}")

    for name in [s for s in systems if s in BASELINES]:
        try:
            mdl = make_baseline(name, seed, epochs=baseline_epochs).fit(X["train"], Y["train"], X["val_select"], Y["val_select"])
        except OptionalDependencyError as e:
            results[name] = {"status": f"blocked: {e}"}
            continue
        finish(name, mdl.score(X["val_select"]), mdl.score(X["test"]), manifest_cfg={"baseline": name, "epochs": baseline_epochs})

    if any(s in systems for s in ("triage_only", "vr_fraudnet")):
        Xf, enc_info = X, None
        if ds.spec.graph and use_graph_encoder:
            try:
                Xrows = np.zeros((len(f), X["train"].shape[1]), np.float32)
                for k in X:
                    Xrows[split.partitions[k]] = X[k]
                z, enc_info = _stage0(ds, split, Xrows, y_all, seed, align, encoder_epochs, device)
                Xf = {k: np.hstack([X[k], z[k]]) for k in X}
            except OptionalDependencyError as e:
                results["stage0"] = {"status": f"blocked: {e}"}
        w = cost_sample_weights(Y["train"], 300.0, 5.0, 0.5)               # published cost-sensitive weight 0.50; unit costs are the S1-implied ones for D1 (see docs)
        tm = TriageModel(params=triage_params, seed=seed).fit(Xf["train"], Y["train"], Xf["val_select"], Y["val_select"], sample_weight=w)
        P = {k: tm.predict_proba(Xf[k]) for k in ("val_select", "val_calibration", "test")}
        if "triage_only" in systems:
            finish("triage_only", P["val_select"], P["test"], {"stage0": enc_info, "triage": {"best_iteration": tm.fit_.best_iteration}}, {"stage": "triage_only"})
        if "vr_fraudnet" in systems:
            published = PUBLISHED_ROUTING_THRESHOLDS.get(ds.spec.id)
            fit = fit_thresholds if fit_thresholds is not None else published is None
            if fit:
                th = select_thresholds(P["val_select"], Y["val_select"], max_escalation=max_escalation, c_fn=300.0, c_fp=5.0, c_review=5.0)
                low, high, th_src = th["low"], th["high"], "fitted on VAL_SELECT (reconstructed objective, D-02)"
            else:
                (low, high), th_src = published, "published (Sec. 3.8.1)"
            if provider is None:
                results["vr_fraudnet"] = {"status": "blocked", "missing": "a rationale provider (LlamaLoRAProvider with licensed base weights and a LoRA adapter)",
                                          "thresholds": {"low": low, "high": high, "source": th_src}}
            else:
                builder = EvidenceBuilder(ds)
                mode = "demo" if getattr(provider, "is_fixture", False) else "research"
                if ablation is not None and not ablation.is_full:
                    mode = "demo_ablation" if getattr(provider, "is_fixture", False) else "ablation"
                mk_pipe = lambda **kw: VRFraudNetPipeline(low=low, high=high, provider=provider, index=index, mode=mode, ablation=ablation, **kw)
                cases = lambda part: [Case(str(ids[i]), float(p), builder.snapshot(int(i))) for i, p in zip(split.partitions[part], P[part])]
                sel = mk_pipe().run_many(cases("val_select"))
                mixer = VerifiedMixer().fit([r.triage_probability for r in sel], [r.verifier.value for r in sel],
                                            [np.nan if r.rationale_probability_used is None else r.rationale_probability_used for r in sel], Y["val_select"])
                cal = mk_pipe(mixer=mixer).run_many(cases("val_calibration"))
                conf = {a: SplitConformal(a).fit([r.risk_score for r in cal], Y["val_calibration"]) for a in (CONFORMAL["alpha_primary"], CONFORMAL["alpha_sensitivity"])}
                pipe = mk_pipe(mixer=mixer, conformal=conf[CONFORMAL["alpha_primary"]])
                res_sel = pipe.run_many(cases("val_select"))
                res_test = pipe.run_many(cases("test"))
                s_val = np.array([r.risk_score for r in res_sel])
                s_test = np.array([r.risk_score for r in res_test])
                routes = {r.value: int(sum(x.route is r for x in res_test)) for r in Route}
                extra = {"thresholds": {"low": low, "high": high, "source": th_src}, "route_counts": routes, "mixer_w": mixer.w_, "stage0": enc_info,
                         "conformal": {str(a): c.coverage([r.risk_score for r in res_test], Y["test"]) | {"threshold": c.threshold, "n_cal": c.n_cal} for a, c in conf.items()},
                         "human_review_fraction": float(np.mean([not r.automated for r in res_test]))}
                finish("vr_fraudnet" if ablation is None else f"vr_fraudnet_{ablation.name}", s_val, s_test, extra, {"provider": provider.name, "ablation": ablation.name if ablation else "full"})
    np.savez_compressed(out_dir / "predictions.npz", **{k: v for k, v in preds.items()})
    summary = {"dataset": ds.spec.id, "seed": seed, "kind": kind, "systems": {k: (v.get("metrics") or {"status": v.get("status")}) for k, v in results.items()}, "seconds": round(time.time() - t0, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary


def _window_metrics(f, split, Y, s_test, thr, local_idx) -> dict:
    y = Y["test"][local_idx]
    if len(y) == 0 or y.sum() == 0:
        return {"status": "no_positives_in_window", "n": int(len(y))}
    s = s_test[local_idx]
    return {"AUPRC": M.auprc(y, s), "n": int(len(y)), "n_positive": int(y.sum()), **{k: v for k, v in M.threshold_metrics(y, s, thr).items() if k in ("f1", "mcc")}}
