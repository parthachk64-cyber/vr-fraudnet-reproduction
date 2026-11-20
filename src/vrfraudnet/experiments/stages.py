"""The paper-scale workflow as separate, resumable stages over a workspace directory (`--work DIR`).

    prepare -> train-encoder -> train-triage -> build-index -> train-rationale -> fit-mixer -> calibrate -> predict -> evaluate

Each stage reads the artifacts of the earlier ones and writes its own, so a long run can be stopped and continued, and every artifact can be
checked. A stage that lacks an input fails early with the exact missing item. On a benchmark dataset no stage substitutes a fixture; the
`--fixture` switches exist only for dataset SYN and stamp the outputs `synthetic_fixture`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from ..calibration.conformal import SplitConformal
from ..data import splits as SP
from ..data.evidence import EvidenceBuilder
from ..data.providers import load_dataset, preprocessor_for
from ..errors import FixtureInResearchModeError, MissingInputError, VRFraudNetError
from ..evaluation import metrics as M
from ..models.mixer import VerifiedMixer
from ..models.pipeline import Case, VRFraudNetPipeline
from ..models.rationale import LlamaLoRAProvider, RuleFixtureProvider, build_prompt
from ..models.grammar import canonicalise
from ..models.triage import TriageModel, cost_sample_weights, select_thresholds
from ..published import CONFORMAL, PUBLISHED_ROUTING_THRESHOLDS
from ..retrieval.index import Exemplar, ExemplarIndex, HashingEmbedder, SentenceTransformerEmbedder, require_research_embedder, summarise
from ..training.manifest import build_manifest
from ..training.objectives import RuleTargetBuilder, TargetStats, make_counterfactual

PARTS = ("train", "val_select", "val_calibration", "test")


class Workspace:
    def __init__(self, work: Path, dataset: str, raw_dir: Optional[Path], seed: int = 0):
        self.dir, self.dataset, self.raw_dir, self.seed = Path(work), dataset.upper(), raw_dir, seed
        self.dir.mkdir(parents=True, exist_ok=True)
        self.synthetic = self.dataset == "SYN"

    def p(self, name: str) -> Path:
        return self.dir / name

    def need(self, name: str, stage: str) -> Path:
        path = self.p(name)
        if not path.exists():
            raise MissingInputError(f"workspace artifact {name}", needed_for=f"this stage (run `{stage}` first)", how_to_fix=f"vrfraudnet {stage} --dataset {self.dataset} --work {self.dir}")
        return path

    def kind(self) -> str:
        return "synthetic_fixture" if self.synthetic else "benchmark"

    def dataset_obj(self):
        return load_dataset(self.dataset, self.raw_dir)

    def manifest(self, stage: str, cfg: dict, split=None, ids=None) -> dict:
        m = build_manifest(kind=self.kind(), dataset=self.dataset, seed=self.seed, config={"stage": stage, **cfg},
                           split_fingerprint=split.fingerprint(ids) if split is not None else None)
        return m


def prepare(ws: Workspace, split_kwargs: Optional[dict] = None) -> dict:
    ds = ws.dataset_obj()
    split = SP.make_split(ds.spec.id, ds.frame, **(split_kwargs or {}))
    f = ds.frame
    pp = preprocessor_for(ds, ws.seed)
    tr = f.iloc[split.partitions["train"]]
    X = {"train": pp.fit_transform_train(tr, tr["__label"].to_numpy())}
    for k in PARTS[1:]:
        X[k] = pp.transform(f.iloc[split.partitions[k]])
    y = {k: f["__label"].to_numpy()[split.partitions[k]] for k in PARTS}
    np.savez_compressed(ws.p("prepared.npz"), **{f"X_{k}": v for k, v in X.items()}, **{f"y_{k}": v for k, v in y.items()}, **{f"pos_{k}": split.partitions[k] for k in split.partitions})
    joblib.dump({"preprocessor": pp, "split_params": split.params, "notes": split.notes}, ws.p("preprocessor.joblib"))
    prev = split.prevalence(f["__label"].to_numpy())
    info = {"dataset": ds.spec.id, "fingerprint": split.fingerprint(f["__id"].to_numpy()), "prevalence": prev, "realised_n_features": pp.n_features_,
            "declared_features": ds.meta.get("realised_n_features"), "stated_prevalence_pct": ds.spec.stated_prevalence_pct, "notes": split.notes,
            "manifest": ws.manifest("prepare", split.params, split, f["__id"].to_numpy())}
    ws.p("prepare.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    return info


def _load_prepared(ws: Workspace):
    z = np.load(ws.need("prepared.npz", "prepare"), allow_pickle=False)
    X = {k: z[f"X_{k}"] for k in PARTS}
    y = {k: z[f"y_{k}"] for k in PARTS}
    pos = {k[4:]: z[k] for k in z.files if k.startswith("pos_")}
    return X, y, pos


def train_encoder(ws: Workspace, *, epochs: Optional[int] = None, align: str = "mean", device: str = "cpu") -> dict:
    from .predictive import _stage0

    ds = ws.dataset_obj()
    if not ds.spec.graph:
        raise VRFraudNetError(f"{ds.spec.id} has no graph: Stage 0 is bypassed (past-only tabular features are used instead; see docs/ARCHITECTURE.md)")
    X, y, pos = _load_prepared(ws)
    f = ds.frame
    split = SP.SplitManifest(ds.spec.id, {k: v for k, v in pos.items()})
    Xrows = np.zeros((len(f), X["train"].shape[1]), np.float32)
    for k in PARTS:
        Xrows[pos[k]] = X[k]
    z, info = _stage0(ds, split, Xrows, f["__label"].to_numpy(), ws.seed, align, epochs, device)
    np.savez_compressed(ws.p("stage0.npz"), **{f"z_{k}": v for k, v in z.items()})
    info["manifest"] = ws.manifest("train-encoder", {"align": align, "epochs": epochs})
    ws.p("encoder.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    return info


def _features(ws: Workspace):
    X, y, pos = _load_prepared(ws)
    if ws.p("stage0.npz").exists():
        z = np.load(ws.p("stage0.npz"))
        X = {k: np.hstack([X[k], z[f"z_{k}"]]) for k in PARTS}
    return X, y, pos


def train_triage(ws: Workspace, *, params: Optional[dict] = None, fit_thresholds: Optional[bool] = None, max_escalation: Optional[float] = None) -> dict:
    if max_escalation is None:
        max_escalation = 0.35 if ws.synthetic else 0.05
    X, y, pos = _features(ws)
    w = cost_sample_weights(y["train"], 300.0, 5.0, 0.5)
    tm = TriageModel(params=params, seed=ws.seed).fit(X["train"], y["train"], X["val_select"], y["val_select"], sample_weight=w)
    P = {k: tm.predict_proba(X[k]) for k in PARTS}
    pub = PUBLISHED_ROUTING_THRESHOLDS.get(ws.dataset)
    fit = fit_thresholds if fit_thresholds is not None else pub is None
    if fit:
        th = select_thresholds(P["val_select"], y["val_select"], max_escalation=max_escalation, c_fn=300.0, c_fp=5.0, c_review=5.0)
        th = {**th, "source": "fitted on VAL_SELECT (reconstructed objective, docs/DECISIONS.md D-02)"}
    else:
        th = {"low": pub[0], "high": pub[1], "source": "published (Sec. 3.8.1)"}
    joblib.dump(tm, ws.p("triage.joblib"))
    np.savez_compressed(ws.p("triage_proba.npz"), **{f"p_{k}": v for k, v in P.items()})
    info = {"thresholds": th, "best_iteration": tm.fit_.best_iteration, "val_auprc": tm.fit_.val_auprc, "n_features": X["train"].shape[1],
            "manifest": ws.manifest("train-triage", {"params": tm.params, "stage0": ws.p("stage0.npz").exists()})}
    ws.p("triage.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    return info


def _embedder(ws: Workspace, fixture: bool):
    if fixture:
        if not ws.synthetic:
            raise FixtureInResearchModeError("--fixture is only allowed for dataset SYN")
        return HashingEmbedder()
    e = SentenceTransformerEmbedder()
    require_research_embedder(e)
    return e


def build_index(ws: Workspace, *, fixture: bool = False, max_exemplars: int = 20000, backend: str = "flat") -> dict:
    """Index VERIFIED review-band TRAINING exemplars. Rationale targets are rule-built (D-15) because the paper does not say how its were authored."""
    ds = ws.dataset_obj()
    X, y, pos = _features(ws)
    P = np.load(ws.need("triage_proba.npz", "train-triage"))
    th = json.loads(ws.need("triage.json", "train-triage").read_text())["thresholds"]
    f, builder = ds.frame, EvidenceBuilder(ds)
    tp = pos["train"]
    # review-band rows from the TRAIN partition by the train-partition triage score (in-sample: documented limitation)
    tm = joblib.load(ws.need("triage.joblib", "train-triage"))
    ptr = tm.predict_proba(X["train"])
    band = np.nonzero((ptr >= th["low"]) & (ptr <= th["high"]))[0][:max_exemplars]
    snaps = [builder.snapshot(int(tp[i])) for i in band]
    stats = TargetStats.fit(snaps)
    tb = RuleTargetBuilder(stats)
    ex, dropped = [], 0
    for i, s in zip(band, snaps):
        rat = tb.build(s, float(ptr[i]), int(y["train"][i]))
        if rat is None:
            dropped += 1
            continue
        ex.append(Exemplar(s.transaction_id, s.prediction_time, frozenset(s.entities.values()), summarise(s.fields, "review", float(ptr[i])), rat, "review"))
    idx = ExemplarIndex(_embedder(ws, fixture), backend).build(ex)
    man = idx.save(ws.p("index"))
    info = {"n_indexed": len(ex), "n_candidates": len(band), "dropped_unverifiable": dropped, "index": man, "manifest": ws.manifest("build-index", {"fixture": fixture})}
    ws.p("index.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    return info


def corpus(ws: Workspace, max_rows: int = 5000) -> tuple[list[dict], list[dict]]:
    """(train corpus, validation corpus) of {prompt, target[, cf_prompt, cf_target]} from verified review-band TRAIN / VAL_SELECT rows."""
    ds = ws.dataset_obj()
    X, y, pos = _features(ws)
    P = np.load(ws.need("triage_proba.npz", "train-triage"))
    th = json.loads(ws.need("triage.json", "train-triage").read_text())["thresholds"]
    tm = joblib.load(ws.p("triage.joblib"))
    builder, rng = EvidenceBuilder(ds), np.random.default_rng(ws.seed)
    out = {}
    stats = None
    for part in ("train", "val_select"):
        p = tm.predict_proba(X[part]) if part == "train" else P["p_val_select"]
        rows = np.nonzero((p >= th["low"]) & (p <= th["high"]))[0][:max_rows]
        snaps = [builder.snapshot(int(pos[part][i])) for i in rows]
        stats = stats or TargetStats.fit(snaps)
        tb, lst = RuleTargetBuilder(stats), []
        for i, s in zip(rows, snaps):
            rat = tb.build(s, float(p[i]), int(y[part][i]))
            if rat is None:
                continue
            r = {"prompt": build_prompt(s, [], float(p[i])), "target": canonicalise(rat)}
            cf = make_counterfactual(s, rat, rng) if part == "train" else None
            if cf:
                r.update(cf_prompt=build_prompt(cf[0], [], float(p[i])), cf_target=canonicalise(cf[1]))
            lst.append(r)
        out[part] = lst
    return out["train"], out["val_select"]


def _provider(ws: Workspace, base_model, adapter, fixture: bool):
    if fixture:
        if not ws.synthetic:
            raise FixtureInResearchModeError("--fixture is only allowed for dataset SYN")
        return RuleFixtureProvider()
    return LlamaLoRAProvider(base_model, adapter)


def _cases(ws, ds, builder, part, pos, P):
    ids = ds.frame["__id"].to_numpy()
    return [Case(str(ids[i]), float(p), builder.snapshot(int(i))) for i, p in zip(pos[part], P[f"p_{part}"])]


def _index(ws, fixture, backend="flat"):
    if not ws.p("index").exists():
        return None
    return ExemplarIndex.load(ws.p("index"), _embedder(ws, fixture), backend)


def fit_mixer(ws: Workspace, *, base_model=None, adapter=None, fixture: bool = False) -> dict:
    ds = ws.dataset_obj()
    X, y, pos = _features(ws)
    P = np.load(ws.need("triage_proba.npz", "train-triage"))
    th = json.loads(ws.need("triage.json", "train-triage").read_text())["thresholds"]
    pipe = VRFraudNetPipeline(low=th["low"], high=th["high"], provider=_provider(ws, base_model, adapter, fixture), index=_index(ws, fixture),
                              mode="demo" if fixture else "research")
    res = pipe.run_many(_cases(ws, ds, EvidenceBuilder(ds), "val_select", pos, P))
    mixer = VerifiedMixer().fit([r.triage_probability for r in res], [r.verifier.value for r in res],
                                [np.nan if r.rationale_probability_used is None else r.rationale_probability_used for r in res], y["val_select"])
    joblib.dump(mixer, ws.p("mixer.joblib"))
    info = {"w": mixer.w_, "strata": sorted(mixer.strata_), "n": len(res), "manifest": ws.manifest("fit-mixer", {"fixture": fixture})}
    ws.p("mixer.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def calibrate(ws: Workspace, *, base_model=None, adapter=None, fixture: bool = False) -> dict:
    ds = ws.dataset_obj()
    X, y, pos = _features(ws)
    P = np.load(ws.need("triage_proba.npz", "train-triage"))
    th = json.loads(ws.need("triage.json", "train-triage").read_text())["thresholds"]
    mixer = joblib.load(ws.need("mixer.joblib", "fit-mixer"))
    pipe = VRFraudNetPipeline(low=th["low"], high=th["high"], provider=_provider(ws, base_model, adapter, fixture), index=_index(ws, fixture), mixer=mixer,
                              mode="demo" if fixture else "research")
    res = pipe.run_many(_cases(ws, ds, EvidenceBuilder(ds), "val_calibration", pos, P))
    conf = {a: SplitConformal(a).fit([r.risk_score for r in res], y["val_calibration"]) for a in (CONFORMAL["alpha_primary"], CONFORMAL["alpha_sensitivity"])}
    joblib.dump(conf, ws.p("conformal.joblib"))
    info = {str(a): {"threshold": c.threshold, "n_cal": c.n_cal} for a, c in conf.items()}
    ws.p("conformal.json").write_text(json.dumps({**info, "manifest": ws.manifest("calibrate", {"fixture": fixture})}, indent=1, default=float), encoding="utf-8")
    return info


def predict(ws: Workspace, *, base_model=None, adapter=None, fixture: bool = False, part: str = "test", limit: Optional[int] = None) -> Path:
    ds = ws.dataset_obj()
    X, y, pos = _features(ws)
    P = np.load(ws.need("triage_proba.npz", "train-triage"))
    th = json.loads(ws.need("triage.json", "train-triage").read_text())["thresholds"]
    mixer, conf = joblib.load(ws.need("mixer.joblib", "fit-mixer")), joblib.load(ws.need("conformal.joblib", "calibrate"))
    pipe = VRFraudNetPipeline(low=th["low"], high=th["high"], provider=_provider(ws, base_model, adapter, fixture), index=_index(ws, fixture), mixer=mixer,
                              conformal=conf[CONFORMAL["alpha_primary"]], mode="demo" if fixture else "research")
    cases = _cases(ws, ds, EvidenceBuilder(ds), part, pos, P)[:limit]
    out = ws.p(f"predictions_{part}.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(pipe.run(c).to_json().replace("\n", "") + "\n")
    return out


def evaluate(ws: Workspace) -> dict:
    from ..types import TransactionResult

    X, y, pos = _features(ws)
    res = [TransactionResult.from_dict(json.loads(l)) for l in ws.need("predictions_test.jsonl", "predict").read_text(encoding="utf-8").splitlines() if l]
    P = np.load(ws.p("triage_proba.npz"))
    yt = y["test"][: len(res)]
    thr_pipe = joblib.load(ws.p("mixer.joblib"))
    sel = np.load(ws.p("triage_proba.npz"))["p_val_select"]
    score = np.array([r.risk_score for r in res])
    thr = M.select_f1_threshold(y["val_select"], thr_pipe.transform(sel, ["not_invoked"] * len(sel), [np.nan] * len(sel))[0])
    conf = joblib.load(ws.p("conformal.joblib"))[CONFORMAL["alpha_primary"]]
    out = {"n": len(res), "metrics": M.all_metrics(yt, score, thr), "conformal": conf.coverage(score, yt), "human_review_fraction": float(np.mean([not r.automated for r in res])),
           "provenance": ws.manifest("evaluate", {}), "note": "F1 threshold from VAL_SELECT scores of the common-path mixer stratum (frozen)"}
    ws.p("evaluation.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out
