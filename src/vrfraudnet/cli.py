"""`vrfraudnet` command line. Every subcommand fails early with the exact missing input (exit code 2) instead of guessing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .errors import VRFraudNetError


def _snapshot_from_json(d: dict):
    from .verifier import EvidenceSnapshot, GraphEvidence

    g = d.get("graph")
    graph = GraphEvidence(tuple(tuple(e) for e in g["edges"]), g.get("complete", False), d["prediction_time"], g.get("radius", 99)) if g else None
    return EvidenceSnapshot(d["transaction_id"], d["prediction_time"], d.get("fields", {}), d.get("event_times", {}), d.get("entities", {}), d.get("evidence_index", {}), graph)


def cmd_demo(a):
    from .demo import run_demo

    run_demo(Path(a.out) if a.out else None, seed=a.seed)


def cmd_verify(a):
    from .demo import explain
    from .types import Route, TransactionResult, VerifierState
    from .verifier import verify

    ev = _snapshot_from_json(json.loads(Path(a.evidence).read_text(encoding="utf-8")))
    rat = json.loads(Path(a.rationale).read_text(encoding="utf-8"))
    rep = verify(rat, ev)
    print(f"verifier: {rep.state.value.upper()}")
    for c in rep.claims:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] claim[{c.index}] {c.claim_type:<19} {c.status:<12} {c.reason}")
    for r in rep.rule_failures:
        print(f"  [FAIL] rule: {r}")
    return 0 if rep.accepted else 1


def cmd_data(a):
    from .data.providers import check_dataset
    from .data.registry import REGISTRY, get

    if a.action == "list":
        for d, s in REGISTRY.items():
            print(f"{d:<4} {s.name}\n     page:   {s.page}\n     access: {s.access}\n     files:  {', '.join(f.name for f in s.files) or '-'}\n     split:  {s.split_rule}")
            for amb in s.ambiguities:
                print(f"     NOTE:   {amb}")
        return 0
    rep = check_dataset(a.dataset, Path(a.raw_dir), hash_files=not a.no_hash)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["ok"] else 1


def _ws(a):
    from .experiments.stages import Workspace

    return Workspace(Path(a.work), a.dataset, Path(a.raw_dir) if a.raw_dir else None, a.seed)


def cmd_stage(a):
    from .experiments import stages as S

    ws = _ws(a)
    name = a.cmd
    if name == "prepare":
        info = S.prepare(ws)
        print(json.dumps({k: info[k] for k in ("dataset", "fingerprint", "realised_n_features", "declared_features", "prevalence")}, indent=1, default=float))
    elif name == "train-encoder":
        print(json.dumps(S.train_encoder(ws, epochs=a.epochs, align=a.align, device=a.device), indent=1, default=float)[:2000])
    elif name == "train-triage":
        print(json.dumps(S.train_triage(ws), indent=1, default=float))
    elif name == "build-index":
        print(json.dumps(S.build_index(ws, fixture=a.fixture), indent=1, default=float))
    elif name == "train-rationale":
        from .training.rationale_train import train_lora

        tr, va = S.corpus(ws)
        out = train_lora(tr, va, ws.p("lora"), base_model=a.base_model, epochs=a.epochs, seed=a.seed)
        print(json.dumps(out, indent=1, default=float)[:2000])
    elif name == "fit-mixer":
        print(json.dumps(S.fit_mixer(ws, base_model=a.base_model, adapter=a.adapter, fixture=a.fixture), indent=1, default=float))
    elif name == "calibrate":
        print(json.dumps(S.calibrate(ws, base_model=a.base_model, adapter=a.adapter, fixture=a.fixture), indent=1, default=float))
    elif name == "predict":
        print(S.predict(ws, base_model=a.base_model, adapter=a.adapter, fixture=a.fixture, limit=a.limit))
    elif name == "evaluate":
        print(json.dumps(S.evaluate(ws), indent=1, default=float))
    return 0


def cmd_run(a):
    from .experiments.predictive import ALL_SYSTEMS, run_predictive
    from .models.pipeline import Ablation
    from .models.rationale import LlamaLoRAProvider, RuleFixtureProvider

    provider = None
    if a.fixture:
        if a.dataset.upper() != "SYN":
            raise VRFraudNetError("--fixture is only allowed for dataset SYN (benchmark runs never use the fixture provider)")
        provider = RuleFixtureProvider()
    elif a.base_model:
        provider = LlamaLoRAProvider(a.base_model, a.adapter)
    abl = None
    if a.ablation:
        flag = {"A1": "use_verifier", "A2": "use_gate", "A3": "use_schema", "A6": "use_llm", "A7": "use_mixer"}.get(a.ablation)
        if flag is None:
            raise VRFraudNetError("inference ablations: A1, A2, A3, A6, A7 (A4/A5/A9 are training-time; A8 is a weight sensitivity)")
        abl = Ablation(a.ablation, **{flag: False})
    systems = ALL_SYSTEMS if not a.systems else tuple(a.systems)
    split_kwargs = {"q": a.q} if a.q is not None else None
    for sd in a.seed:
        s = run_predictive(a.dataset, Path(a.raw_dir) if a.raw_dir else None, sd, Path(a.runs), systems=systems, provider=provider, ablation=abl, split_kwargs=split_kwargs,
                           tag=f"_q{a.q:.2f}" if a.q is not None else "", verbose=True, baseline_epochs=a.baseline_epochs, encoder_epochs=a.encoder_epochs, device=a.device)
        print(json.dumps(s, indent=1, default=float))
    return 0


def cmd_sweep(a):
    from .experiments.runner import Context, plan, sweep

    raw = dict(kv.split("=", 1) for kv in (a.raw or []))
    ctx = Context(Path(a.runs), {k: Path(v) for k, v in raw.items()}, a.base_model, a.adapter, None, tuple(a.seed) if a.seed else Context.__dataclass_fields__["seeds"].default)
    print(json.dumps(plan(ctx) if a.plan else sweep(ctx, a.only, dry_run=a.dry_run), indent=1, default=str))
    return 0


def cmd_tables(a):
    from .experiments import tables as T
    from .published import SEEDS

    root, seeds = Path(a.runs), tuple(a.seed) if a.seed else SEEDS
    k = a.table
    if k == "table5":
        t = T.build_table5(root, a.dataset, seeds=seeds, allow_fixture=a.allow_fixture)
    elif k == "table6":
        t = T.build_table6(root, seeds=seeds, allow_fixture=a.allow_fixture)
    elif k == "table7":
        t = T.build_table7(root, a.dataset, seeds, a.allow_fixture)
    elif k == "table8a":
        t = T.build_table8_windows(root, "D1", ["lightgbm", "tabtransformer", "vr_fraudnet"], seeds, a.reference, a.allow_fixture)
    elif k == "table8c":
        t = T.build_table8_windows(root, "D4", ["lightgbm", "tabtransformer", "vr_fraudnet"], seeds, a.reference, a.allow_fixture)
    elif k == "table8b":
        t = T.build_table8b(root, ["lightgbm", "tabtransformer", "vr_fraudnet"], seeds, allow_fixture=a.allow_fixture)
    elif k == "s4a":
        t = T.build_s4a(root, a.dataset, seeds, a.allow_fixture)
    elif k == "s1":
        from .evaluation.cost import CostModel

        t = T.build_s1(root, a.dataset, CostModel(a.c_fn, a.c_fp, a.c_tp), seeds, a.systems or ["lightgbm", "tabtransformer", "vr_fraudnet"], a.allow_fixture)
    else:
        raise VRFraudNetError(f"unknown table {k!r}")
    print(T.to_markdown(t))
    return 0


def cmd_audit(a):
    from .evaluation.audit import run_audit, to_markdown

    f = run_audit()
    if a.out_dir:
        d = Path(a.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "AUDIT.md").write_text(to_markdown(f), encoding="utf-8")
        (d / "audit.json").write_text(json.dumps([x.to_dict() for x in f], indent=1, default=float), encoding="utf-8")
    if a.json:
        print(json.dumps([x.to_dict() for x in f], indent=1, default=float))
    else:
        for x in f:
            print(f"{x.id:<8} {x.severity:<9} {x.status:<13} {x.title}")
    return 1 if any(x.severity == "blocking" and x.status == "inconsistent" for x in f) and a.strict else 0


def cmd_status(a):
    from .experiments.status import render

    text = render(Path(a.runs) if a.runs else None)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_docs(a):
    from .docs import stage_table_markdown, write_figures

    print(write_figures(Path(a.docs_dir)))
    print(stage_table_markdown())
    return 0


def cmd_latency(a):
    from .data.evidence import EvidenceBuilder
    from .data.synthetic import make_synthetic
    from .evaluation.latency import measure
    from .models.pipeline import Case, VRFraudNetPipeline
    from .models.rationale import RuleFixtureProvider

    ds = make_synthetic(600, seed=0)
    eb = EvidenceBuilder(ds)
    pipe = VRFraudNetPipeline(low=0.2, high=0.8, provider=RuleFixtureProvider(), mode="demo")
    p = 0.05 if a.path == "common" else 0.5
    r = measure(lambda i: (pipe.run(Case("x", p, eb.snapshot(300 + (i % 100)))).route and None), path=a.path, warmup=a.warmup, timed=a.timed)
    r["WARNING"] = "FIXTURE timing on this machine; NOT comparable with the paper's A100 numbers (see protocol_match)"
    print(json.dumps(r, indent=1))
    return 0


def cmd_robustness(a):
    import pandas as pd

    from .evaluation.robustness import score_variants

    df = pd.read_parquet(a.file) if a.file.endswith(".parquet") else pd.read_csv(a.file)
    print(json.dumps(score_variants(df, a.threshold, a.dataset, a.denominator), indent=1, default=float))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vrfraudnet", description="Companion implementation of VR-FraudNet (doi:10.1007/s44163-026-02355-0).")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo", help="five-minute CPU walkthrough on a synthetic fixture")
    s.add_argument("--out"); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_demo)
    s = sub.add_parser("verify", help="verify a rationale JSON against an evidence JSON and print why it passes or fails")
    s.add_argument("rationale"); s.add_argument("evidence"); s.set_defaults(f=cmd_verify)
    s = sub.add_parser("data", help="list datasets or check your downloaded files")
    s.add_argument("action", choices=["list", "check"]); s.add_argument("--dataset"); s.add_argument("--raw-dir"); s.add_argument("--no-hash", action="store_true"); s.set_defaults(f=cmd_data)
    for name in ("prepare", "train-encoder", "train-triage", "build-index", "train-rationale", "fit-mixer", "calibrate", "predict", "evaluate"):
        s = sub.add_parser(name, help=f"stage: {name} (workspace-based)")
        s.add_argument("--dataset", required=True); s.add_argument("--raw-dir"); s.add_argument("--work", required=True); s.add_argument("--seed", type=int, default=0)
        s.add_argument("--base-model"); s.add_argument("--adapter"); s.add_argument("--fixture", action="store_true", help="SYN only")
        s.add_argument("--epochs", type=int); s.add_argument("--align", default="mean"); s.add_argument("--device", default="cpu"); s.add_argument("--limit", type=int)
        s.set_defaults(f=cmd_stage)
    s = sub.add_parser("run", help="one predictive experiment: baselines + triage + VR-FraudNet, per seed")
    s.add_argument("--dataset", required=True); s.add_argument("--raw-dir"); s.add_argument("--runs", default="runs"); s.add_argument("--seed", type=int, nargs="+", default=[42])
    s.add_argument("--base-model"); s.add_argument("--adapter"); s.add_argument("--fixture", action="store_true"); s.add_argument("--ablation"); s.add_argument("--systems", nargs="+")
    s.add_argument("--q", type=float); s.add_argument("--baseline-epochs", type=int, default=30); s.add_argument("--encoder-epochs", type=int); s.add_argument("--device", default="cpu"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("sweep", help="plan or run every experiment whose inputs exist (resumable)")
    s.add_argument("--runs", default="runs"); s.add_argument("--raw", nargs="*", help="D1=/path D2=/path ..."); s.add_argument("--base-model"); s.add_argument("--adapter")
    s.add_argument("--seed", type=int, nargs="*"); s.add_argument("--only", nargs="*"); s.add_argument("--dry-run", action="store_true"); s.add_argument("--plan", action="store_true"); s.set_defaults(f=cmd_sweep)
    s = sub.add_parser("tables", help="build a REPRODUCED table from saved run files")
    s.add_argument("table", choices=["table5", "table6", "table7", "table8a", "table8b", "table8c", "s1", "s4a"]); s.add_argument("--runs", default="runs"); s.add_argument("--dataset")
    s.add_argument("--seed", type=int, nargs="*"); s.add_argument("--allow-fixture", action="store_true"); s.add_argument("--reference", default="primary_test")
    s.add_argument("--systems", nargs="*"); s.add_argument("--c-fn", type=float); s.add_argument("--c-fp", type=float); s.add_argument("--c-tp", type=float); s.set_defaults(f=cmd_tables)
    s = sub.add_parser("audit", help="recompute derived quantities in the published article (needs no data)")
    s.add_argument("--out-dir"); s.add_argument("--json", action="store_true"); s.add_argument("--strict", action="store_true"); s.set_defaults(f=cmd_audit)
    s = sub.add_parser("status", help="the status ledger"); s.add_argument("--runs"); s.add_argument("--out"); s.add_argument("--markdown", action="store_true"); s.set_defaults(f=cmd_status)
    s = sub.add_parser("docs", help="regenerate docs/figures from the stage specification"); s.add_argument("--docs-dir", default="docs"); s.set_defaults(f=cmd_docs)
    s = sub.add_parser("latency", help="measure FIXTURE-path timing here (never comparable with the paper unless protocol_match is all true)")
    s.add_argument("--path", choices=["common", "escalation"], default="common"); s.add_argument("--warmup", type=int, default=50); s.add_argument("--timed", type=int, default=200); s.set_defaults(f=cmd_latency)
    s = sub.add_parser("robustness", help="score a SUPPLIED fixed variant-score file"); s.add_argument("file"); s.add_argument("--dataset", required=True)
    s.add_argument("--threshold", type=float, required=True); s.add_argument("--denominator", default="detected_originals"); s.set_defaults(f=cmd_robustness)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "data" and a.action == "check" and not (a.dataset and a.raw_dir):
            raise VRFraudNetError("data check needs --dataset and --raw-dir")
        return a.f(a) or 0
    except VRFraudNetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
