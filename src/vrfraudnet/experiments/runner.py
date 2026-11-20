"""Experiment registry and sweep runner: one command per experiment, a dry-run that names exactly what is missing, and resumable execution.

`plan` never raises for missing inputs: it reports them. `sweep` runs the experiments whose inputs exist, skips any (dataset, seed) whose
summary.json is already on disk (resume), and records blocked ones with their reason instead of fabricating results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..data.providers import check_dataset
from ..published import SEEDS
from .predictive import ALL_SYSTEMS, run_predictive


@dataclass
class Context:
    runs_root: Path
    raw_dirs: dict = field(default_factory=dict)       # dataset id -> Path
    base_model: Optional[str] = None
    adapter: Optional[str] = None
    variants_dir: Optional[Path] = None
    seeds: Sequence[int] = SEEDS


EXPERIMENTS = {
    "predictive-D1": {"title": "Table 5a (+ Tables 6, 8a inputs)", "datasets": ["D1"], "llm": True},
    "predictive-D2": {"title": "Table 5b (+ Table 6 input)", "datasets": ["D2"], "llm": True},
    "predictive-D3": {"title": "Table 5c (+ Table 6 input)", "datasets": ["D3"], "llm": True},
    "predictive-D4": {"title": "D4 strict-inductive + shock windows (Table 8c)", "datasets": ["D4"], "llm": True},
    "predictive-D5": {"title": "D5 message-passing stress test", "datasets": ["D5"], "llm": True},
}


def _missing(ctx: Context, spec: dict) -> list[str]:
    miss = []
    for d in spec["datasets"]:
        raw = ctx.raw_dirs.get(d)
        if raw is None:
            miss.append(f"--raw-dir for {d}")
        else:
            rep = check_dataset(d, Path(raw), hash_files=False)
            if not rep["ok"]:
                miss += [f"{d}: {p}" for p in rep["problems"]]
    return miss


def llm_missing(ctx: Context) -> list[str]:
    out = []
    if not ctx.base_model:
        out.append("base language model (Meta-Llama-3.1-8B-Instruct; gated)")
    if not ctx.adapter:
        out.append("LoRA adapter (the authors' adapter is not distributed)")
    return out


def plan(ctx: Context) -> list[dict]:
    rows = []
    for eid, spec in EXPERIMENTS.items():
        data_missing = _missing(ctx, spec)
        rows.append({"id": eid, "title": spec["title"], "baselines": "ready" if not data_missing else "blocked", "missing_for_baselines": data_missing,
                     "vr_fraudnet": "ready" if not data_missing and not llm_missing(ctx) else "blocked", "missing_for_vr_fraudnet": data_missing + llm_missing(ctx),
                     "done_seeds": [s for s in ctx.seeds if (Path(ctx.runs_root) / spec["datasets"][0] / f"seed{s}" / "summary.json").is_file()]})
    return rows


def sweep(ctx: Context, only: Optional[Sequence[str]] = None, dry_run: bool = False, provider=None, **kw) -> list[dict]:
    log = []
    for row in plan(ctx):
        if only and row["id"] not in only:
            continue
        spec = EXPERIMENTS[row["id"]]
        d = spec["datasets"][0]
        if row["baselines"] == "blocked":
            log.append({"id": row["id"], "status": "blocked", "missing": row["missing_for_baselines"]})
            continue
        for seed in ctx.seeds:
            if seed in row["done_seeds"]:
                log.append({"id": row["id"], "seed": seed, "status": "skipped (already done)"})
                continue
            if dry_run:
                log.append({"id": row["id"], "seed": seed, "status": "would run", "vr_fraudnet": row["vr_fraudnet"]})
                continue
            systems = ALL_SYSTEMS if provider is not None else tuple(s for s in ALL_SYSTEMS if s != "vr_fraudnet")
            s = run_predictive(d, ctx.raw_dirs[d], seed, ctx.runs_root, systems=systems, provider=provider, **kw)
            log.append({"id": row["id"], "seed": seed, "status": "done", "summary": s["systems"]})
    return log
