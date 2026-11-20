"""Run manifests and checkpoint selection. Every output a number can be traced to carries one of these."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "xgboost", "torch", "transformers", "peft", "sentence-transformers", "faiss-cpu", "faiss-gpu")


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def git_revision(root: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root or Path.cwd(), capture_output=True, text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a repository, or git absent
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def build_manifest(*, kind: str, dataset: str, seed: Optional[int], config: dict, split_fingerprint: Optional[str] = None, artifacts: Optional[dict] = None,
                   model_revision: Optional[str] = None, adapter_revision: Optional[str] = None, metric_definition: Optional[dict] = None, notes=()) -> dict:
    """`kind` is 'benchmark' or 'synthetic_fixture'. Table builders for reproduced results refuse the latter."""
    if kind not in ("benchmark", "synthetic_fixture"):
        raise ValueError("kind must be 'benchmark' or 'synthetic_fixture'")
    return {"kind": kind, "dataset": dataset, "seed": seed, "split_fingerprint": split_fingerprint, "config": config, "config_hash": config_hash(config),
            "git_revision": git_revision(), "packages": package_versions(), "base_model_revision": model_revision, "adapter_revision": adapter_revision,
            "metric_definition": metric_definition or {"auprc": "average_precision (step interpolation, distinct thresholds)", "recall_top_k": "k=ceil(0.01N), ties to lower row index",
                                                       "f1_threshold": "validation-selected, frozen, largest among ties", "orientation": "1 = fraud, higher = riskier"},
            "artifact_sha256": artifacts or {}, "notes": list(notes), "argv": sys.argv[1:]}


def select_checkpoint(history: list[dict]) -> int:
    """Index of the checkpoint to keep: lowest validation loss; ties broken by higher schema-valid rate, then higher verifier-acceptance rate
    (Sec. 3.3.1: 'validation loss, with the schema-valid generation rate and verifier acceptance rate ... as secondary selection criteria')."""
    if not history:
        raise ValueError("empty history")
    return min(range(len(history)), key=lambda i: (round(history[i]["val_loss"], 6), -history[i].get("schema_valid_rate", 0.0), -history[i].get("verifier_acceptance_rate", 0.0), i))
