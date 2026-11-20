"""Loader for the PUBLISHED numbers in references/reported/.

Separation rule: published numbers are reference data. Inference, training and the table builders for
reproduced results never import this module; only the audit, the status ledger and the explicit
"reported vs reproduced" comparison do.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .errors import MissingInputError


def references_dir() -> Path:
    env = os.environ.get("VRFRAUDNET_REFERENCES")
    candidates = [Path(env)] if env else []
    candidates += [Path(__file__).resolve().parents[2] / "references" / "reported", Path.cwd() / "references" / "reported"]
    for c in candidates:
        if c.is_dir():
            return c
    raise MissingInputError("references/reported", needed_for="audit and reported-vs-reproduced comparison",
                            how_to_fix="run from a repository checkout, or set VRFRAUDNET_REFERENCES to the directory")


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    return json.loads((references_dir() / f"{name}.json").read_text(encoding="utf-8"))


def row(rows: list[dict], field: str, value: str) -> dict:
    for r in rows:
        if r.get(field) == value:
            return r
    raise KeyError(f"{value!r} not found in column {field!r}")


def mean(cell) -> float:
    """A published cell is {'mean': x, 'sd': y}, {'value': x} or a bare number."""
    if isinstance(cell, dict):
        return float(cell["mean"] if "mean" in cell else cell["value"])
    return float(cell)
