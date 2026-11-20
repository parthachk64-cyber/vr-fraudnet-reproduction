"""Latency measurement that records what was actually measured.

The paper's protocol (Sec. 3.8.2): batch size 1, 1,000 warm-up and 10,000 timed transactions, GPU synchronised
immediately before and after each timed inference, percentiles from per-transaction times, on an A100 80 GB.
`measure` follows that protocol when asked, but REPORTS the real hardware and whether each protocol element
was met. A CPU demo timing must never be presented against the paper's numbers.
"""
from __future__ import annotations

import platform
import time
from typing import Callable, Optional

import numpy as np

from ..published import LATENCY_PROTOCOL


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor() or platform.machine(),
            "cpu_count": __import__("os").cpu_count(), "gpu": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:  # torch is optional
        pass
    return info


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def measure(fn: Callable[[int], Optional[int]], *, path: str, warmup: int = 100, timed: int = 1000, batch_size: int = 1) -> dict:
    """Time `fn(i)` for i in range(timed). `fn` may return the number of tokens it generated (counted and reported)."""
    for i in range(warmup):
        fn(-1 - i)
    times, tokens = [], []
    for i in range(timed):
        _sync()
        t0 = time.perf_counter()
        n = fn(i)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
        if n is not None:
            tokens.append(int(n))
    a = np.asarray(times)
    hw = hardware_info()
    out = {"path": path, "n_timed": timed, "n_warmup": warmup, "batch_size": batch_size, "gpu_synchronised": hw["gpu"] is not None,
           "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
           "mean_ms": float(a.mean()), "hardware": hw,
           "generated_tokens_mean": float(np.mean(tokens)) if tokens else None,
           "protocol_match": {
               "batch_size_1": batch_size == LATENCY_PROTOCOL["batch_size"], "warmup_ge_1000": warmup >= LATENCY_PROTOCOL["warmup"],
               "timed_ge_10000": timed >= LATENCY_PROTOCOL["timed"], "gpu_synchronised": hw["gpu"] is not None,
               "paper_hardware_class": bool(hw["gpu"] and "A100" in hw["gpu"])}}
    out["comparable_to_paper"] = all(out["protocol_match"].values())
    return out


def decode_plausibility(p99_ms: float, generated_tokens: Optional[float], *, tokens_per_s_ceiling: float = 120.0) -> dict:
    """Sanity check for an escalation-path latency: could that many tokens be decoded that fast at batch size 1?

    `tokens_per_s_ceiling` is a deliberately generous single-stream decoding rate for an 8B model in bf16 on one GPU
    (an assumption, stated, not a measurement). With no token count the check cannot run, and says so.
    """
    if not generated_tokens:
        return {"checked": False, "reason": "generated-token count not reported"}
    floor_ms = 1000.0 * generated_tokens / tokens_per_s_ceiling
    return {"checked": True, "min_plausible_ms": floor_ms, "p99_ms": p99_ms, "plausible": p99_ms >= floor_ms,
            "assumed_tokens_per_s_ceiling": tokens_per_s_ceiling}
