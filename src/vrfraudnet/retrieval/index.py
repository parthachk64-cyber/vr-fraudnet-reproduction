"""Stage 2 retrieval: verified TRAINING exemplars shown to the rationale model as format demonstrations.

Rules enforced in code, not by convention:
  * only exemplars from the training partition, that are schema-valid AND verifier-accepted, can be indexed;
  * eligibility (earlier timestamp, different transaction, no shared entity) is applied BEFORE the final top-k,
    so a rejected near-neighbour is replaced by the next eligible one rather than shrinking the result;
  * fewer than k eligible exemplars (or none) is a valid answer; nothing is back-filled from validation/test;
  * a retrieved exemplar is a demonstration. It is never evidence: evidence IDs belong to the current case only.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from ..errors import FixtureInResearchModeError, LeakageError, OptionalDependencyError, SchemaError
from ..schemas import validate_rationale


@dataclass(frozen=True)
class Exemplar:
    id: str
    timestamp: Optional[float]
    entity_ids: frozenset
    summary: str
    rationale: dict
    band: str
    partition: str = "train"
    verified: bool = True

    def to_json(self) -> dict:
        d = asdict(self)
        d["entity_ids"] = sorted(self.entity_ids)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Exemplar":
        return cls(**{**d, "entity_ids": frozenset(d["entity_ids"])})


class Embedder(Protocol):
    name: str
    is_fixture: bool

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free character n-gram hashing embedder. A FIXTURE: it exists so the demo and tests run offline.
    Research mode refuses it (`require_research_embedder`)."""

    is_fixture = True

    def __init__(self, dim: int = 256):
        self.dim, self.name = dim, f"hashing-fixture-{dim}"

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f" {t.lower()} "
            for j in range(len(t) - 2):
                h = int.from_bytes(hashlib.blake2b(t[j:j + 3].encode(), digest_size=4).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-12)


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (the paper's encoder), L2-normalised so inner product = cosine similarity."""

    is_fixture = False

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise OptionalDependencyError("sentence-transformers", "llm", "the MiniLM retrieval encoder") from e
        self.name, self._m = model_name, SentenceTransformer(model_name, device=device)

    def embed(self, texts):
        return np.asarray(self._m.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)


def require_research_embedder(e: Embedder) -> None:
    if getattr(e, "is_fixture", False):
        raise FixtureInResearchModeError(f"embedder {e.name!r} is a fixture; benchmark runs need {SentenceTransformerEmbedder.__name__}")


@dataclass
class Query:
    id: str
    text: str
    timestamp: Optional[float] = None
    entity_ids: frozenset = field(default_factory=frozenset)


class ExemplarIndex:
    def __init__(self, embedder: Embedder, backend: str = "flat"):
        if backend not in ("flat", "faiss"):
            raise ValueError("backend must be 'flat' or 'faiss'")
        self.embedder, self.backend = embedder, backend
        self.exemplars: list[Exemplar] = []
        self._emb: Optional[np.ndarray] = None
        self._faiss = None

    # -------------------------------------------------------------- build
    def build(self, exemplars: Sequence[Exemplar]) -> "ExemplarIndex":
        for e in exemplars:
            if e.partition != "train":
                raise LeakageError(f"exemplar {e.id} comes from partition {e.partition!r}; only 'train' may be indexed")
            if not e.verified:
                raise LeakageError(f"exemplar {e.id} was not accepted by the verifier; unverified exemplars are never indexed")
            errs = validate_rationale(e.rationale)
            if errs:
                raise SchemaError(f"exemplar {e.id} is not schema-valid: {errs[:2]}")
        self.exemplars = list(exemplars)
        self._emb = self.embedder.embed([e.summary for e in self.exemplars]) if self.exemplars else np.zeros((0, 1), np.float32)
        self._build_backend()
        return self

    def _build_backend(self):
        if self.backend == "faiss":
            try:
                import faiss
            except ImportError as e:  # pragma: no cover
                raise OptionalDependencyError("faiss-cpu", "llm", "the FAISS retrieval backend") from e
            self._faiss = faiss.IndexFlatIP(self._emb.shape[1])
            if len(self.exemplars):
                self._faiss.add(np.ascontiguousarray(self._emb))

    def __len__(self) -> int:
        return len(self.exemplars)

    # -------------------------------------------------------------- query
    def eligible(self, q: Query) -> np.ndarray:
        ok = np.ones(len(self.exemplars), dtype=bool)
        for i, e in enumerate(self.exemplars):
            if e.id == q.id:
                ok[i] = False
            elif q.entity_ids and (e.entity_ids & q.entity_ids):
                ok[i] = False
            elif q.timestamp is not None and (e.timestamp is None or not e.timestamp < q.timestamp):
                ok[i] = False                      # same-time or future exemplars are never eligible
        return ok

    def retrieve(self, q: Query, k: int = 4) -> list[Exemplar]:
        if not len(self.exemplars):
            return []
        ok = self.eligible(q)
        if not ok.any():
            return []
        qv = self.embedder.embed([q.text])[0]
        n = len(self.exemplars)
        if self.backend == "flat":
            sims = self._emb @ qv
            sims = np.where(ok, sims, -np.inf)
            order = np.lexsort((np.arange(n), -sims))          # similarity desc, then insertion order (deterministic ties)
            picks = [int(i) for i in order[:k] if np.isfinite(sims[i])]
        else:
            kk = min(n, max(4 * k, 16))
            while True:
                sims, ids = self._faiss.search(qv[None, :].astype(np.float32), kk)
                cand = [(float(s), int(i)) for s, i in zip(sims[0], ids[0]) if i >= 0 and ok[i]]
                if len(cand) >= k or kk >= n:
                    break
                kk = min(n, kk * 2)
            cand.sort(key=lambda t: (-t[0], t[1]))
            picks = [i for _, i in cand[:k]]
        return [self.exemplars[i] for i in picks]

    # -------------------------------------------------------------- persistence
    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "exemplars.jsonl").open("w", encoding="utf-8") as f:
            for e in self.exemplars:
                f.write(json.dumps(e.to_json(), sort_keys=True) + "\n")
        np.save(directory / "embeddings.npy", self._emb)
        manifest = {"embedder": self.embedder.name, "is_fixture_embedder": bool(self.embedder.is_fixture), "n": len(self),
                    "backend": self.backend, "sha256": {n: _sha(directory / n) for n in ("exemplars.jsonl", "embeddings.npy")}}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return manifest

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, backend: str = "flat") -> "ExemplarIndex":
        man = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for n, h in man["sha256"].items():
            if _sha(directory / n) != h:
                raise LeakageError(f"index file {n} does not match its recorded checksum")
        if man["embedder"] != embedder.name:
            raise ValueError(f"index was built with {man['embedder']!r}, not {embedder.name!r}")
        idx = cls(embedder, backend)
        idx.exemplars = [Exemplar.from_json(json.loads(l)) for l in (directory / "exemplars.jsonl").read_text(encoding="utf-8").splitlines() if l]
        idx._emb = np.load(directory / "embeddings.npy")
        idx._build_backend()
        return idx


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarise(fields: dict, band: str, prob: float) -> str:
    """Standardised transaction summary used for BOTH indexing and querying (same function = same text space)."""
    items = ", ".join(f"{k}={v}" for k, v in sorted(fields.items()) if re.match(r"^[a-z]", k))
    return f"band={band}; p_triage={prob:.2f}; {items}"
