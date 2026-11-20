"""Stage 2: the schema-constrained rationale model.

Two providers implement one interface:

  * `LlamaLoRAProvider`: the real pathway. Meta-Llama-3.1-8B-Instruct (or any causal LM you point it at) + an optional PEFT LoRA adapter,
    greedy decoding, grammar-masked logits, 384 new tokens at most. It needs weights that are licensed/gated and are NOT in this repository.
  * `RuleFixtureProvider`: a deterministic, rule-based FIXTURE for the offline demo and the tests. `is_fixture` is True and research mode
    refuses it; it never pretends to be the language model.

A rationale is only a PROPOSAL. Nothing here decides whether a claim is true; the verifier does.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from ..errors import MissingInputError, OptionalDependencyError
from ..published import LORA_PARAMS
from ..retrieval.index import Exemplar
from ..verifier.evidence import EvidenceSnapshot
from .grammar import GrammarLogitsProcessor, TokenTrie, build_rationale_grammar, canonicalise, vocab_texts

LLAMA_PAGE = "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
INSTRUCTION = ("You propose a structured fraud rationale as ONE JSON object with keys verdict, probability, claims, evidence_refs. "
               "Every claim must be a typed, machine-checkable statement about the CURRENT transaction's evidence only. "
               "The examples show the output FORMAT; they are not evidence about the current transaction.")


@dataclass(frozen=True)
class Generation:
    text: str
    complete: bool
    n_tokens: int


class RationaleProvider(Protocol):
    name: str
    is_fixture: bool

    def generate(self, ev: EvidenceSnapshot, demos: Sequence[Exemplar], p_triage: float, *, constrained: bool = True) -> Generation: ...


def build_prompt(ev: EvidenceSnapshot, demos: Sequence[Exemplar], p_triage: float, max_demos: int = 4) -> str:
    parts = [INSTRUCTION]
    for i, d in enumerate(list(demos)[:max_demos], 1):
        parts.append(f"EXAMPLE {i} (format only)\nSUMMARY: {d.summary}\nOUTPUT: {canonicalise(d.rationale)}")
    parts.append("CURRENT EVIDENCE: " + json.dumps(ev.to_prompt_dict(), sort_keys=True))
    parts.append(f"TRIAGE_PROBABILITY: {p_triage:.4f}\nOUTPUT: ")
    return "\n\n".join(parts)


# ------------------------------------------------------------------------------------------- fixture
class RuleFixtureProvider:
    """Deterministic, rule-based demo provider. `behaviour` lets the walkthrough and tests show the failure modes."""

    is_fixture = True
    name = "rule-fixture"

    def __init__(self, behaviour: str = "faithful"):
        if behaviour not in ("faithful", "false_claim", "unknown_field", "bad_json", "truncated"):
            raise ValueError(f"unknown behaviour {behaviour!r}")
        self.behaviour = behaviour

    def generate(self, ev, demos, p_triage, *, constrained=True) -> Generation:
        fraud = p_triage >= 0.5
        claims, refs = [], []
        rev = {f: eid for eid, f in ev.evidence_index.items()}
        for f in sorted(ev.fields):
            v = ev.fields[f]
            if isinstance(v, (int, float)) and not isinstance(v, bool) and f in rev:
                # a true statement about the record, deliberately loose so it holds: the fixture is not a detector
                claims.append({"type": "numeric_comparison", "field": f, "op": ">=" if fraud else "<=", "value": math.floor(v) if fraud else math.ceil(v), "evidence_id": rev[f]})
                refs.append(rev[f])
            if len(claims) == 2:
                break
        for f in sorted(ev.fields):
            v = ev.fields[f]
            if isinstance(v, str) and f in rev:
                claims.append({"type": "set_membership", "field": f, "op": "in", "value": [v], "evidence_id": rev[f]})
                refs.append(rev[f])
                break
        if self.behaviour == "false_claim" and claims:
            c = claims[0]
            c["op"], c["value"] = (("<", c["value"] - 1) if c["op"] == ">=" else (">", c["value"] + 1))
        if self.behaviour == "unknown_field" and claims:
            claims[0]["field"] = "invented_field"
        rationale = {"verdict": "fraud" if fraud else "legitimate", "probability": round(min(max(p_triage, 0.0), 1.0), 4),
                     "claims": claims, "evidence_refs": sorted(set(refs))}
        text = canonicalise(rationale)
        if self.behaviour == "bad_json":
            text = text.replace("{", "{{", 1)
        if self.behaviour == "truncated":
            return Generation(text[: len(text) // 2], False, 384)
        return Generation(text, True, max(1, len(text) // 3))


# ------------------------------------------------------------------------------------------- real pathway
def validate_adapter(adapter_dir: Path, expected: dict = LORA_PARAMS) -> list[str]:
    """Compare a PEFT adapter_config.json with the published LoRA configuration. Returns differences (empty = matches)."""
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise MissingInputError(f"adapter_config.json in {adapter_dir}", needed_for="loading a LoRA adapter", how_to_fix="Provide a PEFT adapter directory (see docs/STAGE2.md)")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    diffs = []
    for key, want in (("r", expected["r"]), ("lora_alpha", expected["lora_alpha"]), ("lora_dropout", expected["lora_dropout"])):
        if cfg.get(key) != want:
            diffs.append(f"{key}: adapter has {cfg.get(key)!r}, paper specifies {want!r}")
    if set(cfg.get("target_modules") or []) != set(expected["target_modules"]):
        diffs.append(f"target_modules: adapter has {cfg.get('target_modules')!r}, paper specifies {list(expected['target_modules'])!r}")
    return diffs


class LlamaLoRAProvider:
    is_fixture = False

    def __init__(self, base_model: Optional[str] = None, adapter: Optional[str] = None, *, model=None, tokenizer=None, device: Optional[str] = None,
                 dtype: Optional[str] = None, max_input_tokens: int = LORA_PARAMS["max_input_tokens"], max_new_tokens: int = LORA_PARAMS["max_output_tokens"]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise OptionalDependencyError("transformers and torch", "llm", "the LoRA rationale model") from e
        self.max_input, self.max_new = max_input_tokens, max_new_tokens
        if model is None:
            if not base_model:
                raise MissingInputError("base language model", needed_for="Stage 2 rationale generation",
                                        how_to_fix=f"Accept the licence at {LLAMA_PAGE}, download the weights (huggingface-cli download), and pass --base-model PATH_OR_ID")
            dt = getattr(torch, dtype or ("bfloat16" if torch.cuda.is_available() else "float32"))
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dt)
        if adapter:
            diffs = validate_adapter(Path(adapter))
            if diffs:
                raise ValueError("adapter does not match the published LoRA configuration: " + "; ".join(diffs))
            try:
                from peft import PeftModel
            except ImportError as e:  # pragma: no cover
                raise OptionalDependencyError("peft", "llm", "loading a LoRA adapter") from e
            model = PeftModel.from_pretrained(model, adapter)
        self.model, self.tok = model.eval(), tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.name = f"llm:{getattr(getattr(model, 'config', None), '_name_or_path', 'custom')}" + (f"+lora:{adapter}" if adapter else "")
        self._texts = vocab_texts(tokenizer)
        self._trie = TokenTrie(self._texts)
        self.eos_id = tokenizer.eos_token_id

    def _fit_prompt(self, ev, demos, p):
        for k in range(len(demos), -1, -1):                              # drop demonstrations first if the prompt is too long
            ids = self.tok(build_prompt(ev, demos[:k], p, max_demos=k), return_tensors="pt", add_special_tokens=True).input_ids
            if ids.shape[1] <= self.max_input:
                return ids
        return ids[:, -self.max_input:]

    def generate(self, ev, demos, p_triage, *, constrained: bool = True) -> Generation:
        import torch
        from transformers import LogitsProcessorList

        ids = self._fit_prompt(ev, list(demos), p_triage).to(self.device)
        procs = None
        grammar = build_rationale_grammar(ev)
        if constrained:
            procs = LogitsProcessorList([GrammarLogitsProcessor(grammar, self._trie, self._texts, self.eos_id, ids.shape[1])])
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new, do_sample=False, logits_processor=procs, pad_token_id=self.eos_id, eos_token_id=self.eos_id)
        new = out[0, ids.shape[1]:].tolist()
        if new and new[-1] == self.eos_id:
            new = new[:-1]
        text = "".join(self._texts.get(t, "") for t in new)
        return Generation(text, grammar.accepts(text), len(new))
