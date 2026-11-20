"""Stage 2 LoRA training (Sec. 3.3.1 / 3.8.1): frozen base model, LoRA on q/k/v/o, r=16, alpha=32, dropout 0.05, AdamW lr 2e-4 (the 'x10^4' in
Sec. 3.3.1 is a typesetting loss; Sec. 3.8.1 prints 0.0002), weight decay 0.01, 5% linear warm-up, 3 epochs, effective batch 32, <=1,024 input and
<=384 output tokens, bfloat16 where supported.

Loss = CE(target | prompt) + lambda_cf * CE(counterfactual target | counterfactual prompt)  (lambda_cf = 0.25). The paper names a "counterfactual
rationale-stability mechanism" and a weight but not its form; this is a reconstruction (docs/DECISIONS.md D-14).
Only verified, review-band, TRAINING-partition targets belong in `corpus` (see `training.objectives.RuleTargetBuilder`).
This function needs PEFT and a base model. It has been exercised end to end on a tiny randomly-initialised Llama (tests/test_llm_pathway.py);
it has NOT been run on Meta-Llama-3.1-8B-Instruct here (no licensed weights, no GPU).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..errors import MissingInputError, OptionalDependencyError
from ..published import LORA_PARAMS, OBJECTIVE_WEIGHTS
from .manifest import build_manifest, select_checkpoint


def _lora_config(params: dict):
    from peft import LoraConfig

    return LoraConfig(r=params["r"], lora_alpha=params["lora_alpha"], lora_dropout=params["lora_dropout"], target_modules=list(params["target_modules"]),
                      task_type="CAUSAL_LM", bias="none")


def _encode(tok, prompt: str, target: str, max_in: int, max_out: int):
    p = tok(prompt, add_special_tokens=True).input_ids[-max_in:]
    t = tok(target, add_special_tokens=False).input_ids[:max_out] + [tok.eos_token_id]
    return p + t, [-100] * len(p) + t


def _batchify(tok, rows, device):
    import torch

    n = max(len(r[0]) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([r[0] + [pad] * (n - len(r[0])) for r in rows], device=device)
    lab = torch.tensor([r[1] + [-100] * (n - len(r[1])) for r in rows], device=device)
    att = torch.tensor([[1] * len(r[0]) + [0] * (n - len(r[0])) for r in rows], device=device)
    return ids, att, lab


def train_lora(corpus: Sequence[dict], val_corpus: Sequence[dict], out_dir: Path, *, base_model: Optional[str] = None, model=None, tokenizer=None,
               params: Optional[dict] = None, micro_batch: int = 4, epochs: Optional[int] = None, max_steps: Optional[int] = None, seed: int = 0,
               device: Optional[str] = None, eval_fn: Optional[Callable] = None, lambda_cf: float = OBJECTIVE_WEIGHTS["counterfactual"]) -> dict:
    """`corpus` rows: {"prompt", "target"} and optionally {"cf_prompt", "cf_target"}. `eval_fn(model) -> {"schema_valid_rate", "verifier_acceptance_rate"}` supplies
    the secondary checkpoint criteria (generation-based, so it is supplied by the caller)."""
    try:
        import torch
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as e:  # pragma: no cover
        raise OptionalDependencyError("torch, transformers, peft", "llm", "LoRA training") from e
    p = {**LORA_PARAMS, **(params or {})}
    if model is None:
        if not base_model:
            raise MissingInputError("base language model", needed_for="LoRA training", how_to_fix="Accept the licence at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct, download it, pass --base-model")
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=getattr(torch, p["dtype"]) if torch.cuda.is_available() else torch.float32)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    for prm in model.parameters():
        prm.requires_grad_(False)                                     # base weights stay frozen
    model = get_peft_model(model, _lora_config(p)).to(device)
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=p["lr"], weight_decay=p["weight_decay"])
    accum = max(1, p["effective_batch_size"] // micro_batch)
    epochs = epochs or p["epochs"]
    steps_per_epoch = max(1, math.ceil(len(corpus) / (micro_batch * accum)))
    total = max_steps or steps_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(p["warmup_ratio"] * total), total)
    enc = lambda r, pre="": _encode(tokenizer, r[pre + "prompt"], r[pre + "target"], p["max_input_tokens"], p["max_output_tokens"])
    train_rows = [(enc(r), enc(r, "cf_") if "cf_prompt" in r else None) for r in corpus]
    val_rows = [enc(r) for r in val_corpus]
    out_dir = Path(out_dir)
    history, step = [], 0

    def loss_of(rows):
        ids, att, lab = _batchify(tokenizer, rows, device)
        return model(input_ids=ids, attention_mask=att, labels=lab).loss

    done = False
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        run_loss, nb = 0.0, 0
        for i in range(0, len(order), micro_batch):
            chunk = [train_rows[j] for j in order[i:i + micro_batch]]
            loss = loss_of([c[0] for c in chunk]) / accum
            cf = [c[1] for c in chunk if c[1] is not None]
            if cf:
                loss = loss + lambda_cf * loss_of(cf) / accum
            loss.backward()
            run_loss += float(loss) * accum
            nb += 1
            if nb % accum == 0 or i + micro_batch >= len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if max_steps and step >= max_steps:
                    done = True
                    break
        model.eval()
        with torch.no_grad():
            vl = sum(float(loss_of(val_rows[i:i + micro_batch])) for i in range(0, len(val_rows), micro_batch)) / max(1, math.ceil(len(val_rows) / micro_batch))
        rec = {"epoch": epoch, "train_loss": run_loss / max(nb, 1), "val_loss": vl, "steps": step, **(eval_fn(model) if eval_fn else {})}
        history.append(rec)
        ckpt = out_dir / f"epoch_{epoch}"
        model.save_pretrained(ckpt)
        if done:
            break
    best = select_checkpoint(history)
    best_dir = out_dir / f"epoch_{best}"
    manifest = build_manifest(kind="benchmark", dataset="stage2", seed=seed, config={"lora": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}, "lambda_cf": lambda_cf, "micro_batch": micro_batch},
                              model_revision=base_model, notes=["checkpoint selected by validation loss, then schema-valid rate, then verifier acceptance"])
    (out_dir / "training_manifest.json").write_text(json.dumps({"history": history, "selected_epoch": best, "selected_dir": str(best_dir), "manifest": manifest}, indent=1, default=str), encoding="utf-8")
    return {"history": history, "selected_epoch": best, "selected_dir": str(best_dir)}
