"""The REAL transformers/PEFT code paths, exercised on a tiny randomly-initialised Llama with a character-level tokenizer.

This proves the plumbing (grammar-masked generation inside `generate`, `from_pretrained`, LoRA training, adapter loading and validation), not the
quality of any model. It has not been run on Meta-Llama-3.1-8B-Instruct: that needs licensed weights and a GPU (see docs/STAGE2.md, marker `weights`).
"""
import json
import string

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from vrfraudnet.data.evidence import EvidenceBuilder
from vrfraudnet.data.synthetic import make_synthetic
from vrfraudnet.errors import MissingInputError
from vrfraudnet.models.grammar import build_rationale_grammar, canonicalise
from vrfraudnet.models.rationale import LlamaLoRAProvider, build_prompt, validate_adapter
from vrfraudnet.schemas import validate_rationale
from vrfraudnet.training.objectives import RuleTargetBuilder, TargetStats, make_counterfactual
from vrfraudnet.training.rationale_train import train_lora
from vrfraudnet.verifier import verify_text


def make_tiny_llama(path):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    chars = list(string.printable.replace("\n", "").replace("\r", "").replace("\t", "").replace("\x0b", "").replace("\x0c", ""))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + chars)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=4096, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_llama(tmp_path_factory.mktemp("tiny_llama"))


@pytest.fixture(scope="module")
def ev():
    return EvidenceBuilder(make_synthetic(900, seed=5)).snapshot(700)


def test_missing_base_model_is_a_named_error():
    with pytest.raises(MissingInputError) as e:
        LlamaLoRAProvider()
    assert "meta-llama/Meta-Llama-3.1-8B-Instruct" in str(e.value) and "licence" in str(e.value).lower()


def test_constrained_generation_inside_hf_generate_yields_schema_valid_output(tiny, ev):
    p = LlamaLoRAProvider(str(tiny), max_input_tokens=3000, max_new_tokens=1200)
    assert not p.is_fixture
    gen = p.generate(ev, [], 0.5, constrained=True)
    assert gen.n_tokens > 0
    if gen.complete:
        obj = json.loads(gen.text)
        assert validate_rationale(obj, allowed_fields=ev.allowed_fields, allowed_evidence_ids=ev.evidence_ids) == []
    else:                                                               # token limit before the object closed: a viable prefix, rejected downstream
        assert build_rationale_grammar(ev).viable(gen.text)
        assert not verify_text(gen.text, ev).accepted


def test_a_token_limit_leaves_incomplete_output_which_is_rejected(tiny, ev):
    p = LlamaLoRAProvider(str(tiny), max_input_tokens=3000, max_new_tokens=10)
    gen = p.generate(ev, [], 0.5, constrained=True)
    assert not gen.complete and gen.n_tokens == 10
    assert not verify_text(gen.text, ev).accepted


def test_unconstrained_generation_from_a_random_model_is_not_valid(tiny, ev):
    p = LlamaLoRAProvider(str(tiny), max_input_tokens=3000, max_new_tokens=60)
    gen = p.generate(ev, [], 0.5, constrained=False)
    assert not verify_text(gen.text, ev).accepted                       # what the grammar mask (ablation A3) is for


def test_prompt_contains_only_current_evidence_and_format_examples(ev):
    from vrfraudnet.retrieval.index import Exemplar
    demo = Exemplar("D1", 1.0, frozenset(), "wire amount=1", {"verdict": "uncertain", "probability": 0.5, "claims": [], "evidence_refs": []}, "review")
    txt = build_prompt(ev, [demo], 0.42)
    assert "format only" in txt and "CURRENT EVIDENCE" in txt and ev.transaction_id in txt and "TRIAGE_PROBABILITY: 0.4200" in txt


def _corpus(n=24):
    ds = make_synthetic(1500, seed=6)
    eb = EvidenceBuilder(ds)
    snaps = [eb.snapshot(i) for i in range(400, 400 + n + 100)]
    b = RuleTargetBuilder(TargetStats.fit(snaps[:100]))
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for s, y in zip(snaps[100:], ds.frame["__label"].iloc[500:500 + n + 100]):
        rat = b.build(s, 0.5, int(y))
        if rat is None:
            continue
        row = {"prompt": build_prompt(s, [], 0.5), "target": canonicalise(rat)}
        cf = make_counterfactual(s, rat, rng)
        if cf:
            row.update(cf_prompt=build_prompt(cf[0], [], 0.5), cf_target=canonicalise(cf[1]))
        rows.append(row)
    return rows


def test_lora_training_saves_a_valid_adapter_and_the_provider_can_load_it(tiny, tmp_path, ev):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _corpus()
    assert len(rows) >= 8 and any("cf_prompt" in r for r in rows)
    tok = AutoTokenizer.from_pretrained(tiny)
    base = AutoModelForCausalLM.from_pretrained(tiny)
    out = train_lora(rows[:8], rows[8:12], tmp_path / "lora", model=base, tokenizer=tok, epochs=1, micro_batch=2, max_steps=2, seed=0,
                     params={"max_input_tokens": 1500, "max_output_tokens": 400, "effective_batch_size": 4})
    best = out["selected_dir"]
    assert (tmp_path / "lora" / "training_manifest.json").is_file() and out["history"][0]["val_loss"] > 0
    assert validate_adapter(best) == []                                  # r=16, alpha=32, dropout=0.05, q/k/v/o: the published configuration
    p = LlamaLoRAProvider(str(tiny), adapter=best, max_input_tokens=3000, max_new_tokens=1200)
    gen = p.generate(ev, [], 0.5)
    assert gen.n_tokens > 0 and "lora" in p.name


def test_base_weights_are_frozen_during_lora_training(tiny, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(tiny)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    rows = _corpus(12)
    train_lora(rows[:6], rows[6:8], tmp_path / "l2", model=base, tokenizer=AutoTokenizer.from_pretrained(tiny), epochs=1, micro_batch=2, max_steps=2,
               params={"max_input_tokens": 1500, "max_output_tokens": 400, "effective_batch_size": 4})
    after = base.state_dict()
    changed = [k for k in before if "lora" not in k and not torch.equal(before[k], after[k.replace("", "")] if k in after else before[k])]
    pm = [k for k in after if "base_layer" in k]
    assert pm, "PEFT wraps the targeted projections"
    # the wrapped base_layer weights equal the originals
    for k in pm:
        orig = k.replace(".base_layer", "")
        assert torch.equal(after[k], before[orig])


def test_adapter_with_the_wrong_configuration_is_refused(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "target_modules": ["q_proj"]}), encoding="utf-8")
    diffs = validate_adapter(d)
    assert len(diffs) == 4 and any("r:" in x for x in diffs)
    with pytest.raises(MissingInputError):
        validate_adapter(tmp_path / "nope")
