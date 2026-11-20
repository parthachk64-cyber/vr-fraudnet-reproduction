# Stage 2: the rationale model

Specified: Llama-3.1-8B-Instruct, LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o, 3 epochs, lr 2e-4, bf16, greedy decoding, at most 8 claims, MiniLM/FAISS retrieval of at most 4 verified train exemplars.

Implemented and tested on a **tiny random Llama** (`tests/test_llm_pathway.py`, needs the `llm` extra): grammar-to-NFA compilation with vocabulary-trie token masks, `GrammarLogitsProcessor` for HF `generate`, LoRA training with a frozen base, adapter save/load validation, checksummed retrieval index.

Not available: the authors' fitted adapter (upstream records it as not retained). A paper-scale Stage 2 therefore means training your own adapter on verified review-band train targets (`vrfraudnet train-rationale`). Training targets come from the deterministic rule builder in D-15, not the authors' unpublished authoring, so results will differ and must be reported as a reconstruction.

Schema: `schemas/rationale_v1.json` (ours, D-03), cross-checked against an independent validator in tests.
