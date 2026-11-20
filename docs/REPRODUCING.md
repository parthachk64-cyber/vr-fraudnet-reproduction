# Reproducing at paper scale

Nothing below has been run by this repository on real benchmark data. Follow it, then read `docs/STATUS.md`; items flip to `benchmark_run` only when a run file exists.

1. `pip install -e ".[trees,graph,llm,bench,dev]"` (GPU torch per your CUDA).
2. Download D1 first (smallest). `vrfraudnet data check --dataset D1 --raw-dir <dir>`.
3. `vrfraudnet prepare --dataset D1 --raw-dir <dir>`.
4. Baselines and triage: `vrfraudnet run --dataset D1 --raw-dir <dir> --seed 42 123 456 789 2024 3141 2718 1001 1729 4096`.
5. Stage 2 needs the gated Meta-Llama-3.1-8B-Instruct weights and your own adapter: `vrfraudnet train-rationale ...`, then `fit-mixer`, `calibrate`, `predict`, `evaluate`.
6. `vrfraudnet tables table5 --dataset D1`, then D2/D3 and `table6`. Compare against `references/reported/` by hand; builders never read it.
7. `vrfraudnet audit` needs no data.

## Versions

The paper reports Python 3.11.7 and Transformers 4.40.1. Llama 3.1 uses `rope_scaling` type `llama3`, added in Transformers 4.43, so 4.40.1 is not expected to load the specified base model. This repository is tested with Transformers 4.43.4 on Python 3.10 (CI on 3.10-3.12). The 4.40.1 failure is inferred from release history and has not been observed here.

## Minimum missing artifacts for an exact rerun

Provider-terms dataset downloads; gated Llama-3.1-8B weights; the authors' LoRA adapter (not distributed); the frozen edited-variant benchmark files for Table 9; an A100 80 GB host for latency; the unstated D4 shock step and other items in `docs/DECISIONS.md`.
