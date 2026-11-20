# vr-fraudnet-reproduction

A separately maintained, independently written companion implementation of **VR-FraudNet** (Sourav et al., *Discover Artificial Intelligence* 6:1403, 2026, [doi:10.1007/s44163-026-02355-0](https://doi.org/10.1007/s44163-026-02355-0)).

> **Authors' own reproducibility package: <https://github.com/abdulbasit12312/VR-FraudNet>.** That repository is the authors' reference. This one is a different codebase, written from the article and its Supplementary DOCX (S1-S4). **No upstream code is copied.** Provenance, commit examined, license and differences: [docs/UPSTREAM.md](docs/UPSTREAM.md).

It lets you follow one transaction through the five stages, see exactly why a rationale passed or failed verification, and rerun the predictive, calibration and statistical evaluations **when the data and artifacts exist**. It ships no datasets, no model weights and no published numbers inside inference code.

## Quick start (CPU, no downloads, about five minutes)

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[trees,dev]"
vrfraudnet demo                                      # one transaction through all five stages, synthetic fixture
vrfraudnet verify examples/rationale_accepted.json examples/evidence_wire.json
vrfraudnet verify examples/rationale_rejected.json examples/evidence_wire.json
vrfraudnet audit                                     # recomputes derived numbers of the published article; needs no data
pytest -q
```

The demo runs on an **obviously synthetic fixture**. Its output exercises the code and is never a reproduction of any published result.

## The five stages

| Stage | What | Code |
|---|---|---|
| 0 | Time-conditioned spectral graph encoder (Beta-kernel filters, 32-d time embedding, 64-d output) | `models/spectral.py`, `training/encoder_train.py` |
| 1 | LightGBM triage, class-balanced focal surrogate, inclusive review band | `models/triage.py` |
| 2 | Schema-constrained rationale LM (Llama-3.1-8B + LoRA, grammar-masked greedy decoding, train-only retrieval) | `models/rationale.py`, `models/grammar.py`, `retrieval/` |
| 3 | Fixed deterministic verifier: five typed primitives plus cross-claim and verdict consistency. Unknown is never absent | `verifier/` |
| 4 | Isotonic mixer and split-conformal prediction sets | `models/mixer.py`, `calibration/conformal.py` |

Details, shapes, equation-to-source map and failure policy: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status at a glance

Authoritative ledger: [docs/STATUS.md](docs/STATUS.md) (states: implemented / fixture-exercised / benchmark-run / blocked).

* **Done and tested:** verifier, grammar-constrained decoding (tiny random Llama), splits and leak-safe transforms, mixer, conformal, metrics, Wilcoxon/Holm, cost model, table builders, mathematical audit.
* **Blocked, with named reasons:** every benchmark table (datasets need your own provider-terms download), Stage 2 at paper scale (gated Llama-3.1-8B weights, the authors' LoRA adapter is not distributed), Table 9 (frozen edited-variant files not distributed), latency (A100 host).
* **No benchmark has been run by this repository yet.** Nothing here claims to have reproduced the paper's experiments.

## Datasets

Five adapters (D1 BAF, D2 AMLworld HI-Small, D3 IEEE-CIS, D4 Elliptic++, D5 DGraph-Fin). None is redistributed. `vrfraudnet data list` prints access pages and terms; `vrfraudnet data check --dataset D1 --raw-dir <dir>` validates your files. See [docs/DATA.md](docs/DATA.md).

## Transcribed versus recalculated

* `references/reported/` holds numbers **transcribed from the article** and its supplement, for comparison only. They are never read by inference, training or table builders.
* `runs/` holds numbers **this code produced**. Table builders read only `runs/`, refuse fixture runs unless asked (then watermark them), and error on missing seeds.
* `vrfraudnet audit` **recalculates** quantities implied by the published numbers (for example the Recall@top-1% ceiling) and lists inconsistencies without correcting them: [docs/AUDIT.md](docs/AUDIT.md).

## Reconstructed choices

Where the paper is silent this repository chose, and recorded each choice, its alternative and its code location: [docs/DECISIONS.md](docs/DECISIONS.md). Reproducing at paper scale: [docs/REPRODUCING.md](docs/REPRODUCING.md).

## What this repository will not do

It will not search for edits that evade a detector. Robustness (Table 9) only **scores supplied, frozen** variant files.

## About

Maintained by **Partha Chakraborty** ([@parthachk64-cyber](https://github.com/parthachk64-cyber)), a co-author of the article. His role in the published work, as stated in the article: probability calibration, split-conformal analysis, temporal-drift assessment, statistical verification, and independent checking of revised results and cross-references. Software authorship is claimed here only for code the maintainer adopts and commits under his own identity.

How to cite: the **article** is the preferred citation (see [CITATION.cff](CITATION.cff)); the software has separate metadata in the same file.

## License

MIT, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
