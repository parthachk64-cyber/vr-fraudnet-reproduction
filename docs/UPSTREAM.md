# Upstream reproducibility package: what was examined

This repository is a **separately maintained, independently written** implementation. The authors' reproducibility package is the technical reference it was checked against.

| Field | Value |
|---|---|
| Repository | https://github.com/abdulbasit12312/VR-FraudNet |
| Examined | 2026-11-20 (default branch `main`) |
| Commit examined | `7f552994bde0efe4e6706ceccdfb398d8e6450d0` (2026-09-18, "docs: record the published v1.0.0-manuscript-revision release") |
| Release tag | `v1.0.0-manuscript-revision` -> `a71efdc7ee04ffa9a3475659470cc2451f850eaf` |
| License | MIT, "Copyright (c) 2026 The VR-FraudNet authors" (the GitHub API reports `NOASSERTION`, probably because of the data note appended to the standard MIT text) |
| Code reused here | **None.** No upstream source file, rule file, schema or asset is copied into this repository. Everything was written from the article and its supplement. |

The upstream README states that it ships **no datasets, no model checkpoints, no LoRA adapter and no predictions**. That is what is at the pinned commit: `artifacts/lora/README.md` records that a standalone archival copy of the fitted adapter "was not retained as a release-ready artifact", and `results/` contains only a README and a `.gitkeep`. Its README also says the package "is not a claim to have reproduced the manuscript's numbers". Nothing was fabricated to fill that gap here either.

## Published version of record vs upstream's manuscript revision

Upstream's `docs/MANUSCRIPT_AUDIT.md` was written against an earlier manuscript revision. Each of its findings was re-checked against the **published** article (Discover Artificial Intelligence 6:1403, accepted 2026-09-20). Where upstream's status already says "resolved/withdrawn" and the published text agrees, it is not re-reported.

| Upstream finding | In the published version | This repository |
|---|---|---|
| A-01 D3 Recall@top-1% above its arithmetic ceiling | **Still present** (Table 5c: 0.4234-0.6234 vs ceiling 0.2859). The supplement now supplies the held-out positives (S1c: 4,134) and Table 8d the partition size (118,108), so the ceiling is computed from numbers the paper itself prints. | `vrfraudnet audit` A-01D3 (blocking) |
| A-02 Table 6 deltas disagree with Table 5 | **Resolved** (Table 6 now equals the Table-5-implied differences, +24.68 ... +3.81). | A-02 consistent |
| A-03 DeLong with unpreserved vectors | **Withdrawn** by the authors (Sec. 3.8.3). | not implemented |
| A-05 Eq. 5 malformed | **Still present** in the printed equation; Sec. 3.8.1 states the intended rule. | D-01, A-20 |
| A-11 learning rate "2 x10^4" | **Still present** in Sec. 3.3.1. | D-19, A-21 |
| A-12 effect size unnamed | **Still present**; and recomputing from the printed V values reproduces neither rank-biserial nor Z/sqrt(n). | A-05 unresolvable |
| A-16 D1 drift reference | **Still present**: Table 8a "Train AUPRC" equals the Table 5a *test* value. | A-07 note |
| A-17 escalation latency | Protocol now stated (batch 1, 1,000 warm-up, 10,000 timed, GPU sync, A100 80 GB); still no generated-token count. | `evaluation/latency.py::decode_plausibility` |

### New observations not in upstream's list (found by this repository's audit)

* **Table 6 raw p-values do not belong to the printed V values** for three baselines. Example: p = 1.86e-9 is the exact tail for V = 464, but V = 462 is printed. (A-04)
* **Supplementary S4a n_cal = 100,000 for D1 cannot come from the described split**: the conformal calibration set is half of a validation segment that is 10% of a 1,000,000-row dataset, at most 50,000 rows. (A-19)
* Table 6 prints no V for logistic regression. (A-23)

Every one of these is reproducible with `vrfraudnet audit` and is preserved, not corrected.

## Why an independent implementation

A reproduction that copies the original's code can only reproduce its mistakes. Writing from the specification found the places where the specification is silent (docs/DECISIONS.md). The upstream repository remains the authors' reference for their own pipeline and is linked prominently from the README.
