# Methods and evaluation

* **Metrics** (`evaluation/metrics.py`): AUPRC (step-wise average precision, D-08), ROC-AUC, F1 at a threshold frozen on VAL_SELECT, Recall@top-1%, MCC, ECE (10 equal-width bins, D-09).
* **Statistics** (`evaluation/stats.py`): one-sided Wilcoxon signed-rank (exact tail when small), Holm adjustment, effect size under both common conventions. Seeds repeated on identical records are dependent, so this is a secondary sensitivity analysis.
* **Cost** (`evaluation/cost.py`): explicit cost model; unit costs are an experiment assumption recorded by the caller.
* **Mathematical audit** (`evaluation/audit.py`, `vrfraudnet audit`): recomputes quantities implied by published numbers, for example the Recall@top-1% ceiling `min(1, 0.01 N / P)`. Findings are listed, never corrected: `docs/AUDIT.md`.
* **Robustness** (`evaluation/robustness.py`): scores a supplied, frozen variant-score file. The repository does not generate or search for edits that evade a detector.
* **Provenance**: run files carry `kind` = `synthetic_fixture` or `benchmark`. Table builders read only `runs/`, refuse fixtures unless `--allow-fixture` (then watermark), and fail on missing seeds.
