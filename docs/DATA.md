# Data

No dataset is redistributed. Run `vrfraudnet data list` for each dataset's page, access terms, expected files, split contract and the paper's ambiguities; run `vrfraudnet data check --dataset D1 --raw-dir <dir>` to validate your download (file presence, schema, optional hashes).

| ID | Dataset | Access | Split contract |
|---|---|---|---|
| D1 | BAF Base (NeurIPS 2022) | Kaggle, terms | months 0-5 train/dev (last 10% validation), 6-7 test |
| D2 | AMLworld HI-Small | Kaggle, terms | time-sorted 60/20/20, 24 h graph windows |
| D3 | IEEE-CIS | Kaggle competition rules | `TransactionDT` below the 0.80 quantile train/dev, above test; expanding folds q in 0.60-0.90 |
| D4 | Elliptic++ | public repo, cloud-hosted files | time steps 1-34 train/dev, 35-49 test, unknown labels never used |
| D5 | DGraph-Fin | provider registration | temporal node-arrival split over labelled nodes, 70/15/15 (reconstruction) |
| SYN | synthetic fixture | none | exercises code only, never a benchmark |

Every validation segment is halved chronologically into selection and conformal-calibration parts. Ambiguities (D1 within-month order, D2 alignment, D3 validation fraction, D4 shock step, D5 cut points) are in `docs/DECISIONS.md` and printed by `data list`.

Tests use `tests/provider_files.py`, which writes tiny schema-faithful stand-ins; these are not the real data.
