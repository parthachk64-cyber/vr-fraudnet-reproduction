"""Regenerates references/reported/*.json from rows copied verbatim from the published article PDF
(pages noted per table) and parsed from the publisher's Supplementary Material 1 DOCX.

    python tools/transcribe_reported.py --supplement path/to/44163_2026_2355_MOESM1_ESM.docx

These files are REFERENCE data: the paper's published numbers, kept separate from anything this
repository computes. No inference or table-building code reads them as input; they are used only by the
audit, the status ledger and the "reported vs reproduced" comparison.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DOC = "Sourav et al., Discover Artificial Intelligence 6:1403 (2026), doi:10.1007/s44163-026-02355-0"
OUT = Path(__file__).resolve().parents[1] / "references" / "reported"

PM = "±"
NUM = r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?"


def norm(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").replace("–", "-").replace(" ", " ").strip()


def pair(tok: str):
    """'0.3245 ± 0.0089' -> {'mean': .., 'sd': ..}; a bare number -> {'mean': ..}"""
    tok = norm(tok)
    if PM in tok:
        m, s = [t.strip() for t in tok.split(PM)]
        return {"mean": float(m), "sd": float(s)}
    return {"mean": float(tok)}


def triple(tok: str):
    """'0.5231 / 0.5205 / 0.5158' -> three values for the -50% / +50% / ... sensitivity cells (order as printed)."""
    return [float(norm(t)) for t in tok.split("/")]


def table5_rows(block: str, metrics=("AUPRC", "ROC-AUC", "F1", "Recall@top-1%", "MCC")):
    rows = []
    for line in [l for l in block.strip().splitlines() if l.strip()]:
        parts = re.findall(rf"{NUM}\s*{PM}\s*{NUM}", norm(line))
        name = re.split(rf"\s{NUM}\s*{PM}", norm(line))[0].strip()
        assert len(parts) == 5, line
        rows.append({"method": name, **{m: pair(p) for m, p in zip(metrics, parts)}})
    return rows


T5 = {
    "5a": ("D1 BAF Base (1.10% fraud prevalence)", 21, """
Logistic regression 0.3245 ± 0.0089 0.8312 ± 0.0048 0.3567 ± 0.0095 0.5245 ± 0.0142 0.3478 ± 0.0102
Isolation forest 0.2987 ± 0.0103 0.8156 ± 0.0056 0.3237 ± 0.0108 0.4912 ± 0.0162 0.3198 ± 0.0118
MLP 0.4123 ± 0.0078 0.8723 ± 0.0042 0.4378 ± 0.0083 0.6098 ± 0.0123 0.4287 ± 0.0089
1D-CNN 0.4287 ± 0.0083 0.8784 ± 0.0044 0.4467 ± 0.0089 0.6157 ± 0.0128 0.4356 ± 0.0094
LSTM 0.4456 ± 0.0080 0.8845 ± 0.0042 0.4584 ± 0.0085 0.6298 ± 0.0124 0.4498 ± 0.0091
Random forest 0.4534 ± 0.0072 0.8867 ± 0.0038 0.4623 ± 0.0079 0.6342 ± 0.0117 0.4534 ± 0.0085
XGBoost 0.4823 ± 0.0067 0.8967 ± 0.0035 0.4845 ± 0.0073 0.6587 ± 0.0107 0.4734 ± 0.0079
LightGBM 0.4912 ± 0.0064 0.9012 ± 0.0034 0.4923 ± 0.0070 0.6651 ± 0.0103 0.4823 ± 0.0076
TabTransformer 0.5012 ± 0.0061 0.9078 ± 0.0032 0.4998 ± 0.0067 0.6789 ± 0.0098 0.4912 ± 0.0072
VR-FraudNet 0.5247 ± 0.0058 0.9156 ± 0.0030 0.5181 ± 0.0063 0.6912 ± 0.0092 0.5063 ± 0.0069"""),
    "5b": ("D2 AMLworld HI-Small (0.10% fraud prevalence)", 21, """
Logistic regression 0.2645 ± 0.0098 0.9234 ± 0.0033 0.3287 ± 0.0102 0.4567 ± 0.0148 0.2987 ± 0.0103
Isolation forest 0.2398 ± 0.0108 0.9123 ± 0.0038 0.2987 ± 0.0112 0.4234 ± 0.0162 0.2645 ± 0.0117
MLP 0.3287 ± 0.0089 0.9412 ± 0.0028 0.4012 ± 0.0093 0.5234 ± 0.0132 0.3645 ± 0.0091
1D-CNN 0.3478 ± 0.0092 0.9456 ± 0.0027 0.4187 ± 0.0095 0.5398 ± 0.0134 0.3812 ± 0.0093
LSTM 0.3823 ± 0.0086 0.9523 ± 0.0024 0.4423 ± 0.0090 0.5723 ± 0.0124 0.4078 ± 0.0085
Random forest 0.4012 ± 0.0083 0.9587 ± 0.0023 0.4612 ± 0.0086 0.5934 ± 0.0118 0.4287 ± 0.0080
XGBoost 0.4523 ± 0.0078 0.9645 ± 0.0021 0.4934 ± 0.0080 0.6234 ± 0.0110 0.4567 ± 0.0070
LightGBM 0.4712 ± 0.0075 0.9678 ± 0.0020 0.5087 ± 0.0077 0.6398 ± 0.0107 0.4712 ± 0.0070
TabTransformer 0.5137 ± 0.0069 0.9745 ± 0.0018 0.5612 ± 0.0072 0.6912 ± 0.0098 0.5198 ± 0.0066
VR-FraudNet 0.5824 ± 0.0065 0.9812 ± 0.0016 0.6087 ± 0.0061 0.7384 ± 0.0092 0.5612 ± 0.0062"""),
    "5c": ("D3 IEEE-CIS, fixed chronological TransactionDT holdout (3.50% fraud prevalence)", 21, """
Logistic regression 0.5234 ± 0.0073 0.8412 ± 0.0040 0.5231 ± 0.0078 0.4567 ± 0.0098 0.5034 ± 0.0083
Isolation forest 0.4823 ± 0.0084 0.8234 ± 0.0046 0.4923 ± 0.0089 0.4234 ± 0.0112 0.4712 ± 0.0094
MLP 0.6234 ± 0.0066 0.8823 ± 0.0034 0.5934 ± 0.0070 0.5234 ± 0.0090 0.5687 ± 0.0075
1D-CNN 0.6398 ± 0.0064 0.8867 ± 0.0033 0.6087 ± 0.0068 0.5345 ± 0.0088 0.5823 ± 0.0072
LSTM 0.6534 ± 0.0061 0.8923 ± 0.0031 0.6198 ± 0.0066 0.5478 ± 0.0085 0.5934 ± 0.0070
Random forest 0.6712 ± 0.0058 0.8989 ± 0.0030 0.6312 ± 0.0063 0.5612 ± 0.0082 0.6087 ± 0.0068
XGBoost 0.6987 ± 0.0055 0.9089 ± 0.0028 0.6487 ± 0.0061 0.5812 ± 0.0078 0.6287 ± 0.0065
LightGBM 0.7087 ± 0.0053 0.9134 ± 0.0027 0.6587 ± 0.0059 0.5891 ± 0.0076 0.6378 ± 0.0063
TabTransformer 0.7234 ± 0.0051 0.9201 ± 0.0026 0.6712 ± 0.0057 0.6034 ± 0.0074 0.6489 ± 0.0061
VR-FraudNet 0.7456 ± 0.0049 0.9312 ± 0.0025 0.6878 ± 0.0054 0.6234 ± 0.0071 0.6612 ± 0.0060"""),
}

T6 = """
Logistic regression +24.68 – 9.31e−10 1.00 8.38e−9
Isolation forest +27.73 465 9.31e−10 1.00 8.38e−9
MLP +16.28 465 9.31e−10 0.96 8.38e−9
1D-CNN +14.55 465 9.31e−10 0.94 8.38e−9
LSTM +12.38 465 9.31e−10 0.91 8.38e−9
Random Forest +10.90 465 9.31e−10 0.88 8.38e−9
XGBoost +7.31 462 1.86e−9 0.83 8.38e−9
LightGBM +6.05 458 4.66e−9 0.79 9.32e−9
TabTransformer +3.81 442 1.40e−7 0.66 1.40e−7"""

T7 = {
    "7a": ("Ablation and lambda sensitivity on D1 BAF Base", """
A6|w/o Large LLM|0.4969 ± 0.0078|0.9036 ± 0.0040|0.4944 ± 0.0084|0.6647 ± 0.0118|0.4833 ± 0.0089|−2.78
A1|w/o Verifier V|0.5029 ± 0.0073|0.9061 ± 0.0038|0.4990 ± 0.0079|0.6707 ± 0.0112|0.4885 ± 0.0084|−2.18
A3|w/o Schema|0.5080 ± 0.0071|0.9084 ± 0.0036|0.5045 ± 0.0076|0.6765 ± 0.0108|0.4929 ± 0.0081|−1.67
A2|w/o Triage Tθ|0.5149 ± 0.0070|0.9111 ± 0.0035|0.5101 ± 0.0074|0.6823 ± 0.0106|0.4982 ± 0.0079|−0.98
A9|w/o Counterfactual loss|0.5158 ± 0.0067|0.9118 ± 0.0034|0.5108 ± 0.0072|0.6831 ± 0.0104|0.4989 ± 0.0078|−0.89
A4|w/o Adversarial training|0.5189 ± 0.0064|0.9128 ± 0.0033|0.5142 ± 0.0069|0.6864 ± 0.0099|0.5021 ± 0.0075|−0.58
A7|w/o Learned mixer|0.5199 ± 0.0063|0.9134 ± 0.0032|0.5152 ± 0.0068|0.6873 ± 0.0098|0.5028 ± 0.0074|−0.48
A5|w/o Cost-sensitive loss|0.5213 ± 0.0061|0.9141 ± 0.0031|0.5161 ± 0.0066|0.6889 ± 0.0096|0.5042 ± 0.0072|−0.34
A8|lambda sensitivity ±50%|0.5231 / 0.5205 / 0.5158|0.9148 / 0.9136 / 0.9122|0.5172 / 0.5157 / 0.5126|0.6898 / 0.6872 / 0.6843|0.5049 / 0.5017 / 0.4998|−0.16 / −0.42 / −0.89
Full|Full VR-FraudNet|0.5247 ± 0.0058|0.9156 ± 0.0030|0.5181 ± 0.0063|0.6912 ± 0.0092|0.5063 ± 0.0069|0.00"""),
    "7b": ("Ablation and lambda sensitivity on D2 AMLworld HI-Small", """
A6|w/o Large LLM|0.5472 ± 0.0086|0.9692 ± 0.0021|0.5728 ± 0.0080|0.7012 ± 0.0117|0.5267 ± 0.0079|−3.52
A1|w/o Verifier V|0.5546 ± 0.0080|0.9737 ± 0.0019|0.5798 ± 0.0075|0.7089 ± 0.0111|0.5334 ± 0.0074|−2.78
A3|w/o Schema|0.5612 ± 0.0078|0.9756 ± 0.0018|0.5862 ± 0.0073|0.7167 ± 0.0108|0.5398 ± 0.0072|−2.12
A2|w/o Triage Tθ|0.5690 ± 0.0074|0.9778 ± 0.0017|0.5934 ± 0.0070|0.7245 ± 0.0103|0.5478 ± 0.0069|−1.34
A9|w/o Counterfactual loss|0.5719 ± 0.0072|0.9785 ± 0.0017|0.5961 ± 0.0068|0.7278 ± 0.0101|0.5512 ± 0.0067|−1.05
A4|w/o Adversarial training|0.5752 ± 0.0070|0.9794 ± 0.0016|0.5994 ± 0.0066|0.7312 ± 0.0098|0.5547 ± 0.0066|−0.72
A7|w/o Learned mixer|0.5766 ± 0.0069|0.9798 ± 0.0016|0.6010 ± 0.0065|0.7331 ± 0.0097|0.5562 ± 0.0065|−0.58
A5|w/o Cost-sensitive loss|0.5783 ± 0.0067|0.9803 ± 0.0016|0.6034 ± 0.0064|0.7351 ± 0.0095|0.5582 ± 0.0064|−0.41
A8|lambda sensitivity ±50%|0.5813 / 0.5788 / 0.5762|0.9809 / 0.9804 / 0.9798|0.6070 / 0.6048 / 0.6029|0.7372 / 0.7348 / 0.7321|0.5602 / 0.5580 / 0.5550|−0.11 / −0.36 / −0.62
Full|Full VR-FraudNet|0.5824 ± 0.0065|0.9812 ± 0.0016|0.6087 ± 0.0061|0.7384 ± 0.0092|0.5612 ± 0.0062|0.00"""),
}

T8A = """
LightGBM|0.4912 ± 0.0064|0.4534 ± 0.0073|-3.78|0.4287 ± 0.0079|-6.25
TabTransformer|0.5012 ± 0.0061|0.4687 ± 0.0069|-3.25|0.4456 ± 0.0075|-5.56
A1 w/o Verifier V|0.5029 ± 0.0073|0.4756 ± 0.0078|-2.73|0.4642 ± 0.0084|-3.87
A4 w/o Adversarial training|0.5189 ± 0.0064|0.4934 ± 0.0070|-2.55|0.4756 ± 0.0076|-4.33
A9 w/o Counterfactual loss|0.5158 ± 0.0067|0.4845 ± 0.0072|-3.13|0.4663 ± 0.0078|-4.95
Full VR-FraudNet|0.5247 ± 0.0058|0.5089 ± 0.0061|-1.58|0.4945 ± 0.0066|-3.02"""

T8B = """
LightGBM|0.6712 ± 0.0072|0.6823 ± 0.0068|0.7087 ± 0.0053|0.7012 ± 0.0061|0.6967 ± 0.0065|0.6920|-1.20
TabTransformer|0.6989 ± 0.0064|0.7123 ± 0.0058|0.7234 ± 0.0051|0.7187 ± 0.0055|0.7156 ± 0.0057|0.7138|-0.78
A1 w/o Verifier V|0.7012 ± 0.0070|0.7156 ± 0.0067|0.7298 ± 0.0061|0.7234 ± 0.0064|0.7178 ± 0.0066|0.7176|-1.20
A4 w/o Adversarial training|0.7156 ± 0.0066|0.7287 ± 0.0061|0.7398 ± 0.0055|0.7345 ± 0.0058|0.7298 ± 0.0060|0.7297|-1.00
A9 w/o Counterfactual loss|0.7089 ± 0.0068|0.7234 ± 0.0064|0.7367 ± 0.0058|0.7298 ± 0.0061|0.7234 ± 0.0064|0.7244|-1.33
Full VR-FraudNet|0.7234 ± 0.0058|0.7345 ± 0.0054|0.7456 ± 0.0049|0.7423 ± 0.0051|0.7398 ± 0.0053|0.7371|-0.58"""

T8C = """
LightGBM|0.5634 ± 0.0078|0.5234 ± 0.0088|-4.00|0.3534 ± 0.0142|-21.00|-17.00
TabTransformer|0.5612 ± 0.0089|0.5398 ± 0.0083|-2.14|0.3987 ± 0.0134|-16.25|-14.11
A1 w/o Verifier V|0.6712 ± 0.0084|0.6298 ± 0.0080|-4.14|0.5567 ± 0.0118|-11.45|-7.31
A4 w/o Adversarial training|0.6612 ± 0.0086|0.6234 ± 0.0078|-3.78|0.5398 ± 0.0124|-12.14|-8.36
A9 w/o Counterfactual loss|0.6534 ± 0.0083|0.6087 ± 0.0083|-4.47|0.5034 ± 0.0128|-15.00|-10.53
Full VR-FraudNet|0.6823 ± 0.0072|0.6534 ± 0.0070|-2.89|0.6087 ± 0.0094|-7.36|-4.47"""

T8D = [(1, 0.60, 354324, 236216), (2, 0.70, 413378, 177162), (3, 0.80, 472432, 118108),
       (4, 0.85, 501959, 88581), (5, 0.90, 531486, 59054)]

T9 = {
    "9a": ("budget B = 1", """
LightGBM|0.3245 ± 0.0148|0.1823 ± 0.0124|0.4567 ± 0.0162|0.2089 ± 0.0132|0.2931
TabTransformer|0.2812 ± 0.0136|0.1534 ± 0.0113|0.4234 ± 0.0152|0.1789 ± 0.0124|0.2592
A1 w/o Verifier V|0.1812 ± 0.0114|0.1187 ± 0.0098|0.3387 ± 0.0138|0.1378 ± 0.0103|0.1941
A4 w/o Adversarial training|0.3089 ± 0.0142|0.1923 ± 0.0126|0.4498 ± 0.0156|0.2760 ± 0.0128|0.3068
Full VR-FraudNet|0.0689 ± 0.0058|0.0487 ± 0.0049|0.0892 ± 0.0064|0.0598 ± 0.0053|0.0667"""),
    "9b": ("budget B = 2", """
LightGBM|0.4523 ± 0.0162|0.2789 ± 0.0142|0.5823 ± 0.0178|0.3445 ± 0.0148|0.4145
TabTransformer|0.4123 ± 0.0156|0.2434 ± 0.0136|0.5398 ± 0.0172|0.2812 ± 0.0142|0.3692
A1 w/o Verifier V|0.2812 ± 0.0136|0.2089 ± 0.0124|0.4623 ± 0.0162|0.2434 ± 0.0132|0.2990
A4 w/o Adversarial training|0.4187 ± 0.0156|0.3098 ± 0.0142|0.5634 ± 0.0172|0.3398 ± 0.0148|0.4079
Full VR-FraudNet|0.1089 ± 0.0072|0.0789 ± 0.0062|0.1287 ± 0.0078|0.0898 ± 0.0067|0.1016"""),
    "9c": ("budget B = 4", """
LightGBM|0.6234 ± 0.0184|0.4523 ± 0.0167|0.7234 ± 0.0198|0.5039 ± 0.0173|0.5758
TabTransformer|0.5812 ± 0.0178|0.4123 ± 0.0162|0.6812 ± 0.0192|0.4734 ± 0.0167|0.5370
A1 w/o Verifier V|0.4187 ± 0.0162|0.3623 ± 0.0152|0.6234 ± 0.0184|0.4089 ± 0.0162|0.4533
A4 w/o Adversarial training|0.5812 ± 0.0178|0.4612 ± 0.0167|0.6912 ± 0.0192|0.4923 ± 0.0173|0.5565
Full VR-FraudNet|0.1812 ± 0.0098|0.1289 ± 0.0084|0.2098 ± 0.0108|0.1487 ± 0.0094|0.1672"""),
}

T1 = {  # Table 1 (page 7)
    "D1": {"name": "BAF - Bank Account Fraud (Base variant)", "modality": "tabular", "size": "1,000,000 instances", "input_features": 30,
           "fraud_rate_pct": 1.10, "time_span": "8 months (month 0-7)", "split": "train m=0-5; test m=6-7; validation = last 10% of train, chronological"},
    "D2": {"name": "AMLworld HI-Small", "modality": "graph", "size": "515,088 accounts (nodes); 5,078,345 transactions (edges)",
           "input_features": "9 graph attrs + entity metadata", "fraud_rate_pct": 0.10, "time_span": "10 days", "split": "temporal 60/20/20 by transaction timestamp"},
    "D3": {"name": "IEEE-CIS Fraud Detection (Vesta)", "modality": "tabular + time", "size": "590,540 transactions",
           "input_features": "394 transaction columns + 41 identity columns joined on TransactionID", "fraud_rate_pct": 3.50, "time_span": "~6 months",
           "split": "TransactionDT < 0.8-quantile train; >= 0.8 test; 5-fold time-series CV with expanding window"},
    "D4": {"name": "Elliptic++ transaction graph", "modality": "graph + time", "size": "203,769 transaction nodes; 234,355 money-flow edges",
           "input_features": 183, "fraud_rate_pct": 2.23, "time_span": "49 timesteps (~2 weeks each)", "split": "strict-inductive train t=1-34; test t=35-49; pre/post-shock evaluation"},
    "D5": {"name": "DGraph-Fin (Finvolution)", "modality": "graph", "size": "3,700,550 nodes; 4,300,999 directed edges",
           "input_features": "17 node features + 11 edge types + edge timestamps", "fraud_rate_pct": 1.30, "time_span": "full Finvolution P2P platform history",
           "split": "temporal node-arrival split; background nodes retained only for message passing"},
}

TEXT_CLAIMS = [  # numbers stated in prose, with page
    ("abstract_d1_auprc", "VR-FraudNet D1 AUPRC 0.5247 vs TabTransformer 0.5012", 1, [0.5247, 0.5012]),
    ("abstract_d2_auprc", "D2 AUPRC 0.5824 vs 0.5137", 1, [0.5824, 0.5137]),
    ("abstract_d3_auprc", "D3 AUPRC 0.7456 vs 0.7234", 1, [0.7456, 0.7234]),
    ("d4_post_shock", "D4 post-shock AUPRC 0.6823 -> 0.6087, absolute reduction 7.36 pp, exceeds the 5 pp target", 1, [0.6823, 0.6087, 7.36]),
    ("evasion_means", "mean evasion rates 0.0667 / 0.1016 / 0.1672 for budgets 1 / 2 / 4", 1, [0.0667, 0.1016, 0.1672]),
    ("p99_latency_ms", "p99 latency 10.23 ms common path, 175.23 ms escalation path", 1, [10.23, 175.23]),
    ("latency_protocol", "batch size 1; 1,000 warm-up; 10,000 timed; GPU synchronised; A100 80 GB", 23, [1, 1000, 10000]),
    ("conformal_alpha", "primary alpha 0.05, sensitivity alpha 0.10", 22, [0.05, 0.10]),
    ("routing_thresholds", "D1 0.20/0.80, D2 0.15/0.85, D3 0.22/0.78, D4 0.18/0.82, D5 0.17/0.83", 22, []),
    ("cost_weights", "cost-sensitive 0.50, adversarial 0.50, counterfactual 0.25", 19, [0.50, 0.50, 0.25]),
]


def src(page, table, kind="main_text_table"):
    return {"document": DOC, "page": page, "table": table, "kind": kind, "transcribed_by": "tools/transcribe_reported.py"}


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build_main():
    dump("table1.json", {"source": src(7, "Table 1"), "datasets": T1,
                         "qualifiers": ["Fraud rate is the dataset-level rate stated by the paper; the evaluated partition's own prevalence is a separate quantity."]})
    t5 = {"source": src(21, "Table 5a-c"), "metric_units": "AUPRC/ROC-AUC/F1/Recall@top-1%/MCC are fractions in [0,1]; +- is the sample SD over ten seeds (Sec. 3.8.3)",
          "panels": {}}
    for k, (title, page, block) in T5.items():
        t5["panels"][k] = {"title": title, "page": page, "rows": table5_rows(block)}
    dump("table5.json", t5)

    rows = []
    for line in T6.strip().splitlines():
        t = norm(line).split()
        name = " ".join(t[:-5] if t[-5].startswith("+") or t[-5].startswith("-") else t[:-5])
        # last five tokens: delta, V, raw p, effect, holm
        *nm, d, v, p, e, h = t
        rows.append({"baseline": " ".join(nm), "mean_delta_auprc_pp": float(d), "wilcoxon_V": None if v == "-" else int(v),
                     "raw_p_one_sided": float(p), "effect_size": float(e), "holm_adjusted_p": float(h)})
    dump("table6.json", {"source": src(24, "Table 6"), "n_pairs": 30, "V_max": 465,
                         "qualifiers": ["Logistic regression has no V printed ('-')", "Effect-size convention is not named in the paper"], "rows": rows})

    for k, (title, block) in T7.items():
        rr = []
        for line in block.strip().splitlines():
            idv, var, *cells, delta = line.split("|")
            row = {"id": idv, "variant": var}
            for m, c in zip(("AUPRC", "ROC-AUC", "F1", "Recall@top-1%", "MCC"), cells):
                row[m] = triple(c) if "/" in c else pair(c)
            row["delta_auprc_pp"] = triple(norm(delta)) if "/" in delta else float(norm(delta))
            rr.append(row)
        dump(f"table{k}.json", {"source": src(28, f"Table {k}"), "title": title, "rows": rr,
                                 "qualifiers": ["A8 cells are three values as printed (the -50% / +50% / ... ordering is not labelled in the paper)"]})

    def rows8(block, cols):
        out = []
        for line in block.strip().splitlines():
            parts = line.split("|")
            row = {"method": parts[0]}
            for c, v in zip(cols, parts[1:]):
                row[c] = pair(v) if PM in v else float(norm(v))
            out.append(row)
        return out

    t8 = {"source": src(31, "Table 8a-d"), "panels": {
        "8a": {"title": "D1 BAF Base controlled month drift",
               "qualifiers": ["The 'Train AUPRC' column equals the Table 5a TEST AUPRC for the same models; its reference is ambiguous."],
               "rows": rows8(T8A, ["train_auprc", "test_m6", "delta_m6_pp", "test_m7", "delta_m7_pp"])},
        "8b": {"title": "D3 five-fold expanding window (Fold 3 = primary holdout)",
               "rows": rows8(T8B, ["fold1", "fold2", "fold3", "fold4", "fold5", "mean_across_folds_descriptive", "late_delta_f5_minus_f3_pp"])},
        "8c": {"title": "D4 Elliptic++ pre/post shock", "rows": rows8(T8C, ["train_auprc", "pre_shock", "delta_pre_pp", "post_shock", "delta_post_pp", "shock_impact_pp"])},
        "8d": {"title": "D3 fold definitions", "rows": [{"fold": f, "cut_quantile": q, "nominal_train_n": a, "nominal_test_n": b} for f, q, a, b in T8D]}}}
    dump("table8.json", t8)

    t9 = {"source": src(33, "Table 9a-c"),
          "qualifiers": ["The dataset(s) behind Table 9 are not named in the paper.", "Columns: split, delay, mule, feature-perturbation (families e1, e2, e3, e5); e4 (injection) is excluded for structured models."],
          "panels": {}}
    for k, (title, block) in T9.items():
        t9["panels"][k] = {"title": title, "rows": rows8(block, ["split", "delay", "mule", "feat", "mean"])}
    dump("table9.json", t9)
    dump("text_claims.json", {"source": {"document": DOC}, "claims": [
        {"id": i, "text": t, "page": p, "values": v} for i, t, p, v in TEXT_CLAIMS]})


# ---------------------------------------------------------------- supplement
CELL = re.compile(rf"^\$?\s*({NUM.replace('.', r'[.]') if False else NUM}|[\d,]+(?:\.\d+)?)(?:\s*{PM}\s*({NUM}))?$")


def cell(raw: str):
    t = norm(raw)
    m = re.match(rf"^(\$?)\s*(-?[\d,]*\.?\d+)\s*(?:{PM}\s*(\d*\.?\d+))?\s*$", t)
    if not m:
        return t
    unit = "USD" if m.group(1) == "$" else None
    val = float(m.group(2).replace(",", ""))
    out = {"value": val}
    if m.group(3) is not None:
        out["sd"] = float(m.group(3))
    if unit:
        out["unit"] = unit
    return out


def build_supplement(docx_path: Path):
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(docx_path))
    items = []
    for ch in d.element.body.iterchildren():
        if ch.tag.endswith("}p"):
            t = Paragraph(ch, d).text.strip()
            if t:
                items.append(("P", t))
        elif ch.tag.endswith("}tbl"):
            items.append(("T", [[c.text.strip() for c in r.cells] for r in Table(ch, d).rows]))
    tables, title = {}, None
    for kind, val in items:
        if kind == "P":
            m = re.match(r"Supplementary Table (S\d)(\(\w\))?\s+(.*)", val)
            title = (m.group(1) + (m.group(2) or "").strip("()"), m.group(3)) if m else title
        else:
            tid, ttl = title
            hdr, body = val[0], val[1:]
            rows = []
            for r_i, r in enumerate(body, start=1):
                rows.append({"_row": r_i, **{h: cell(c) for h, c in zip(hdr, r)}})
            tables[tid] = {"title": ttl, "columns": hdr, "rows": rows,
                           "source": {"document": "Supplementary Material 1 (44163_2026_2355_MOESM1_ESM.docx)", "table": f"Supplementary Table {tid}",
                                      "kind": "supplement_docx", "row_index_base": 1}}
    for tid, t in tables.items():
        dump(f"supplement_{tid.lower()}.json", t)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--supplement", type=Path, default=None)
    a = ap.parse_args()
    build_main()
    if a.supplement:
        t = build_supplement(a.supplement)
        print("supplement tables:", ", ".join(sorted(t)))
    print("wrote", OUT)
