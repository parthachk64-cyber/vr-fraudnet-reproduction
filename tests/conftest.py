"""Shared fixtures. Everything here is invented: synthetic IDs, no real people, accounts or data."""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")        # transformers would otherwise import TensorFlow if it is installed (Keras 3 clash)
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest

from vrfraudnet.verifier.evidence import EvidenceSnapshot, GraphEvidence

T0 = 1_000_000.0


@pytest.fixture
def evidence() -> EvidenceSnapshot:
    """A synthetic wire transfer (loosely modelled on the paper's Table 4 case E1)."""
    graph = GraphEvidence(
        edges=(("ACC_SYN_001", "ACC_SYN_077", T0 - 600), ("ACC_SYN_077", "ACC_SYN_900", T0 - 300)),
        complete=True, prediction_time=T0)
    return EvidenceSnapshot(
        transaction_id="TX_SYN_0001", prediction_time=T0,
        fields={"amount": 9800.0, "recent_transfer_count": 7, "payment_type": "wire", "account_age_days": 3},
        event_times={"last_transfer_time": T0 - 1200.0, "account_opened_time": T0 - 3 * 86400.0},
        entities={"origin": "ACC_SYN_001", "beneficiary": "ACC_SYN_900"},
        evidence_index={"E_AMT": "amount", "E_CNT": "recent_transfer_count", "E_PAY": "payment_type", "E_AGE": "account_age_days",
                        "E_LAST": "last_transfer_time", "E_OPEN": "account_opened_time", "E_ORG": "origin", "E_BEN": "beneficiary"},
        graph=graph)


def claim(type_, field, op, value, eid):
    return {"type": type_, "field": field, "op": op, "value": value, "evidence_id": eid}


@pytest.fixture
def good_rationale() -> dict:
    return {
        "verdict": "fraud", "probability": 0.91,
        "claims": [
            claim("numeric_comparison", "amount", ">=", 5000, "E_AMT"),
            claim("set_membership", "payment_type", "in", ["wire", "ach"], "E_PAY"),
            claim("temporal_relation", "last_transfer_time", "within", 3600, "E_LAST"),
            claim("graph_path", "origin", "path_exists", ["ACC_SYN_900", 2], "E_ORG"),
            claim("entity_match", "beneficiary", "==", "ACC_SYN_900", "E_BEN"),
        ],
        "evidence_refs": ["E_AMT", "E_PAY", "E_LAST", "E_ORG", "E_BEN"],
    }
