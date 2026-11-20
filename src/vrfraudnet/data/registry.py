"""Dataset registry: where each benchmark comes from, which files and columns are expected, and how it is partitioned.

IMPORTANT about provenance: the file names, column names and label codes below are what the providers'
documentation describes (as recalled at build time). They are CHECKED, never assumed: `vrfraudnet data check`
compares them with your actual files and reports every difference. Raw data is never downloaded or redistributed
by this repository; gated providers need you to accept their terms and download manually.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..errors import VRFraudNetError


@dataclass(frozen=True)
class ExpectedFile:
    name: str
    required_columns: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    page: str
    access: str
    files: tuple[ExpectedFile, ...]
    label_column: str
    time_description: str
    stated_prevalence_pct: Optional[float]
    stated_rows: Optional[int]
    split_rule: str
    drop_columns: tuple[str, ...] = ()
    protected_columns: tuple[str, ...] = ()
    leakage_controls: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    graph: bool = False
    synthetic: bool = False


BAF_COLUMNS = ("fraud_bool", "month", "income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
               "customer_age", "days_since_request", "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h", "velocity_24h",
               "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "employment_status", "credit_risk_score", "email_is_free",
               "housing_status", "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards", "proposed_credit_limit",
               "foreign_request", "source", "session_length_in_minutes", "device_os", "keep_alive_session", "device_distinct_emails_8w",
               "device_fraud_count")

D1 = DatasetSpec(
    id="D1", name="BAF - Bank Account Fraud (Base variant), NeurIPS 2022",
    page="https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022",
    access="Kaggle account and acceptance of the dataset terms (CC BY-NC-SA 4.0 per the dataset page). Download Base.csv manually.",
    files=(ExpectedFile("Base.csv", BAF_COLUMNS, "1,000,000 rows; the Variant I-V files are NOT the primary benchmark (Variant III appears in Supp. S4a only)"),),
    label_column="fraud_bool", time_description="`month` (0-7). Row order within a month is the only finer ordering available.",
    stated_prevalence_pct=1.10, stated_rows=1_000_000,
    split_rule="months 0-5 train/development; validation = final 10% of that period (chronological); months 6-7 test. Validation halves: selection / conformal calibration.",
    drop_columns=("fraud_bool",), protected_columns=("customer_age",),
    leakage_controls=("drop fraud_bool from inputs", "exclude protected attribute from predictors (kept for subgroup evaluation only)", "preserve the train/test month boundary",
                      "target encoding fitted on training rows only, out-of-fold for the training rows themselves"),
    ambiguities=("Table 1/2 say 30 input features; the raw file has 30 features + label + month, so excluding the protected attribute leaves 29 (month is used for ordering, not as a predictor by default).",
                 "Order within a month is not time-stamped; 'final 10% chronologically' is taken as the last 10% of rows after a stable sort by month."))

D2 = DatasetSpec(
    id="D2", name="AMLworld HI-Small (IBM Transactions for Anti-Money Laundering)",
    page="https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    access="Kaggle account and acceptance of the dataset terms. Download HI-Small_Trans.csv (and optionally HI-Small_accounts.csv).",
    files=(ExpectedFile("HI-Small_Trans.csv", ("Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency", "Amount Paid",
                                                 "Payment Currency", "Payment Format", "Is Laundering"), "5,078,345 transaction rows"),
           ExpectedFile("HI-Small_Patterns.txt", (), "OPTIONAL. Contains laundering-pattern labels: a label ORACLE. It is never read as a feature.")),
    label_column="Is Laundering", time_description="`Timestamp` (about ten days).", stated_prevalence_pct=0.10, stated_rows=5_078_345,
    split_rule="time-sorted 60/20/20; 24-hour historical graph windows. Transactions are edges between accounts.",
    drop_columns=("Is Laundering",), leakage_controls=("exclude pattern-label oracle", "no future edges at prediction time", "per-bank label segregation"),
    ambiguities=("How account (node) embeddings map to transaction (edge) targets is not stated; see docs/DECISIONS.md D-11 (`alignment` policy).",
                 "'Per-bank partition / label segregation' is not defined operationally in the paper."), graph=True)

D3 = DatasetSpec(
    id="D3", name="IEEE-CIS Fraud Detection (Vesta)",
    page="https://www.kaggle.com/competitions/ieee-fraud-detection/data",
    access="Kaggle account and acceptance of the competition rules. Download train_transaction.csv and train_identity.csv. The competition's unlabeled test files are NOT used for scoring.",
    files=(ExpectedFile("train_transaction.csv", ("TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"), "590,540 rows, 394 columns incl. id and label"),
           ExpectedFile("train_identity.csv", ("TransactionID",), "41 columns incl. TransactionID")),
    label_column="isFraud", time_description="`TransactionDT` (seconds from an unknown origin; ~6 months).", stated_prevalence_pct=3.50, stated_rows=590_540,
    split_rule="TransactionDT < 0.80-quantile: train/development; >= 0.80: test. Expanding-window folds at q in (0.60, 0.70, 0.80, 0.85, 0.90).",
    drop_columns=("isFraud", "TransactionID", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D11", "D12", "D13", "D14"),
    leakage_controls=("TransactionDT used for ordering/partitioning only, never as a predictor", "UID-style shortcuts (card/address/D-column reconstructions) not built",
                      "drop D2-D9 and D11-D14", "identity table joined on TransactionID; missing identity handled within the record", "PCA of the 339 V-columns fitted on training rows only"),
    ambiguities=("The validation fraction inside the development period is not stated; 10% (as for D1) is used and recorded.",))

D4 = DatasetSpec(
    id="D4", name="Elliptic++ transaction graph", page="https://github.com/git-disl/EllipticPlusPlus",
    access="Public repository; the data files are linked from its README (cloud-hosted). Download the transaction features, classes and edge list files.",
    files=(ExpectedFile("txs_features.csv", (), "first column transaction id, second the time step, then 183 features"),
           ExpectedFile("txs_classes.csv", (), "class: 1 = illicit, 2 = licit, 3 = unknown (as documented by the provider)"),
           ExpectedFile("txs_edgelist.csv", (), "directed money-flow edges between transactions")),
    label_column="class", time_description="time step 1-49 (about two weeks each).", stated_prevalence_pct=2.23, stated_rows=203_769,
    split_rule="strict-inductive: time steps 1-34 train/development, 35-49 test. Unknown labels never supervise or score. Pre/post-shock windows within the test period.",
    drop_columns=(), leakage_controls=("drop the 89 pre-aggregated neighbour features, keep the 94 local features", "strict-inductive boundary", "no pseudo-labelling of unknown nodes"),
    ambiguities=("The paper does not state the shock time step. The dark-market shutdown in the Elliptic literature is at step 43; that is a RECONSTRUCTION, configurable, and recorded.",
                 "2.23% is the illicit share over ALL nodes (labelled and unlabelled); the share among labelled nodes is far higher.",
                 "Train-only scaling for a time step never seen in training: pooled training statistics are used (D-12)."), graph=True)

D5 = DatasetSpec(
    id="D5", name="DGraph-Fin (Finvolution)", page="https://dgraph.xinye.com/dataset",
    access="Registration on the provider's site and acceptance of its terms. Download dgraphfin.npz.",
    files=(ExpectedFile("dgraphfin.npz", ("x", "y", "edge_index", "edge_type", "edge_timestamp"), "3,700,550 nodes; 4,300,999 directed typed, timestamped edges"),),
    label_column="y", time_description="edge timestamps only; node arrival time is DERIVED as the earliest incident edge timestamp.",
    stated_prevalence_pct=1.30, stated_rows=3_700_550,
    split_rule="temporal node-arrival split over labelled nodes; background nodes (labels 2/3) give message-passing context only.",
    leakage_controls=("loss and evaluation restricted to labelled nodes", "background nodes never labelled or pseudo-labelled", "only edges before the prediction time are visible"),
    ambiguities=("Cut points of the node-arrival split are not stated; 70/15/15 by arrival-time quantile is used and recorded.",
                 "1.30% is a share over labelled nodes, not over all 3,700,550 nodes."), graph=True)

SYN = DatasetSpec(
    id="SYN", name="Synthetic fixture (obviously invented; NOT a benchmark)", page="(generated by vrfraudnet.data.synthetic)", access="none",
    files=(), label_column="label", time_description="synthetic seconds", stated_prevalence_pct=None, stated_rows=None,
    split_rule="60/20/20 chronological with validation halved for selection / conformal calibration", graph=True, synthetic=True,
    ambiguities=("Runs on SYN are exercises of the code, never reproductions of the paper.",))

REGISTRY: dict[str, DatasetSpec] = {s.id: s for s in (D1, D2, D3, D4, D5, SYN)}
PRIMARY = ("D1", "D2", "D3")


def get(dataset_id: str) -> DatasetSpec:
    try:
        return REGISTRY[dataset_id.upper()]
    except KeyError as e:
        raise VRFraudNetError(f"unknown dataset {dataset_id!r}; choose from {sorted(REGISTRY)}") from e
