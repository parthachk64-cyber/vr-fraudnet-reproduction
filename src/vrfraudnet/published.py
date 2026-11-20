"""Settings the paper states, with the section that states them.

These are *published* values. Anything fitted from data (thresholds re-selected on validation, mixer
parameters, conformal quantiles) lives in run artifacts and is recorded separately. Nothing here is
a result.
"""
from __future__ import annotations

PAPER = {
    "title": "Grounding language models with deterministic verifiers for fraud detection",
    "journal": "Discover Artificial Intelligence",
    "volume": 6,
    "article_number": 1403,
    "year": 2026,
    "doi": "10.1007/s44163-026-02355-0",
    "received": "2026-05-12",
    "accepted": "2026-09-20",
    "published_online": "2026-10-08",
}

AUTHORS = [
    "Md Sultanul Arefin Sourav", "Evha Rozario", "Md Ashiqul Islam", "Kaniz Sultana Chy", "Samia Akter",
    "Md Shahiduzzaman", "Mohammad Sazzad Hossain", "Maria Kabtia", "Partha Chakraborty", "Abdul Basit",
]

# Section 3.8.1: routing thresholds selected on validation, per dataset (low, high). Review band is inclusive.
PUBLISHED_ROUTING_THRESHOLDS = {
    "D1": (0.20, 0.80), "D2": (0.15, 0.85), "D3": (0.22, 0.78), "D4": (0.18, 0.82), "D5": (0.17, 0.83),
}

# Section 3.8.1 "Stage 1": LightGBM settings.
TRIAGE_PARAMS = {
    "n_estimators": 500, "max_depth": 8, "num_leaves": 64, "learning_rate": 0.05, "min_child_samples": 100,
    "colsample_bytree": 0.80, "subsample": 0.80, "subsample_freq": 1, "reg_alpha": 0.10, "reg_lambda": 1.00,
    "early_stopping_rounds": 50,
}

# Section 3.8.1 "Stage 0" and Section 3.1.
ENCODER_PARAMS = {
    "n_filters": 4, "time_embedding_dim": 32, "output_dim": 64, "lr": 1e-3, "weight_decay": 1e-4,
    "max_epochs": 100, "patience": 15, "lr_decay": 0.5, "lr_decay_patience": 5, "min_lr": 1e-5,
    "grad_clip": 1.0, "node_batch_size": 2048, "fanouts": (25, 10),
}

# Sections 3.3.1 and 3.8.1. Section 3.3.1 prints the learning rate as "2 x 10^4" (a typesetting error);
# Section 3.8.1 prints 0.0002. This repository uses 2e-4 and records the conflict in docs/DECISIONS.md.
LORA_PARAMS = {
    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "epochs": 3, "lr": 2e-4, "weight_decay": 0.01,
    "warmup_ratio": 0.05, "max_input_tokens": 1024, "max_output_tokens": 384, "effective_batch_size": 32,
    "dtype": "bfloat16", "decoding": "greedy", "max_claims": 8,
}

RETRIEVAL_PARAMS = {"embedder": "sentence-transformers/all-MiniLM-L6-v2", "top_k": 4, "metric": "cosine"}

# Section 3.6.4: final weights, selected on validation data in the paper.
OBJECTIVE_WEIGHTS = {"cost_sensitive": 0.50, "adversarial": 0.50, "counterfactual": 0.25, "adversarial_budget": 2,
                     "adversarial_fraction": 0.25}

CONFORMAL = {"alpha_primary": 0.05, "alpha_sensitivity": 0.10}

SEEDS = (42, 123, 456, 789, 2024, 3141, 2718, 1001, 1729, 4096)

LATENCY_PROTOCOL = {"batch_size": 1, "warmup": 1000, "timed": 10000, "gpu_sync": True,
                    "hardware": "AMD EPYC 7543 32-core, 256 GB RAM, 1x NVIDIA A100 80 GB",
                    "p99_budget_ms": 200.0}

REPORTED_SOFTWARE = {
    "python": "3.11.7", "cuda": "12.1", "cudnn": "8.9", "torch": "2.2.1", "torch_geometric": "2.5.2", "lightgbm": "4.3.0",
    "xgboost": "2.0.3", "scikit-learn": "1.4.2", "transformers": "4.40.1", "peft": "0.10.0",
    "sentence-transformers": "2.7.0", "faiss-gpu": "1.8.0", "os": "Ubuntu 22.04 LTS",
}

# Section 3.8.1 / Table 1: how the chronological validation segment is divided.
VALIDATION_HALVES = ("selection", "conformal_calibration")

BASELINES = ("logistic_regression", "isolation_forest", "mlp", "cnn1d", "lstm", "random_forest", "xgboost", "lightgbm",
             "tabtransformer")

ABLATIONS = {
    "A1": "w/o Verifier V", "A2": "w/o Triage T_theta", "A3": "w/o Schema", "A4": "w/o Adversarial training",
    "A5": "w/o Cost-sensitive loss", "A6": "w/o Large LLM", "A7": "w/o Learned mixer", "A8": "lambda sensitivity +-50%",
    "A9": "w/o Counterfactual loss",
}

TEMPORAL_Q = (0.60, 0.70, 0.80, 0.85, 0.90)  # Table 8d cut quantiles; fold 3 (0.80) is the primary D3 holdout
