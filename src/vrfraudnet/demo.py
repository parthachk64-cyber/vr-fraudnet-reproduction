"""The five-minute CPU walkthrough: follow transactions through VR-FraudNet and see exactly why a rationale passed or failed.

Runs with the CORE install only (numpy, pandas, scipy, scikit-learn): no login, no network, no GPU, no benchmark data, no gated weights.
It is a demonstration on an invented dataset. Two parts are stand-ins and are labelled as such wherever they appear:
  * the triage model is a logistic regression (the paper uses LightGBM; that needs the `trees` extra);
  * the rationale provider is a rule-based FIXTURE (the paper uses a LoRA-adapted Llama; that needs licensed weights).
Everything downstream of those two (routing, retrieval rules, the verifier, the mixer, conformal sets, serialisation) is the real implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .calibration.conformal import SplitConformal
from .data import splits as SP
from .data.evidence import EvidenceBuilder
from .data.synthetic import make_synthetic
from .models.mixer import VerifiedMixer
from .models.pipeline import Case, VRFraudNetPipeline
from .models.rationale import RuleFixtureProvider
from .types import Route, TransactionResult

BANNER = "SYNTHETIC FIXTURE DEMO: invented data, stand-in triage model, rule-based rationale provider. Not a reproduction of any published result."


def _features(f):
    import numpy as np

    return np.column_stack([np.log1p(f["amount"]), np.log1p(f["account_age_days"].clip(lower=0)), (f["payment_type"] == "wire").astype(float), f["recent_transfer_count"].clip(upper=10)])


def explain(result: TransactionResult) -> str:
    lines = [f"  transaction   {result.transaction_id}", f"  route         {result.route.value}   (automated: {result.automated})",
             f"  triage p      {result.triage_probability:.3f}", f"  verifier      {result.verifier.value}",
             f"  risk score    {result.risk_score:.3f}   [{result.score_provenance}]"]
    if result.rationale_probability_used is not None:
        lines.append(f"  rationale p   {result.rationale_probability_used:.3f}   (used: the verifier accepted every claim)")
    elif result.verifier.value == "rejected":
        lines.append("  rationale p   NOT USED (a rejected rationale never influences the score)")
    if result.conformal_set is not None:
        lines.append(f"  conformal set {sorted(result.conformal_set)}   (0 = legitimate, 1 = fraud; marginal 90% coverage under exchangeability)")
    if result.verification is not None:
        lines.append("  claim audit:")
        for c in result.verification.claims:
            lines.append(f"    [{'PASS' if c.passed else 'FAIL'}] {c.claim_type:<19} {c.status:<12} {c.reason}")
        for r in result.verification.rule_failures:
            lines.append(f"    [FAIL] rule: {r}")
    if result.route is Route.HUMAN_REVIEW:
        lines.append("  -> sent to a human reviewer; no automated decision was taken")
    return "\n".join(lines)


def run_demo(out: Optional[Path] = None, seed: int = 0, quiet: bool = False) -> dict:
    say = (lambda *a: None) if quiet else print
    say(BANNER, "\n")
    ds = make_synthetic(2400, seed=seed)
    sp = SP.make_split("SYN", ds.frame)
    f = ds.frame
    X = _features(f)
    y = f["__label"].to_numpy()
    tri = LogisticRegression(max_iter=500, class_weight="balanced").fit(X[sp.partitions["train"]], y[sp.partitions["train"]])
    p = tri.predict_proba(X)[:, 1]
    sel_y = y[sp.partitions["val_select"]]
    low, high = 0.30, 0.70                                       # an ILLUSTRATIVE fixed band; real runs use the published pair or a validation-fitted one
    say(f"[Stage 0/1] stand-in triage: logistic regression (uncalibrated by design; Stage 4 recalibrates). Illustrative routing band T_low={low}, T_high={high} (inclusive)")

    builder, fixture = EvidenceBuilder(ds), RuleFixtureProvider("faithful")
    mk = lambda **kw: VRFraudNetPipeline(low=low, high=high, provider=fixture, mode="demo", **kw)
    cases = lambda part: [Case(str(f["__id"].iloc[i]), float(p[i]), builder.snapshot(int(i))) for i in sp.partitions[part]]
    sel = mk().run_many(cases("val_select"))
    mixer = VerifiedMixer(min_stratum=25).fit([r.triage_probability for r in sel], [r.verifier.value for r in sel],
                                              [np.nan if r.rationale_probability_used is None else r.rationale_probability_used for r in sel], sel_y)
    cal = mk(mixer=mixer).run_many(cases("val_calibration"))
    conf = SplitConformal(0.10).fit([r.risk_score for r in cal], y[sp.partitions["val_calibration"]])
    say(f"[Stage 4]   mixer fitted on the selection half ({len(sel)} rows); conformal quantile from the held-out calibration half ({conf.n_cal} rows)\n")

    test_pos = sp.partitions["test"]
    tp = p[test_pos]
    i_accept = int(test_pos[np.argmin(tp)])
    review = [int(i) for i, v in zip(test_pos, tp) if low <= v <= high]
    i_review = review[len(review) // 2]
    mk_final = lambda prov: VRFraudNetPipeline(low=low, high=high, provider=prov, mode="demo", mixer=mixer, conformal=conf)
    chosen = [("1. COMMON PATH: below T_low, the verifier is never invoked", i_accept, RuleFixtureProvider("faithful")),
              ("2. REVIEW BAND, rationale verified: every claim is supported by this transaction's evidence", i_review, RuleFixtureProvider("faithful")),
              ("3. REVIEW BAND, rationale REJECTED: the proposed claim is false, so the case goes to a human", i_review, RuleFixtureProvider("false_claim")),
              ("4. REVIEW BAND, rationale REJECTED: the model cites a field that is not in this transaction's evidence", i_review, RuleFixtureProvider("unknown_field"))]
    results = []
    for title, i, prov in chosen:
        r = mk_final(prov).run(Case(str(f["__id"].iloc[i]), float(p[i]), builder.snapshot(i)))
        results.append(r)
        say(title)
        say(explain(r))
        if r.rationale is not None:
            say("  rationale: " + json.dumps(r.rationale, separators=(",", ":"))[:300])
        say("")
    payload = {"banner": BANNER, "thresholds": {"low": low, "high": high}, "results": [r.to_dict() for r in results]}
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        say(f"machine-readable output: {out}")
    return payload
