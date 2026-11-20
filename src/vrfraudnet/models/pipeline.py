"""The five-stage pipeline for ONE transaction at a time.

            p_triage (Stage 0 + 1, computed upstream by TriageModel)
                |
     gate:  p < T_low -> ACCEPT   p > T_high -> REJECT   else ESCALATE (inclusive band)
                |                                              |
   verifier NOT_INVOKED                        retrieval -> Stage 2 -> Stage 3 verifier
                |                                  |  accepted               | rejected
                |                          VERIFIED_AUTO              HUMAN_REVIEW (LM probability never used)
                +------------------> Stage 4: mixer (isotonic) -> calibrated risk -> split-conformal set

Research mode refuses fixture providers/embedders. Ablations are benchmark-only: they need `mode="ablation"` (or `demo_ablation`), an explicit
`Ablation`, and mark every result's provenance so an ablated output can never be mistaken for a protected-policy output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..calibration.conformal import SplitConformal
from ..errors import FixtureInResearchModeError, MissingInputError, ProtocolError
from ..retrieval.index import ExemplarIndex, Query, summarise
from ..types import Route, TransactionResult, VerificationReport, VerifierState
from ..verifier import verify_text
from ..verifier.evidence import EvidenceSnapshot
from .mixer import VerifiedMixer
from .rationale import RationaleProvider
from .triage import ACCEPT, ESCALATE, REJECT, route_probability


@dataclass
class Case:
    transaction_id: str
    p_triage: float
    evidence: EvidenceSnapshot
    entity_ids: frozenset = field(default_factory=frozenset)


@dataclass(frozen=True)
class Ablation:
    """Table 7 variants that change INFERENCE. Training-side ablations (A4 adversarial, A5 cost, A9 counterfactual) are training flags."""

    name: str = "full"
    use_verifier: bool = True      # A1
    use_gate: bool = True          # A2 (False: every transaction goes through Stage 2)
    use_schema: bool = True        # A3 (False: free-form decoding, no grammar mask)
    use_llm: bool = True           # A6
    use_mixer: bool = True         # A7 (False: plain average, no isotonic fit)

    @property
    def is_full(self) -> bool:
        return all((self.use_verifier, self.use_gate, self.use_schema, self.use_llm, self.use_mixer))


class VRFraudNetPipeline:
    def __init__(self, *, low: float, high: float, provider: Optional[RationaleProvider] = None, index: Optional[ExemplarIndex] = None,
                 mixer: Optional[VerifiedMixer] = None, conformal: Optional[SplitConformal] = None, operating_threshold: float = 0.5,
                 mode: str = "research", ablation: Optional[Ablation] = None, top_k: int = 4, direction_hints=None):
        if mode not in ("research", "demo", "ablation", "demo_ablation"):
            raise ValueError("mode must be 'research', 'demo', 'ablation' (benchmark-grade) or 'demo_ablation' (fixtures allowed)")
        self.ablation = ablation or Ablation()
        if not self.ablation.is_full and mode not in ("ablation", "demo_ablation"):
            raise ProtocolError("an Ablation that disables a protective component needs mode='ablation' (or 'demo_ablation' with fixtures)")
        if mode in ("research", "ablation"):
            if provider is not None and provider.is_fixture:
                raise FixtureInResearchModeError(f"provider {provider.name!r} is a fixture; benchmark runs need a real rationale model")
            if index is not None and index.embedder.is_fixture:
                raise FixtureInResearchModeError(f"embedder {index.embedder.name!r} is a fixture; benchmark runs need the MiniLM encoder")
        if low >= high:
            raise ProtocolError("need T_low < T_high")
        self.low, self.high, self.provider, self.index, self.mixer, self.conformal = low, high, provider, index, mixer, conformal
        self.operating_threshold, self.mode, self.top_k, self.direction_hints = operating_threshold, mode, top_k, direction_hints

    # ---------------------------------------------------------------------------------------
    def _calibrate(self, p_t: float, state: VerifierState, p_r: float):
        if self.mixer is None or not self.ablation.use_mixer:
            risk = p_t if np.isnan(p_r) else 0.5 * (p_t + p_r)
            return float(risk), "triage_only_uncalibrated" if np.isnan(p_r) else "average_uncalibrated"
        out, prov = self.mixer.transform([p_t], [state.value], [p_r])
        return float(out[0]), str(prov[0])

    def _result(self, case: Case, route: Route, state: VerifierState, p_r: float, rationale, report, automated: bool) -> TransactionResult:
        risk, prov = self._calibrate(case.p_triage, state if state is not VerifierState.NOT_INVOKED or self.ablation.use_verifier else VerifierState.ACCEPTED, p_r)
        if not self.ablation.is_full:
            prov = f"ABLATION:{self.ablation.name}:{prov}"
        cs = self.conformal.predict_sets([risk])[0] if self.conformal is not None and self.conformal.threshold is not None else None
        return TransactionResult(transaction_id=case.transaction_id, route=route, triage_probability=float(case.p_triage), risk_score=risk, score_provenance=prov,
                                 verifier=state, rationale=rationale, rationale_probability_used=None if np.isnan(p_r) else float(p_r), verification=report,
                                 conformal_set=cs, automated=automated)

    def run(self, case: Case) -> TransactionResult:
        if not 0.0 <= case.p_triage <= 1.0 or not np.isfinite(case.p_triage):
            raise ProtocolError("triage probability must be finite and in [0, 1]")
        gate = ESCALATE if not self.ablation.use_gate else str(route_probability([case.p_triage], self.low, self.high)[0])
        nan = float("nan")
        if gate == ACCEPT:
            return self._result(case, Route.ACCEPT, VerifierState.NOT_INVOKED, nan, None, None, True)
        if gate == REJECT:
            return self._result(case, Route.REJECT, VerifierState.NOT_INVOKED, nan, None, None, True)
        if not self.ablation.use_llm:                                    # A6: no rationale model; decide on the calibrated triage score
            r = self._result(case, Route.HUMAN_REVIEW, VerifierState.NOT_INVOKED, nan, None, None, True)
            return r
        if self.provider is None:
            raise MissingInputError("a rationale provider", needed_for="review-band transactions", how_to_fix="pass provider=LlamaLoRAProvider(...) (or RuleFixtureProvider in demo mode)")
        demos = []
        if self.index is not None:
            ev = case.evidence
            q = Query(case.transaction_id, summarise(ev.fields, "review", case.p_triage), ev.prediction_time, case.entity_ids or frozenset(ev.entities.values()))
            demos = self.index.retrieve(q, self.top_k)
        gen = self.provider.generate(case.evidence, demos, case.p_triage, constrained=self.ablation.use_schema)
        if not self.ablation.use_verifier:                               # A1: parse only; NEVER do this outside an ablation
            try:
                import json

                obj = json.loads(gen.text)
                p_r = float(obj["probability"])
                return self._result(case, Route.VERIFIED_AUTO, VerifierState.NOT_INVOKED, p_r, obj, None, True)
            except Exception:  # noqa: BLE001
                return self._result(case, Route.HUMAN_REVIEW, VerifierState.NOT_INVOKED, nan, None, None, False)
        if not gen.complete:
            report = VerificationReport(VerifierState.REJECTED, (), ("incomplete_generation: token limit reached before the object closed",))
            return self._result(case, Route.HUMAN_REVIEW, VerifierState.REJECTED, nan, None, report, False)
        report = verify_text(gen.text, case.evidence, direction_hints=self.direction_hints)
        if report.accepted:
            import json

            rat = json.loads(gen.text)
            return self._result(case, Route.VERIFIED_AUTO, VerifierState.ACCEPTED, float(rat["probability"]), rat, report, True)
        return self._result(case, Route.HUMAN_REVIEW, VerifierState.REJECTED, nan, None, report, False)

    def run_many(self, cases: Sequence[Case]) -> list[TransactionResult]:
        return [self.run(c) for c in cases]

    def decision(self, result: TransactionResult) -> Optional[str]:
        """'fraud' / 'legitimate' for automated results, None when the case is for a human."""
        if not result.automated or result.route is Route.HUMAN_REVIEW:
            return None
        if result.route is Route.ACCEPT:
            return "legitimate"
        if result.route is Route.REJECT:
            return "fraud"
        return "fraud" if result.risk_score >= self.operating_threshold else "legitimate"
