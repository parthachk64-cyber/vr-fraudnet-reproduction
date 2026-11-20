"""Stage 4a: the validation-fitted isotonic mixer (implementation decision D-04).

The paper says the mixer combines the triage probability, the verifier-accepted rationale probability and the
verifier status through isotonic regression, but not how three inputs become one monotone scalar. The rule
implemented here, stated so it can be challenged:

  * Verifier state selects a STRATUM: common-path (not_invoked), accepted, rejected.
  * not_invoked / rejected rows are scored by the triage probability alone. A rejected rationale's probability is
    NEVER read: a test perturbs it and asserts the output is unchanged.
  * accepted rows are scored by s = (1 - w) p_triage + w p_rationale, with w chosen on the selection half of the
    validation segment by AUPRC (w = 0 if there is nothing to choose from).
  * Each stratum with >= `min_stratum` rows and both classes gets its own isotonic regression; others fall back to a
    single isotonic regression fitted on the triage probability of all rows.

A missing rationale probability is NaN, never 0.0.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..errors import ProtocolError
from ..evaluation.metrics import auprc
from ..types import VerifierState

NOT_INVOKED, ACCEPTED, REJECTED = (s.value for s in VerifierState)


def _iso() -> IsotonicRegression:
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)


class VerifiedMixer:
    def __init__(self, min_stratum: int = 50, w_grid=None, learned: bool = True):
        self.min_stratum, self.learned = min_stratum, learned
        self.w_grid = np.linspace(0.0, 1.0, 11) if w_grid is None else np.asarray(w_grid, float)
        self.w_: float = 0.0
        self.fallback_: Optional[IsotonicRegression] = None
        self.strata_: dict[str, IsotonicRegression] = {}
        self.fitted_: bool = False

    def _score(self, p_t, p_r, state) -> np.ndarray:
        s = np.array(p_t, dtype=float, copy=True)
        acc = state == ACCEPTED
        if acc.any():
            pr = np.where(acc, p_r, 0.0)                      # p_r is only ever read where the rationale was ACCEPTED
            if not np.isfinite(pr[acc]).all():
                raise ProtocolError("an accepted rationale must carry a finite probability")
            s = np.where(acc, (1.0 - self.w_) * p_t + self.w_ * pr, s)
        return s

    def fit(self, p_triage, state, p_rationale, y) -> "VerifiedMixer":
        p_t = np.asarray(p_triage, float)
        state = np.asarray(state, dtype=object).astype(str)
        p_r = np.asarray(p_rationale, float)
        y = np.asarray(y).astype(int)
        if not (len(p_t) == len(state) == len(p_r) == len(y)):
            raise ProtocolError("mixer inputs differ in length")
        self.fallback_ = _iso().fit(p_t, y)
        acc = state == ACCEPTED
        self.w_ = 0.0
        if self.learned and acc.sum() >= self.min_stratum and len(np.unique(y[acc])) == 2:
            best, best_w = -1.0, 0.0
            for w in self.w_grid:
                v = auprc(y[acc], (1 - w) * p_t[acc] + w * p_r[acc])
                if v > best + 1e-12:
                    best, best_w = v, float(w)
            self.w_ = best_w
        score = self._score(p_t, p_r, state)
        self.strata_ = {}
        if self.learned:
            for name in (NOT_INVOKED, ACCEPTED, REJECTED):
                m = state == name
                if m.sum() >= self.min_stratum and len(np.unique(y[m])) == 2:
                    self.strata_[name] = _iso().fit(score[m], y[m])
        self.fitted_ = True
        return self

    def transform(self, p_triage, state, p_rationale):
        """Return (calibrated risk, provenance) where provenance is 'triage_only' or 'triage+verified_rationale'."""
        if not self.fitted_:
            raise ProtocolError("VerifiedMixer.fit must be called first")
        p_t = np.asarray(p_triage, float)
        state = np.asarray(state, dtype=object).astype(str)
        p_r = np.asarray(p_rationale, float)
        score = self._score(p_t, p_r, state)
        out = np.empty(len(p_t))
        for name in (NOT_INVOKED, ACCEPTED, REJECTED):
            m = state == name
            if m.any():
                model = self.strata_.get(name)
                out[m] = model.predict(score[m]) if model is not None else self.fallback_.predict(score[m])
        prov = np.where(state == ACCEPTED, "triage+verified_rationale", "triage_only")
        return out, prov
