import math

import numpy as np
import pytest

from vrfraudnet.calibration.conformal import SplitConformal, conformal_quantile, recency_weights, weighted_conformal_quantile
from vrfraudnet.errors import ProtocolError
from vrfraudnet.models.mixer import VerifiedMixer


def test_finite_sample_quantile_rank():
    s = np.arange(1, 101) / 100.0                       # n = 100
    assert conformal_quantile(s, 0.10) == pytest.approx(0.91)   # ceil(101 * 0.9) = 91st smallest
    assert conformal_quantile(s, 0.05) == pytest.approx(0.96)   # ceil(101 * 0.95) = 96
    assert conformal_quantile([0.1, 0.2, 0.3], 0.05) == float("inf")   # ceil(4 * 0.95) = 4 > n: no finite quantile
    assert conformal_quantile([0.4], 0.5) == 0.4                       # ceil(2 * 0.5) = 1


def test_quantile_validation():
    with pytest.raises(ProtocolError):
        conformal_quantile([], 0.1)
    with pytest.raises(ProtocolError):
        conformal_quantile([0.1], 1.5)


def test_marginal_coverage_under_exchangeability():
    rng = np.random.default_rng(0)
    covs = []
    for rep in range(20):
        p = rng.beta(0.4, 3.0, 4000)
        y = (rng.random(4000) < p).astype(int)           # perfectly calibrated probabilities
        cal, test = slice(0, 2000), slice(2000, 4000)
        cp = SplitConformal(0.10).fit(p[cal], y[cal])
        covs.append(cp.coverage(p[test], y[test])["empirical_coverage"])
    assert np.mean(covs) >= 0.90 - 0.01                  # marginal guarantee: average coverage >= 1 - alpha (within noise)


def test_set_structure_and_empty_sets():
    cp = SplitConformal(0.5).fit(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 1, 0, 0]))
    assert cp.threshold == pytest.approx(0.1)            # scores are all 0.1
    sets = cp.predict_sets([0.95, 0.5, 0.05])
    assert sets[0] == (1,) and sets[2] == (0,)
    assert sets[1] == ()                                 # neither class conforms: empty set is kept, not repaired
    assert cp.coverage(np.array([0.5]), np.array([1]))["empty_sets"] == 1


def test_inf_threshold_gives_full_sets():
    cp = SplitConformal(0.05).fit(np.array([0.9, 0.1]), np.array([1, 0]))      # n too small for alpha
    assert cp.threshold == float("inf")
    assert cp.predict_sets([0.3])[0] == (0, 1)


def test_conformal_input_validation():
    with pytest.raises(ProtocolError):
        SplitConformal(0.1).fit([1.5], [1])
    with pytest.raises(ProtocolError):
        SplitConformal(0.1).fit([0.5, 0.5], [1])
    with pytest.raises(ProtocolError):
        SplitConformal(0.1).predict_sets([0.5])


def test_weighted_conformal_matches_unweighted_with_equal_weights():
    s = np.linspace(0, 1, 99)
    q_w = weighted_conformal_quantile(s, np.ones(99), 0.1)
    assert abs(q_w - conformal_quantile(s, 0.1)) <= 0.011
    w = recency_weights([0, 5, 10], 5)
    assert w[2] == 1.0 and w[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------------------------- mixer
def _mixer_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    p_t = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, n), 0.01, 0.99)
    state = rng.choice(["not_invoked", "accepted", "rejected"], n, p=[0.6, 0.25, 0.15])
    p_r = np.where(state == "accepted", np.clip(0.1 + 0.8 * y + rng.normal(0, 0.1, n), 0, 1), np.nan)
    return p_t, state, p_r, y


def test_mixer_is_monotone_and_in_unit_interval():
    p_t, state, p_r, y = _mixer_data()
    m = VerifiedMixer().fit(p_t, state, p_r, y)
    out, prov = m.transform(p_t, state, p_r)
    assert out.min() >= 0 and out.max() <= 1
    order = np.argsort(p_t[state == "not_invoked"])
    assert np.all(np.diff(out[state == "not_invoked"][order]) >= -1e-12)
    assert set(prov) <= {"triage_only", "triage+verified_rationale"}


def test_final_signal_is_invariant_to_a_rejected_rationale_probability():
    p_t, state, p_r, y = _mixer_data()
    m = VerifiedMixer().fit(p_t, state, p_r, y)
    base, _ = m.transform(p_t, state, p_r)
    rej = state == "rejected"
    for junk in (np.nan, 0.0, 1.0, 0.5, 12345.0):                    # whatever the LLM said, a rejected rationale cannot move the score
        p_r2 = p_r.copy()
        p_r2[rej] = junk
        out, _ = m.transform(p_t, state, p_r2)
        assert np.array_equal(out, base)
    ni = state == "not_invoked"
    p_r3 = p_r.copy()
    p_r3[ni] = 0.99
    assert np.array_equal(m.transform(p_t, state, p_r3)[0], base)    # nor can a value attached to an UNINVOKED verifier


def test_accepted_rationale_changes_the_score_and_missing_is_never_zero():
    p_t, state, p_r, y = _mixer_data()
    m = VerifiedMixer().fit(p_t, state, p_r, y)
    acc = state == "accepted"
    hi, lo = p_r.copy(), p_r.copy()
    hi[acc], lo[acc] = 1.0, 0.0
    assert m.w_ > 0                                                  # the rationale is informative in this synthetic setting
    assert (m.transform(p_t, state, hi)[0][acc] >= m.transform(p_t, state, lo)[0][acc]).all()
    bad = p_r.copy()
    bad[acc] = np.nan
    with pytest.raises(ProtocolError):                               # an accepted rationale without a probability is an error, not a zero
        m.transform(p_t, state, bad)


def test_mixer_falls_back_when_a_stratum_is_tiny_and_requires_fit():
    p_t, state, p_r, y = _mixer_data(300)
    m = VerifiedMixer(min_stratum=10_000).fit(p_t, state, p_r, y)       # every stratum too small
    assert m.strata_ == {} and m.w_ == 0.0
    out, _ = m.transform(p_t, state, p_r)
    assert out.shape == p_t.shape
    with pytest.raises(ProtocolError):
        VerifiedMixer().transform([0.5], ["accepted"], [0.5])
    with pytest.raises(ProtocolError):
        VerifiedMixer().fit([0.5], ["accepted", "x"], [0.5], [1])
