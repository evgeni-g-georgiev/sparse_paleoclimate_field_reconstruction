"""Tests for analog ensemble selection (paleoreco.assim.analog).

Two carry most of the weight. Selection reads the corrected observation pair, so a
treatment that changes y and R has to change which states are chosen, or selection and the
update disagree about what the observations say and nothing raises. And the evidence rule
at ``kappa = 0`` selects exactly what the misfit rule selects, which is what makes it a
generalisation of the published rule rather than a different estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.analog import correlation_indices, evidence_indices, misfit_indices
from paleoreco.assim.ensrf import whitened_block

Y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
R = np.array([0.5, 2.0, 1.0, 0.25, 4.0, 1.5])


@pytest.fixture
def pool():
    rng = np.random.default_rng(5)
    return rng.normal(size=(40, 6))


@pytest.fixture
def obs_block():
    """A correlated H B H^T over six observation cells, the block the evidence rule uses."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=(20, 6))
    return (a.T @ a) / 19.0


def _whitened(obs_block, r):
    """The factorization the evidence rule reads; only its whitened obs block is used, so
    the ``P H^T`` argument is a stub."""
    return whitened_block(np.zeros((1, len(r))), obs_block, r)


def test_misfit_ranking_matches_an_explicit_loop(pool):
    explicit = np.array([float(np.sum((Y - row) ** 2 / R)) for row in pool])
    assert np.array_equal(misfit_indices(pool, Y, R, 7), np.argsort(explicit, kind="stable")[:7])


def test_r_weighting_changes_the_ranking(pool):
    """Unequal R must matter, or the rule is Sun et al. (2022) Eq. 5's unweighted RMSE."""
    y = pool[0] + np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flat = misfit_indices(pool, y, np.ones(6), 5)
    # Distrust exactly the site carrying the discrepancy.
    weighted = misfit_indices(pool, y, np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 5)
    assert not np.array_equal(flat, weighted)


def test_selection_reads_the_corrected_observation_pair(pool):
    """Deflating y and R must move the selection.

    Deflation divides y by the attenuation and inflates R by its square. If selection were
    scored against the raw pair while the update consumed the corrected one, the two would
    silently answer different questions.
    """
    r = np.full(6, 1.0)
    rho = np.array([1.0, 0.4, 1.0, 0.35, 1.0, 0.5])   # only some samples are stale
    assert not np.array_equal(misfit_indices(pool, Y, r, 8),
                              misfit_indices(pool, Y / rho, r / rho ** 2, 8))


def test_selection_is_deterministic_and_ties_break_by_index():
    pool = np.zeros((6, 3))                            # every candidate scores identically
    y, r = np.ones(3), np.ones(3)
    first = misfit_indices(pool, y, r, 4)
    assert np.array_equal(first, np.arange(4))
    assert np.array_equal(first, misfit_indices(pool, y, r, 4))


def test_kappa_zero_selects_exactly_what_the_misfit_rule_selects(pool, obs_block):
    """The identity the rule rests on: ``kappa = 0`` drops H B H^T and leaves d^T R^-1 d.

    Ties included: a pool of duplicated candidates scores identically under both, so the
    order they come back in must be the same as well as the set.
    """
    twinned = np.repeat(pool[:10], 4, axis=0)
    wb = _whitened(obs_block, R)
    for candidates in (pool, twinned):
        assert np.array_equal(evidence_indices(candidates, Y, R, wb, 12, kappa=0.0),
                              misfit_indices(candidates, Y, R, 12))


def test_evidence_matches_an_explicit_mahalanobis_loop(pool, obs_block):
    kappa = 1.7
    sigma = kappa * obs_block + np.diag(R)
    explicit = np.array([float((Y - row) @ np.linalg.solve(sigma, Y - row)) for row in pool])
    got = evidence_indices(pool, Y, R, _whitened(obs_block, R), 7, kappa=kappa)
    assert np.array_equal(got, np.argsort(explicit, kind="stable")[:7])


def test_evidence_and_misfit_disagree_when_the_prior_is_correlated(pool, obs_block):
    """A positive kappa must move the ranking, or the background block is inert."""
    r = np.full(6, 0.5)
    assert not np.array_equal(evidence_indices(pool, Y, r, _whitened(obs_block, r), 8, kappa=1.0),
                              misfit_indices(pool, Y, r, 8))


def test_evidence_reads_the_corrected_observation_pair(pool, obs_block):
    """Deflating y and R must move the selection, as it does for the misfit rule.

    R enters the evidence metric through the factorization, so the corrected pair has to be
    the one the block was built from.
    """
    r = np.full(6, 1.0)
    rho = np.array([1.0, 0.4, 1.0, 0.35, 1.0, 0.5])
    raw = evidence_indices(pool, Y, r, _whitened(obs_block, r), 8, kappa=1.0)
    corrected = evidence_indices(pool, Y / rho, r / rho ** 2,
                                 _whitened(obs_block, r / rho ** 2), 8, kappa=1.0)
    assert not np.array_equal(raw, corrected)


def test_correlation_matches_an_explicit_loop(pool):
    """Sun et al. (2022) Eq. 6: the spatial mean leaves both sides before they correlate."""
    explicit = np.array([float(np.corrcoef(row, Y)[0, 1]) for row in pool])
    assert np.array_equal(correlation_indices(pool, Y, 7),
                          np.argsort(-explicit, kind="stable")[:7])


def test_correlation_is_blind_to_a_uniform_offset(pool):
    """The property that separates the two published rules.

    Eq. 6 removes the spatial mean, so shifting every observation by the same amount leaves
    its ranking untouched while the misfit rule reads the shift.
    """
    r = np.ones(6)
    assert np.array_equal(correlation_indices(pool, Y, 8), correlation_indices(pool, Y + 3.0, 8))
    assert not np.array_equal(misfit_indices(pool, Y, r, 8), misfit_indices(pool, Y + 3.0, r, 8))


def test_correlation_is_blind_to_amplitude(pool):
    """It normalises both sides, so a uniformly rescaled observation ranks identically."""
    assert np.array_equal(correlation_indices(pool, Y, 8), correlation_indices(pool, 2.5 * Y, 8))


def test_per_channel_correlation_resists_a_dominant_channel(pool):
    """Why the pooled transcription is not the only reading of Eq. 6 here.

    Both published papers correlate over one observed variable. Inflating one channel of a
    two-channel vector leaves the pooled score reading that channel alone, while the
    per-channel form weights the two equally.
    """
    channel = np.array([0, 0, 0, 1, 1, 1])
    loud = pool.copy()
    loud[:, 3:] *= 20.0
    y_loud = Y.copy()
    y_loud[3:] *= 20.0
    assert not np.array_equal(correlation_indices(loud, y_loud, 8),
                              correlation_indices(loud, y_loud, 8, channel=channel))


def test_the_two_correlation_readings_agree_on_a_single_channel(pool):
    """Over one channel there is one spatial mean to remove either way."""
    assert np.array_equal(correlation_indices(pool, Y, 40),
                          correlation_indices(pool, Y, 40, channel=np.zeros(6, dtype=int)))


def test_correlation_scores_a_patternless_candidate_without_raising(pool):
    """A constant candidate has no pattern; it must rank rather than produce a NaN."""
    flat = np.vstack([pool, np.full(6, 2.0)])
    assert len(correlation_indices(flat, Y, len(flat))) == len(flat)


def test_raises_when_the_pool_is_smaller_than_the_ensemble(pool, obs_block):
    with pytest.raises(ValueError, match="candidates"):
        misfit_indices(pool, Y, R, 50)
    with pytest.raises(ValueError, match="candidates"):
        evidence_indices(pool, Y, R, _whitened(obs_block, R), 50, kappa=1.0)
    with pytest.raises(ValueError, match="candidates"):
        correlation_indices(pool, Y, 50)
