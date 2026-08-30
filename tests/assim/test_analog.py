"""Tests for analog ensemble selection (paleoreco.assim.analog).

Two carry most of the weight. Selection reads the corrected observation pair, so a
treatment that changes y and R has to change which states are chosen, or selection and the
update disagree about what the observations say and nothing raises. And the evidence rule
reproduces the misfit rule at zero background scale, which is what makes it a
generalisation of the published one rather than a different estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.analog import (
    analog_indices,
    eligible_mask,
    evidence_indices,
    window_indices,
)
from paleoreco.assim.ensrf import whitened_block


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
    y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
    r = np.array([0.5, 2.0, 1.0, 0.25, 4.0, 1.5])
    explicit = np.array([float(np.sum((y - row) ** 2 / r)) for row in pool])
    assert np.array_equal(analog_indices(pool, y, r, 7), np.argsort(explicit, kind="stable")[:7])


def test_r_weighting_changes_the_ranking(pool):
    """Unequal R must matter, or the rule is Wu et al.'s unweighted RMSE by another name."""
    y = pool[0] + np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flat = analog_indices(pool, y, np.ones(6), 5)
    # Distrust exactly the site carrying the discrepancy.
    weighted = analog_indices(pool, y, np.array([100.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 5)
    assert not np.array_equal(flat, weighted)


def test_selection_reads_the_corrected_observation_pair(pool):
    """Deflating y and R must move the selection.

    Stage 2's deflation divides y by the attenuation and inflates R by its square. If
    selection were scored against the raw pair while the update consumed the corrected
    one, the two would silently answer different questions.
    """
    y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
    r = np.full(6, 1.0)
    rho = np.array([1.0, 0.4, 1.0, 0.35, 1.0, 0.5])   # only some samples are stale
    raw = analog_indices(pool, y, r, 8)
    corrected = analog_indices(pool, y / rho, r / rho ** 2, 8)
    assert not np.array_equal(raw, corrected)


def test_selection_is_deterministic_and_ties_break_by_index():
    pool = np.zeros((6, 3))                            # every candidate scores identically
    y, r = np.ones(3), np.ones(3)
    first = analog_indices(pool, y, r, 4)
    assert np.array_equal(first, np.arange(4))
    assert np.array_equal(first, analog_indices(pool, y, r, 4))


def test_evidence_at_zero_scale_reproduces_the_misfit_rule(pool, obs_block):
    """The identity the rule rests on: c = 0 drops H B H^T and leaves d^T R^-1 d.

    U is orthogonal, so summing the whitened residual over its columns is the R-weighted
    misfit however correlated the block is. Without this the evidence rule would be a
    different estimator rather than a generalisation of the published one.
    """
    y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
    r = np.array([0.5, 2.0, 1.0, 0.25, 4.0, 1.5])
    assert np.array_equal(
        evidence_indices(pool, y, _whitened(obs_block, r), 9, scale=0.0),
        analog_indices(pool, y, r, 9))


def test_evidence_matches_an_explicit_mahalanobis_loop(pool, obs_block):
    y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
    r = np.array([0.5, 2.0, 1.0, 0.25, 4.0, 1.5])
    scale = 1.7
    sigma = scale * obs_block + np.diag(r)
    explicit = np.array([float((y - row) @ np.linalg.solve(sigma, y - row)) for row in pool])
    got = evidence_indices(pool, y, _whitened(obs_block, r), 7, scale=scale)
    assert np.array_equal(got, np.argsort(explicit, kind="stable")[:7])


def test_evidence_and_misfit_disagree_when_the_prior_is_correlated(pool, obs_block):
    """A nonzero scale must move the ranking, or the background block is inert."""
    y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
    r = np.full(6, 0.5)
    assert not np.array_equal(evidence_indices(pool, y, _whitened(obs_block, r), 8),
                              analog_indices(pool, y, r, 8))


def test_evidence_reads_the_corrected_observation_pair(pool, obs_block):
    """Deflating y and R must move the selection, as it does for the misfit rule.

    R enters the evidence metric through the factorization, so the corrected pair has to be
    the one the block was built from; scoring a raw pair against a corrected update is the
    disagreement this rule exists to remove.
    """
    y = np.array([0.4, -1.1, 0.2, 0.9, -0.3, 0.0])
    r = np.full(6, 1.0)
    rho = np.array([1.0, 0.4, 1.0, 0.35, 1.0, 0.5])
    raw = evidence_indices(pool, y, _whitened(obs_block, r), 8)
    corrected = evidence_indices(pool, y / rho, _whitened(obs_block, r / rho ** 2), 8)
    assert not np.array_equal(raw, corrected)


def test_evidence_respects_eligibility_and_the_candidate_count(pool, obs_block):
    ages = np.arange(40) * 100.0
    y, r = pool[20].copy(), np.ones(6)
    wb = _whitened(obs_block, r)
    assert 20 in evidence_indices(pool, y, wb, 5)
    banded = evidence_indices(pool, y, wb, 5, eligible=eligible_mask(ages, 2000.0, 500.0))
    assert np.all(np.abs(ages[banded] - 2000.0) >= 500.0)
    with pytest.raises(ValueError, match="eligible candidates"):
        evidence_indices(pool, y, wb, 30, eligible=eligible_mask(ages, 2000.0, 2000.0))


def test_eligible_mask_band_semantics():
    ages = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    assert eligible_mask(ages, 3000, 0.0) is None      # zero width excludes nothing
    mask = eligible_mask(ages, 3000, 1000.0)
    # The bound is inclusive: a state exactly the band width away stays eligible.
    assert list(mask) == [True, True, False, True, True]


def test_exclusion_band_keeps_selected_ages_outside_it(pool):
    ages = np.arange(40) * 100.0
    y, r = pool[20].copy(), np.ones(6)                 # the target's own state is the best match
    assert 20 in analog_indices(pool, y, r, 5)
    banded = analog_indices(pool, y, r, 5, eligible=eligible_mask(ages, 2000.0, 500.0))
    assert np.all(np.abs(ages[banded] - 2000.0) >= 500.0)


def test_window_selects_the_nearest_in_time_and_respects_eligibility():
    ages = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)
    assert sorted(window_indices(ages, 250, 2)) == [2, 3]
    banded = window_indices(ages, 250, 2, eligible=eligible_mask(ages, 250, 150.0))
    assert sorted(banded) == [1, 4]


def test_raises_when_too_few_candidates_are_eligible(pool):
    ages = np.arange(40) * 100.0
    y, r = pool[0], np.ones(6)
    with pytest.raises(ValueError, match="eligible candidates"):
        analog_indices(pool, y, r, 30, eligible=eligible_mask(ages, 2000.0, 2000.0))
    with pytest.raises(ValueError, match="candidates"):
        analog_indices(pool, y, r, 50)
