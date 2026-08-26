"""Tests for analog ensemble selection (paleoreco.assim.analog).

The load-bearing one is that selection reads the corrected observation pair: a treatment
that changes y and R has to change which states are chosen, or selection and the update
disagree about what the observations say and nothing raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.analog import (
    analog_indices,
    eligible_mask,
    window_indices,
)


@pytest.fixture
def pool():
    rng = np.random.default_rng(5)
    return rng.normal(size=(40, 6))


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
