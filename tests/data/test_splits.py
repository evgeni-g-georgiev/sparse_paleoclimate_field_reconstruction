"""Tests for the age axis: spacing, split and D-O windows (paleoreco.data.splits)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.data.splits import DO_EVENT_WINDOWS, age_step, chronological_half_split


def test_do_event_window_b2k_to_bp():
    # Event 5 onset 32500 b2k -> 32450 BP, window [-300, +600].
    lo, hi = DO_EVENT_WINDOWS[5]
    assert (lo, hi) == (32_500 - 50 - 300, 32_500 - 50 + 600)


def test_age_step_reads_a_uniform_axis_and_rejects_a_gap():
    assert age_step(np.arange(29_100, 30_000, 25)) == 25.0
    with pytest.raises(ValueError, match="uniform"):
        age_step(np.array([0, 25, 75]))


def test_chronological_half_split_disjoint_and_complete():
    ages = np.arange(29_000, 49_000, 25)
    older, younger = chronological_half_split(ages)
    assert not (set(older.tolist()) & set(younger.tolist()))
    assert np.array_equal(np.sort(np.concatenate([older, younger])), np.arange(len(ages)))


def test_chronological_half_split_older_half_is_later_indices():
    ages = np.arange(29_000, 49_000, 25)          # ascending yr BP
    older, younger = chronological_half_split(ages)
    assert older[0] == len(ages) // 2
    assert ages[older].min() > ages[younger].max()


def test_chronological_half_split_stride_thins_younger_only():
    ages = np.arange(29_000, 49_000, 25)
    older, younger = chronological_half_split(ages, stride=10)
    older_1, younger_1 = chronological_half_split(ages, stride=1)
    assert np.array_equal(older, older_1)         # the prior half is never thinned
    assert np.array_equal(younger, younger_1[::10])
    assert set(younger.tolist()).issubset(set(younger_1.tolist()))


@pytest.mark.parametrize("n", [12, 13])
def test_chronological_half_split_sizes_even_and_odd(n):
    older, younger = chronological_half_split(np.arange(n))
    assert len(younger) == n // 2                 # the odd state goes to the prior half
    assert len(older) == n - n // 2
