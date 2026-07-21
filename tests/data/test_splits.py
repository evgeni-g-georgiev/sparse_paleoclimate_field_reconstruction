"""Tests for D-O event windows and CV split utilities (paleoreco.data.splits)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.data.splits import (
    DO_EVENT_WINDOWS,
    assign_event_label,
    chronological_half_split,
    make_blocked_cv,
)


def test_do_event_window_b2k_to_bp():
    # Event 5 onset 32500 b2k -> 32450 BP, window [-300, +600].
    lo, hi = DO_EVENT_WINDOWS[5]
    assert (lo, hi) == (32_500 - 50 - 300, 32_500 - 50 + 600)


def test_assign_event_label_inside_outside_and_boundary():
    lo, hi = DO_EVENT_WINDOWS[8]
    ages = np.array([lo - 1, lo, (lo + hi) // 2, hi, hi + 1])
    labels = assign_event_label(ages)
    assert labels[0] == 0          # just before window
    assert labels[1] == 8          # inclusive lower bound
    assert labels[2] == 8
    assert labels[3] == 8          # inclusive upper bound
    assert labels[4] == 0          # just after window


def test_make_blocked_cv_holds_out_test_event():
    ages = np.arange(29_000, 49_000, 25)
    cv = make_blocked_cv(ages, fold_years=1000, embargo_years=1000,
                         holdout_test=True, test_event=8, test_fraction=0.1)
    assert len(cv["test"]) > 0
    assert len(cv["folds"]) > 0
    # Test ages come only from event 8 or between-event ages.
    labels = assign_event_label(ages[cv["test"]])
    assert set(np.unique(labels)).issubset({0, 8})
    # Folds keep train and val disjoint.
    for fold in cv["folds"]:
        assert not (set(fold["train"]) & set(fold["val"]))


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
