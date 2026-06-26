"""Tests for D-O event windows and CV split utilities (paleoreco.splits)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.data.splits import (
    DO_EVENT_WINDOWS,
    assign_event_label,
    block_stride_split,
    make_blocked_cv,
    split_ages_by_do_event,
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


def test_split_ages_by_do_event_partitions_disjointly():
    ages = np.arange(29_000, 49_000, 25)
    split = split_ages_by_do_event(ages, test_event=8, val_event=7)
    idx = np.concatenate([split["train"], split["val"], split["test"]])
    # Every age assigned exactly once.
    assert np.array_equal(np.sort(idx), np.arange(len(ages)))
    assert (assign_event_label(ages[split["test"]]) == 8).all()
    assert (assign_event_label(ages[split["val"]]) == 7).all()


def test_split_ages_rejects_equal_events():
    ages = np.arange(29_000, 49_000, 25)
    with pytest.raises(ValueError):
        split_ages_by_do_event(ages, test_event=8, val_event=8)


def test_block_stride_split_detects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        block_stride_split(400, block_size=40, test_stride=2, test_offset=0,
                           val_stride=2, val_offset=0)


def test_block_stride_split_disjoint_buckets():
    n = 400
    split = block_stride_split(n, block_size=40, test_stride=5, test_offset=0,
                               val_stride=5, val_offset=2)
    s = {k: set(v.tolist()) for k, v in split.items()}
    assert not (s["train"] & s["val"])
    assert not (s["train"] & s["test"])
    assert not (s["val"] & s["test"])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == n


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
