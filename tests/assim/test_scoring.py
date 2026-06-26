"""Tests for coordinate scorings (paleoreco.assim.scoring)."""

from __future__ import annotations

import numpy as np
import pytest

from paleoreco.assim.scoring import ANOMALY, NORMALISED, RAW, score


def test_score_modes():
    values = np.array([2.0, 4.0, 6.0])
    mean = np.array([1.0, 1.0, 1.0])
    std = np.array([2.0, 2.0, 2.0])
    assert np.allclose(score(values, mean, std, RAW), values)
    assert np.allclose(score(values, mean, std, ANOMALY), values - mean)
    assert np.allclose(score(values, mean, std, NORMALISED), (values - mean) / std)


def test_score_rejects_unknown_mode():
    with pytest.raises(ValueError):
        score(np.zeros(2), np.zeros(2), np.ones(2), "bogus")
