"""Tests for the Prior-cube loader and anomaly helpers (paleoreco.data.cube)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paleoreco.data import (
    GRID_SHAPE,
    VARS,
    PaleoFieldDataset,
    apply_anomaly,
    build_prior_cube,
    compute_zscore_stats,
    highpass_states,
)


def _write_prior_csv(path, *, drop_one=False) -> None:
    """A full 2x2 grid over 3 ages in the raw long Prior format."""
    lons, lats, ages = [-10.0, 10.0], [-5.0, 5.0], [1000, 1025, 1050]
    rows = []
    for a in ages:
        for la in lats:
            for lo in lons:
                rows.append({"lon": lo, "lat": la, "age": a,
                             "mtco": -20.0 + a * 1e-3 + lo,
                             "mtwa": 5.0 + a * 1e-3 + la})
    df = pd.DataFrame(rows)
    if drop_one:
        df = df.iloc[1:]  # leave a hole -> a missing (age, lat, lon) cell
    df.to_csv(path, index=True)


def test_build_prior_cube_shapes_and_axes(tmp_path):
    csv = tmp_path / "prior.csv"
    _write_prior_csv(csv)
    out = build_prior_cube(str(csv), cache_path=None)

    assert out["cube"].shape == (3, 2, 2, 2)
    assert list(out["ages"]) == [1000, 1025, 1050]
    assert np.allclose(out["lats"], [-5.0, 5.0])
    assert np.allclose(out["lons"], [-10.0, 10.0])
    assert out["valid"].all()
    # Channel 0 is mtco, channel 1 is mtwa, placed by (lat, lon) index.
    assert out["cube"][0, 0, 0, 0] == pytest.approx(-20.0 + 1.0 + (-10.0))
    assert out["cube"][0, 1, 1, 0] == pytest.approx(5.0 + 1.0 + 5.0)


def test_build_prior_cube_raises_on_missing_cell(tmp_path):
    csv = tmp_path / "prior.csv"
    _write_prior_csv(csv, drop_one=True)
    with pytest.raises(ValueError, match="missing"):
        build_prior_cube(str(csv), cache_path=None)


def test_grid_and_vars_constants():
    assert GRID_SHAPE == (32, 64)
    assert VARS == ("mtco", "mtwa")


def test_anomaly_centring_and_degenerate_masking(cube, valid):
    # Make one cell constant across ages on the mtco channel -> degenerate std.
    cube = cube.copy()
    cube[:, 0, 2, 3] = -4.0
    train_idx = np.arange(cube.shape[0])

    stats = compute_zscore_stats(cube, train_idx, valid)
    assert stats["safe_valid"][2, 3] == False  # noqa: E712 - degenerate cell dropped
    assert stats["safe_valid"].sum() == valid.sum() - 1

    a = apply_anomaly(cube, stats)
    # Masked cell is zeroed in anomaly space.
    assert np.allclose(a[:, 0, 2, 3], 0.0)
    # Adding the climatology back recovers the original on safe cells.
    safe = stats["safe_valid"]
    recovered = a + stats["mean"]
    assert np.allclose(recovered[:, 0, safe], cube[:, 0, safe], atol=1e-4)


def test_dataset_returns_field_plus_mask(cube, valid):
    train_idx = np.arange(cube.shape[0])
    stats = compute_zscore_stats(cube, train_idx, valid)
    a = apply_anomaly(cube, stats)
    ds = PaleoFieldDataset(a, stats["safe_valid"], train_idx)

    assert len(ds) == cube.shape[0]
    sample = ds[0]
    assert tuple(sample.shape) == (3, cube.shape[2], cube.shape[3])
    # Third channel is the binary mask.
    assert np.allclose(sample[2].numpy(), stats["safe_valid"].astype(np.float32))


# ----------------------------------------------------------------------------
# Band-pass along the age axis.
# ----------------------------------------------------------------------------
def test_highpass_window_none_is_the_identity(cube, ages):
    out = highpass_states(cube, ages, None)
    assert np.allclose(out, cube)
    assert out.shape == cube.shape


def test_highpass_removes_the_slow_part_and_keeps_the_fast_one(ages):
    """A trend in age is removed; an alternation between consecutive states survives."""
    n = len(ages)
    trend = (np.arange(n, dtype=float) * 3.0)[:, None] * np.ones((1, 4))
    fast = ((-1.0) ** np.arange(n))[:, None] * np.ones((1, 4))

    # Judge the interior only: at the ends the window is one-sided, so a trend leaves a
    # residual there by construction.
    interior = slice(2, n - 2)
    assert np.abs(highpass_states(trend, ages, 1500.0)[interior]).max() < 1e-10
    kept = highpass_states(fast, ages, 1500.0)[interior]
    assert np.abs(kept).min() > 0.5 * np.abs(fast[interior]).min()


def test_highpass_uses_only_the_states_it_is_given(ages):
    """Filtering a subset must not depend on the states left out of it."""
    n = len(ages)
    x = np.random.default_rng(0).normal(size=(n, 5))
    keep = np.arange(0, n, 2)
    from_subset = highpass_states(x[keep], np.asarray(ages)[keep], 2500.0)
    perturbed = x.copy()
    perturbed[1] += 100.0                       # a state outside ``keep``
    assert np.allclose(from_subset,
                       highpass_states(perturbed[keep], np.asarray(ages)[keep], 2500.0))


def test_highpass_rejects_mismatched_ages_and_bad_windows(ages):
    x = np.zeros((len(ages), 3))
    with pytest.raises(ValueError, match="one value per state"):
        highpass_states(x, ages[:-1], 1500.0)
    with pytest.raises(ValueError, match="positive"):
        highpass_states(x, ages, 0.0)
