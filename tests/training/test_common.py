"""Tests for the shared trainer helpers (set_seed, snapshot).

These live in training/_common.py and are re-exported through training.trainer_ae
(the path the notebooks use); the asserted behaviour is unchanged by the refactor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from paleoreco.training.trainer_ae import _snapshot_state_dict, set_seed


def test_set_seed_makes_draws_reproducible():
    set_seed(123)
    a = torch.randn(5)
    n = float(torch.rand(1))
    set_seed(123)
    b = torch.randn(5)
    m = float(torch.rand(1))
    assert torch.equal(a, b)
    assert n == m


def test_set_seed_different_seeds_differ():
    set_seed(0)
    a = torch.randn(5)
    set_seed(1)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_snapshot_state_dict_is_detached_cpu_clone():
    model = nn.Linear(3, 2)
    snap = _snapshot_state_dict(model)
    for name, tensor in snap.items():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
    # Mutating the live model must not change the snapshot (it is a clone).
    before = snap["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert torch.equal(snap["weight"], before)
