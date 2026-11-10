"""Evaluation utilities for reconstruction skill and calibration.

The modules are used by submodule path (``from paleoreco.eval import da``) rather than
re-exported, since their names are generic:

* :mod:`paleoreco.eval.da`
    Reconstruction skill: pooled scalars, the per-cell CE map, skill against distance to
    the nearest observation, and timescale filters.

* :mod:`paleoreco.eval.calibration`
    Whether a posterior's stated uncertainty matches its errors: CRPS and RCRV.

* :mod:`paleoreco.eval.rows`
    Both of the above as long-format metric rows, the schema every lane writes.
"""
