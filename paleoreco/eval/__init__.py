"""Evaluation utilities for reconstruction skill and calibration.

Both modules are used by submodule path (``from paleoreco.eval import da``) rather than
re-exported, since their names are generic:

* :mod:`paleoreco.eval.da`
    Reconstruction skill: pooled scalars, per-cell maps, skill against distance to the
    nearest observation, timescale filters, and their plotters.

* :mod:`paleoreco.eval.calibration`
    Whether a posterior's stated uncertainty matches its errors: CRPS, CRPSS, RCRV,
    coverage.
"""
