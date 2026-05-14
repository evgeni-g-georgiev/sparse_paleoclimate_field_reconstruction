"""Evaluation utilities for paleoreco models.

The package is split into two submodules so downstream work can reuse the
generic primitives without depending on AE-specific contracts:

* :mod:`paleoreco.eval.shared`
    Model-family-agnostic primitives. POD baseline, per-cell °C RMSE
    arithmetic, all the figure-producing plotters (reconstruction
    grids, RMSE maps, latent-sweep curves, distribution histograms).
    These take numpy arrays: they don't care which model produced them.

* :mod:`paleoreco.eval.ae`
    Autoencoder specific helpers that depend on either the model's
    forward contract (``model(x) -> (x_hat, z)``) or the ``history``
    dict produced by :mod:`paleoreco.train_ae`. When downstream model 
    eval modules arrive, they live alongside as ``paleoreco.eval.diffusion`` etc.,
    and keep using everything in :mod:`paleoreco.eval.shared`.

For convenience the public API of both submodules is re-exported here,
so callers can simply do::

    from paleoreco.eval import plot_per_cell_rmse, plot_loss_curves

without reaching into the submodules.
"""

from paleoreco.eval.ae import (
    plot_loss_curves,
    reconstruct_split,
)
from paleoreco.eval.shared import (
    compute_E_d,
    compute_pod_time_coefficients,
    partition_latent_2d,
    per_cell_rmse_celsius,
    per_mode_learning_accuracy,
    plot_latent_2d,
    plot_latent_sweep,
    plot_per_cell_rmse,
    plot_per_cluster_pod_distributions,
    plot_per_cluster_reconstructions,
    plot_per_mode_learning_curves,
    plot_recon_distribution,
    plot_reconstructions,
    pod_fit,
    pod_predict,
    pod_test_rmse,
)

__all__ = [
    # shared - layer 1 (sweep + headline metric)
    "compute_E_d",
    "per_cell_rmse_celsius",
    "plot_latent_sweep",
    "plot_per_cell_rmse",
    "plot_recon_distribution",
    "plot_reconstructions",
    "pod_fit",
    "pod_predict",
    "pod_test_rmse",
    # shared - layer 2 (Bousquet latent-space deep dive)
    "compute_pod_time_coefficients",
    "partition_latent_2d",
    "per_mode_learning_accuracy",
    "plot_latent_2d",
    "plot_per_cluster_pod_distributions",
    "plot_per_cluster_reconstructions",
    "plot_per_mode_learning_curves",
    # ae
    "plot_loss_curves",
    "reconstruct_split",
]
