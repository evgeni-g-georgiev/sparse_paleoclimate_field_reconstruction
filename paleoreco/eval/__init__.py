"""Evaluation utilities for paleoreco models.

The package is split by model family so downstream work can reuse the
generic primitives without depending on AE-specific contracts:

* :mod:`paleoreco.eval.shared`
    Model-family-agnostic primitives. POD baseline, per-cell °C RMSE
    arithmetic, all the figure-producing plotters (reconstruction
    grids, RMSE maps, latent-sweep curves, distribution histograms).
    These take numpy arrays: they don't care which model produced them.

* :mod:`paleoreco.eval.ae`
    Autoencoder specific helpers that depend on either the model's
    forward contract (``model(x) -> (x_hat, z)``) or the ``history``
    dict produced by :mod:`paleoreco.training.trainer_ae`.

* :mod:`paleoreco.eval.vae`
    Beta-VAE specific helpers that depend on the VAE forward contract
    (``model(x) -> (x_hat, mu, logvar, z)``) or the ``history`` dict
    produced by :mod:`paleoreco.training.trainer_vae`.

* :mod:`paleoreco.eval.gaussianity`
    Pooled-innovation Gaussianity plots (histogram and normal QQ) for
    data-assimilation diagnostics. Takes numpy arrays of standardised
    innovations.

* :mod:`paleoreco.eval.projection`
    Principal-component projections of a field stack, the 1-D linear
    projections used to inspect a sample of fields for departures from
    Gaussianity.

* :mod:`paleoreco.eval.da`
    Reconstruction skill: pooled scalars, per-cell maps, skill against
    distance to the nearest observation, and their plotters.

* :mod:`paleoreco.eval.calibration`
    Whether a posterior's stated uncertainty matches its errors: CRPS,
    CRPSS, RCRV, coverage.

The figure and metric primitives are re-exported here, so callers can do::

    from paleoreco.eval import plot_per_cell_rmse, plot_loss_curves

``da`` and ``calibration`` are used as modules (``from paleoreco.eval import da``)
rather than re-exported, since their names are generic.
"""

from paleoreco.eval.ae import (
    plot_loss_curves,
    reconstruct_split,
)
from paleoreco.eval.vae import (
    compute_vae_diagnostics,
    latent_traversal,
    plot_loss_curves_vae,
    reconstruct_split_vae,
)
from paleoreco.eval.gaussianity import (
    plot_innovation_gaussianity,
    plot_pairwise_gaussianity,
)
from paleoreco.eval.projection import (
    pca_scores,
)
from paleoreco.eval.shared import (
    compute_E_d,
    compute_pod_time_coefficients,
    partition_latent_2d,
    per_cell_rmse_celsius,
    per_mode_learning_accuracy,
    plot_decoded_samples,
    plot_latent_2d,
    plot_latent_2d_with_sigma,
    plot_latent_sweep,
    plot_latent_traversal,
    plot_per_cell_rmse,
    plot_per_cluster_pod_distributions,
    plot_per_cluster_reconstructions,
    plot_per_mode_learning_curves,
    plot_recon_distribution,
    plot_reconstructions,
    latent_tidiness,
    pod_fit,
    pod_predict,
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
    # shared - layer 2 (Bousquet latent-space deep dive)
    "compute_pod_time_coefficients",
    "partition_latent_2d",
    "per_mode_learning_accuracy",
    "plot_latent_2d",
    "plot_per_cluster_pod_distributions",
    "plot_per_cluster_reconstructions",
    "plot_per_mode_learning_curves",
    # shared - beta-VAE rendering primitives
    "plot_decoded_samples",
    "plot_latent_2d_with_sigma",
    "plot_latent_traversal",
    "latent_tidiness",
    # ae
    "plot_loss_curves",
    "reconstruct_split",
    # vae
    "compute_vae_diagnostics",
    "latent_traversal",
    "plot_loss_curves_vae",
    "reconstruct_split_vae",
    # gaussianity
    "plot_innovation_gaussianity",
    "plot_pairwise_gaussianity",
    # projection
    "pca_scores",
]
