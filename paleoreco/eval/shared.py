"""Model-family-agnostic evaluation primitives.

Everything in this module operates on numpy arrays of truth /
reconstruction pairs. There's no coupling to a specific model class
or training-loop history dict. The same functions will be used to
evaluate downstream models.

Conventions
-----------
* ``truth_z`` and ``pred_z`` are ``(N, 2, H, W)`` float arrays in
  **z-score units** (output of :func:`paleoreco.data.apply_zscore` or a
  decoder operating in that space).
* ``zscore_stats`` is the dict returned by
  :func:`paleoreco.data.compute_zscore_stats`, used to invert the
  z-score back to °C anomaly for plotting.
* ``mask`` is the ``safe_valid`` boolean mask of shape ``(H, W)``.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helper: z-score -> °C anomaly conversion.
# ---------------------------------------------------------------------------
def _zscore_to_anomaly(z: np.ndarray, zscore_stats: dict) -> np.ndarray:
    """Convert z-score data to °C anomaly (``x − mean``).

    Anomaly is what climatologists read on a map - absolute °C is
    dominated visually by the climatology (e.g. −40 °C over Antarctica).
    Because ``z = (x − mean) / std``, anomaly = ``z * std`` directly;
    no mean needed.

    Broadcasts ``std`` of shape ``(2, H, W)`` against ``z`` of any
    leading shape ``(..., 2, H, W)``.
    """
    return z * zscore_stats["std"][None]


# ---------------------------------------------------------------------------
# Reconstruction grid (artefact ii).
# ---------------------------------------------------------------------------
def plot_reconstructions(
    truth_z: np.ndarray,
    pred_z: np.ndarray,
    zscore_stats: dict,
    ages: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sample_indices: Sequence[int] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """5 rows × 6 columns: each row is one sample; columns are
    ``(truth_mtco, recon_mtco, err_mtco, truth_mtwa, recon_mtwa, err_mtwa)``
    in **°C anomaly** units.

    Truth and recon for the same channel share a symmetric diverging
    colour scale (99th percentile of ``|truth|`` over the input set).
    The error column has its own scale (99th percentile of ``|err|``).

    ``sample_indices`` selects which rows of ``truth_z`` to plot.
    Default: 5 evenly-spaced indices spanning the input set.
    """
    if sample_indices is None:
        sample_indices = np.linspace(0, len(truth_z) - 1, 5).astype(int).tolist()
    assert len(sample_indices) == 5, "expected 5 sample indices"

    truth_a = _zscore_to_anomaly(truth_z, zscore_stats)
    pred_a = _zscore_to_anomaly(pred_z, zscore_stats)
    err_a = pred_a - truth_a

    # 99th-percentile clipping so a single polar outlier doesn't squash
    # the rest of the figure into a single midtone.
    vmax_data = [float(np.nanpercentile(np.abs(truth_a[:, c]), 99)) for c in range(2)]
    vmax_err = [float(np.nanpercentile(np.abs(err_a[:, c]), 99)) for c in range(2)]

    fig, axes = plt.subplots(5, 6, figsize=(14, 11), constrained_layout=True)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    col_titles = [
        "truth mtco", "recon mtco", "err mtco",
        "truth mtwa", "recon mtwa", "err mtwa",
    ]

    # Cache one image per logical colourbar group: (mtco_data, mtco_err, mtwa_data, mtwa_err).
    im_for_cbar = [None, None, None, None]

    for row_i, idx in enumerate(sample_indices):
        for c in range(2):
            base = c * 3
            v_data = vmax_data[c]
            v_err = vmax_err[c]

            im0 = axes[row_i, base].imshow(
                truth_a[idx, c], origin="lower", extent=extent,
                cmap="RdBu_r", vmin=-v_data, vmax=v_data, aspect="auto",
            )
            axes[row_i, base + 1].imshow(
                pred_a[idx, c], origin="lower", extent=extent,
                cmap="RdBu_r", vmin=-v_data, vmax=v_data, aspect="auto",
            )
            im2 = axes[row_i, base + 2].imshow(
                err_a[idx, c], origin="lower", extent=extent,
                cmap="PuOr_r", vmin=-v_err, vmax=v_err, aspect="auto",
            )
            im_for_cbar[c * 2] = im0
            im_for_cbar[c * 2 + 1] = im2

        axes[row_i, 0].set_ylabel(f"age {int(ages[idx])} BP", fontsize=10)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # One shared colourbar per logical group: truth+recon share, err standalone.
    fig.colorbar(
        im_for_cbar[0], ax=axes[:, 0:2].ravel().tolist(),
        shrink=0.55, location="right", label="mtco anomaly (°C)",
    )
    fig.colorbar(
        im_for_cbar[1], ax=axes[:, 2].tolist(),
        shrink=0.55, location="right", label="mtco err (°C)",
    )
    fig.colorbar(
        im_for_cbar[2], ax=axes[:, 3:5].ravel().tolist(),
        shrink=0.55, location="right", label="mtwa anomaly (°C)",
    )
    fig.colorbar(
        im_for_cbar[3], ax=axes[:, 5].tolist(),
        shrink=0.55, location="right", label="mtwa err (°C)",
    )

    fig.suptitle("Reconstructions — °C anomaly", fontsize=12)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Per-cell RMSE map (artefact iii).
# ---------------------------------------------------------------------------
def per_cell_rmse_celsius(
    truth_z: np.ndarray,
    pred_z: np.ndarray,
    zscore_stats: dict,
    mask: np.ndarray,
) -> np.ndarray:
    """Per-cell RMSE in °C. Returns ``(2, H, W)`` with NaN outside the mask.

    The z-score error ``(pred - truth)`` multiplied by the per-cell
    ``std`` is the °C error. Squaring, averaging across the samples,
    and taking sqrt gives per-cell °C RMSE.
    """
    err_c = (pred_z - truth_z) * zscore_stats["std"][None]
    rmse_c = np.sqrt((err_c ** 2).mean(axis=0))
    return np.where(mask[None], rmse_c, np.nan)


def plot_per_cell_rmse(
    rmse_celsius: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel: per-cell RMSE for mtco and mtwa.

    Shared colour scale across the two channels so the two maps are
    directly comparable. Range is ``[0, P99]`` where ``P99`` is the
    99th percentile of the higher channel's RMSE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    vmax = float(np.nanpercentile(rmse_celsius, 99))

    for c, name in enumerate(("mtco", "mtwa")):
        ax = axes[c]
        im = ax.imshow(
            rmse_celsius[c], origin="lower", extent=extent,
            cmap="magma", vmin=0, vmax=vmax, aspect="auto",
        )
        ax.set_title(f"{name} per-cell test RMSE (°C)")
        ax.set_xlabel("longitude (°)")
        ax.set_ylabel("latitude (°)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# POD baseline + latent-dim sweep (artefact iv).
# ---------------------------------------------------------------------------
def pod_test_rmse(
    cube_z: np.ndarray,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
    mask: np.ndarray,
    ks: Sequence[int],
    random_state: int = 0,
) -> np.ndarray:
    """Best linear k-mode reconstruction RMSE on the test split.

    The POD basis is the truncated SVD of the **fit** data, centred on
    its own mean. The k-mode reconstruction error on the test split is
    the fair "linear bound" for a model with ``k`` latent dimensions —
    a non-linear AE at the same ``latent_dim`` should equal or beat it.

    Note on ``fit_indices`` — fair-data-budget convention
    -----------------------------------------------------
    The trained AE consumes both the train ages (gradient updates) and
    the val ages (early-stopping / best-epoch selection) before being
    evaluated on test. POD has no model-selection step that needs a
    held-out val, so for an apples-to-apples comparison we fit the SVD
    on ``train ∪ val`` - the same total data budget the AE had access
    to before test eval. Pass ``np.concatenate([split["train"],
    split["val"]])`` from the caller. Using ``split["train"]`` alone
    would unfairly handicap POD by ~5% fewer samples, and the missing
    samples would be a single D-O event adjacent to the test event.

    Parameters
    ----------
    cube_z : (N_ages, 2, H, W) float
        Z-scored cube (output of :func:`paleoreco.data.apply_zscore`).
    fit_indices, test_indices : int arrays
        Indices into the ``N_ages`` axis. ``fit_indices`` is what the
        SVD is computed on; ``test_indices`` is the held-out slice the
        RMSE is reported on.
    mask : (H, W) bool
        ``safe_valid`` mask. Only valid cells participate in the SVD
        so the basis isn't polluted by zero-filled cells.
    ks : sequence of int
        Latent dimensions to evaluate, e.g. ``[2, 4, 8, 16, 32, 64]``.
    random_state : int
        Passed to sklearn's TruncatedSVD for reproducibility.

    Returns
    -------
    np.ndarray of shape ``(len(ks),)``
        Test RMSE in z-score units, one per ``k``.
    """
    from sklearn.decomposition import TruncatedSVD

    # Flatten samples to (N, 2*H*W), then restrict columns to valid cells
    # (across both channels). The "* 2" stacks both channels along the
    # column axis - the SVD treats them on equal footing.
    keep = np.concatenate([mask.ravel(), mask.ravel()])
    X_all = cube_z.reshape(cube_z.shape[0], -1)[:, keep]

    X_fit = X_all[fit_indices]
    X_test = X_all[test_indices]

    # Centre on the fit-set mean. POD requires zero-mean data; using the
    # fit-set mean (not the global mean) keeps the test split honest.
    mu = X_fit.mean(axis=0, keepdims=True)
    X_fit_c = X_fit - mu
    X_test_c = X_test - mu

    # Fit once at max(ks) and slice for smaller k - randomised SVD is
    # much cheaper that way than separate fits per k.
    max_k = int(max(ks))
    svd = TruncatedSVD(n_components=max_k, algorithm="randomized", random_state=random_state)
    svd.fit(X_fit_c)
    V = svd.components_  # (max_k, D)

    rmses = []
    for k in ks:
        Vk = V[:k]
        # Project test data onto top-k modes, then back: X̂ = X V_k^T V_k.
        X_hat = X_test_c @ Vk.T @ Vk
        rmses.append(float(np.sqrt(np.mean((X_test_c - X_hat) ** 2))))
    return np.array(rmses, dtype=np.float64)


def plot_latent_sweep(
    latent_dims: Sequence[int],
    model_rmse_z: Sequence[float],
    pod_rmse_z: Sequence[float] | None = None,
    model_rmse_celsius: Sequence[float] | None = None,
    model_label: str = "model",
    save_path: str | None = None,
) -> plt.Figure:
    """Sweep curve. Left: z-score unit RMSE with optional POD overlay.
    Right (if ``model_rmse_celsius`` given): the same curve in °C units.

    Log-2 x-axis with explicit tick labels at the sweep points.
    ``model_label`` is shown in the legend - defaults to ``"model"`` so
    this plotter can be reused by AE, latent diffusion, etc.
    """
    has_celsius = model_rmse_celsius is not None
    n_panels = 2 if has_celsius else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels + 0.5, 4),
        squeeze=False, constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(latent_dims, model_rmse_z, "o-", color="C0", label=model_label, lw=1.6)
    if pod_rmse_z is not None:
        ax.plot(latent_dims, pod_rmse_z, "s--", color="C3", alpha=0.7, label="POD truncation")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(latent_dims))
    ax.set_xticklabels([str(d) for d in latent_dims])
    ax.set_xlabel("latent dim")
    ax.set_ylabel("test RMSE (z-score units)")
    ax.set_title("Latent-dim sweep — z-score (head-to-head with POD)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if has_celsius:
        ax = axes[0, 1]
        ax.plot(latent_dims, model_rmse_celsius, "o-", color="C2", lw=1.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(latent_dims))
        ax.set_xticklabels([str(d) for d in latent_dims])
        ax.set_xlabel("latent dim")
        ax.set_ylabel("test RMSE (°C)")
        ax.set_title("Latent-dim sweep — °C")
        ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Reconstruction distribution (artefact v).
# ---------------------------------------------------------------------------
def plot_recon_distribution(
    truth_z: np.ndarray,
    pred_z: np.ndarray,
    zscore_stats: dict,
    mask: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Overlaid density histograms (truth vs reconstruction) per channel,
    in °C anomaly units, over valid cells only.

    Diagnostic for regression-to-the-mean: a bottleneck too tight to
    encode the variance shows up as a noticeably narrower reconstruction
    histogram than the truth histogram.
    """
    truth_a = _zscore_to_anomaly(truth_z, zscore_stats)
    pred_a = _zscore_to_anomaly(pred_z, zscore_stats)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for c, name in enumerate(("mtco", "mtwa")):
        t_flat = truth_a[:, c][:, mask].ravel()
        p_flat = pred_a[:, c][:, mask].ravel()
        lo = float(min(t_flat.min(), p_flat.min()))
        hi = float(max(t_flat.max(), p_flat.max()))
        bins = np.linspace(lo, hi, 60)

        ax = axes[c]
        ax.hist(t_flat, bins=bins, alpha=0.55, density=True, label="truth")
        ax.hist(p_flat, bins=bins, alpha=0.55, density=True, label="reconstruction")
        ax.axvline(0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"{name} anomaly (°C)")
        ax.set_ylabel("density")
        ax.set_title(f"{name} — truth vs reconstruction (valid cells)")
        ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig
