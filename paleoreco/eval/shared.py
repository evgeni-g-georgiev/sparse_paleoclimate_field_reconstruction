"""Model-family-agnostic evaluation primitives.

Everything in this module operates on numpy arrays of truth /
reconstruction pairs. There's no coupling to a specific model class
or training-loop history dict.

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
# Internal helper: mask-restricted column index for the POD design matrix.
# ---------------------------------------------------------------------------
def _valid_column_mask(mask: np.ndarray, n_channels: int = 2) -> np.ndarray:
    """Boolean column-keep mask for a flattened (N, C*H*W) design matrix.

    POD only operates on valid cells so the basis isn't polluted by
    zero-filled cells outside the mask. The two channels (mtco, mtwa)
    are stacked along the column axis; each contributes ``mask.sum()``
    kept columns, giving ``n_channels * mask.sum()`` columns in total.
    """
    return np.concatenate([mask.ravel()] * n_channels)


# ---------------------------------------------------------------------------
# Internal helper: z-score -> °C anomaly conversion.
# ---------------------------------------------------------------------------
def _zscore_to_anomaly(z: np.ndarray, zscore_stats: dict) -> np.ndarray:
    """Convert z-score data to °C anomaly (``x − mean``).

    Anomaly is what climatologists read on a map; absolute °C is
    visually dominated by the climatology (e.g. −40 °C over Antarctica).
    Because ``z = (x − mean) / std``, anomaly = ``z * std`` directly.

    Broadcasts ``std`` of shape ``(2, H, W)`` against ``z`` of any
    leading shape ``(..., 2, H, W)``.
    """
    return z * zscore_stats["std"][None]


# ---------------------------------------------------------------------------
# Reconstruction grid.
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
# Per-cell RMSE map.
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
# POD baseline + latent-dim sweep.
# ---------------------------------------------------------------------------
def pod_fit(
    cube_z: np.ndarray,
    fit_indices: np.ndarray,
    mask: np.ndarray,
    max_k: int,
    random_state: int = 0,
) -> dict:
    """Fit a truncated POD basis once, on a chosen set of ages.

    POD is the linear (orthogonal) baseline for the AE compressor. The
    basis is the top ``max_k`` right singular vectors of the centred
    fit-set design matrix; smaller ``k`` reconstructions are obtained
    by slicing this basis (no re-fit needed).

    Only valid cells participate so the basis isn't polluted by
    zero-filled masked cells. The two channels (mtco, mtwa) are stacked
    along the column axis so the SVD treats them on equal footing.

    Parameters
    ----------
    cube_z : (N_ages, 2, H, W) float
        Z-scored cube (output of :func:`paleoreco.data.apply_zscore`).
    fit_indices : int array
        Indices into the ``N_ages`` axis used for the SVD.
    mask : (H, W) bool
        ``safe_valid`` mask.
    max_k : int
        Largest latent dimension to be evaluated; the returned basis
        can be sliced down to any ``k <= max_k``.
    random_state : int
        Passed to sklearn's TruncatedSVD for reproducibility.

    Returns
    -------
    dict with keys:
        ``"mu"`` : (1, D_valid) array. Fit-set mean over valid columns.
        ``"V_max"`` : (max_k, D_valid) array. Top-``max_k`` right
            singular vectors (rows are POD modes).
        ``"fit_indices_used"`` : ndarray copy of ``fit_indices``.
        ``"keep_mask"`` : (2*H*W,) bool. Column-keep mask used to
            extract valid columns from a flattened cube.
        ``"shape"`` : ``(2, H, W)``. Original spatial shape, used by
            :func:`pod_predict` to scatter reconstructions back.
    """
    from sklearn.decomposition import TruncatedSVD

    n_channels, H, W = cube_z.shape[1:]
    keep = _valid_column_mask(mask, n_channels=n_channels)
    X_all = cube_z.reshape(cube_z.shape[0], -1)[:, keep]
    X_fit = X_all[fit_indices]

    mu = X_fit.mean(axis=0, keepdims=True)  # (1, D_valid)
    X_fit_c = X_fit - mu

    # Fit once at max_k and slice for smaller k; randomised SVD is
    # cheaper that way than separate fits per k.
    svd = TruncatedSVD(
        n_components=max_k, algorithm="randomized", random_state=random_state,
    )
    svd.fit(X_fit_c)

    return {
        "mu": mu,                                # (1, D_valid)
        "V_max": svd.components_,                # (max_k, D_valid)
        "fit_indices_used": np.asarray(fit_indices).copy(),
        "keep_mask": keep,                       # (2*H*W,)
        "shape": (n_channels, H, W),
    }


def pod_predict(
    cube_z: np.ndarray,
    eval_indices: np.ndarray,
    pod_basis: dict,
    k: int,
) -> np.ndarray:
    """Reconstruct ``cube_z[eval_indices]`` from a fitted POD basis at rank ``k``.

    The recipe: extract valid columns, centre on the fit-set mean,
    project onto the top ``k`` POD modes, back-project, add the mean
    back, and scatter into a full ``(N, 2, H, W)`` array with zero
    outside the mask. Outputs match the convention of an AE
    reconstruction (z-score units, zero on masked cells), so the same
    downstream metric / plot functions consume both.

    Parameters
    ----------
    cube_z : (N_ages, 2, H, W) float
        Z-scored cube. Only ``cube_z[eval_indices]`` is touched.
    eval_indices : int array
        Indices into the ``N_ages`` axis to reconstruct.
    pod_basis : dict
        Output of :func:`pod_fit`.
    k : int
        Number of POD modes to keep. Must satisfy ``1 <= k <= max_k``
        of the fitted basis.

    Returns
    -------
    np.ndarray of shape ``(len(eval_indices), 2, H, W)``.
        Reconstruction in z-score units, zero outside the mask.
    """
    V_max = pod_basis["V_max"]
    mu = pod_basis["mu"]
    keep = pod_basis["keep_mask"]
    n_channels, H, W = pod_basis["shape"]
    max_k = V_max.shape[0]
    if not (1 <= k <= max_k):
        raise ValueError(f"k={k} out of range (basis fit at max_k={max_k}).")

    Vk = V_max[:k]                                # (k, D_valid)
    X = cube_z.reshape(cube_z.shape[0], -1)[:, keep]
    X_eval = X[eval_indices]
    X_c = X_eval - mu
    X_hat_c = X_c @ Vk.T @ Vk                     # (N_eval, D_valid)
    X_hat = X_hat_c + mu                          # back to original mean

    # Scatter into full (N_eval, 2*H*W) with zero on masked cells, then
    # reshape to (N_eval, 2, H, W). Matches the AE output convention.
    n_eval = X_hat.shape[0]
    full = np.zeros((n_eval, n_channels * H * W), dtype=cube_z.dtype)
    full[:, keep] = X_hat.astype(cube_z.dtype, copy=False)
    return full.reshape(n_eval, n_channels, H, W)


def pod_test_rmse(
    cube_z: np.ndarray,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
    mask: np.ndarray,
    ks: Sequence[int],
    random_state: int = 0,
) -> np.ndarray:
    """Per-``k`` test RMSE in z-score units; convenience wrapper.

    Thin wrapper around :func:`pod_fit` + :func:`pod_predict`. The
    headline AE-vs-POD comparison uses :func:`compute_E_d`.

    Returns
    -------
    np.ndarray of shape ``(len(ks),)``
        Per-``k`` masked-RMSE in z-score units over ``test_indices``.
    """
    max_k = int(max(ks))
    basis = pod_fit(cube_z, fit_indices, mask, max_k, random_state=random_state)
    truth = cube_z[test_indices]
    mask_3d = mask[None, None]                    # broadcast over (N, C, H, W)
    rmses = []
    for k in ks:
        pred = pod_predict(cube_z, test_indices, basis, k)
        sq = (pred - truth) ** 2 * mask_3d
        rmses.append(float(np.sqrt(sq.sum() / (truth.shape[0] * 2 * mask.sum()))))
    return np.array(rmses, dtype=np.float64)


# ---------------------------------------------------------------------------
# Bousquet's E_d compression-quality metric.
# ---------------------------------------------------------------------------
def compute_E_d(
    truth_z: np.ndarray,
    pred_z: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Bousquet 2025 Eq. 14: fraction of variance captured, averaged per snapshot.

    .. math::

        E_d = 1 - \\frac{1}{N_t} \\sum_t
                  \\frac{\\sum_{c,(i,j) \\in \\text{mask}} (u_t - \\hat u_t)^2}
                       {\\sum_{c,(i,j) \\in \\text{mask}} u_t^2}.

    Two key points:

    * The ratio is computed **per snapshot** then averaged across
      snapshots, *not* a single global numerator/denominator ratio.
      This matches Bousquet exactly and weighs every snapshot equally
      regardless of its magnitude.
    * The sum runs over valid cells across **both** channels, so the
      mask broadcasts to ``(N, 2, H, W)``.

    Parameters
    ----------
    truth_z, pred_z : (N, 2, H, W) float
        Truth and reconstruction in z-score units.
    mask : (H, W) bool
        ``safe_valid`` mask.
    eps : float
        Guards against a snapshot with vanishing ``||u||^2`` (cheap
        defence; not expected on z-scored data).

    Returns
    -------
    float
        ``E_d``: values near 1 mean near-perfect compression. Bousquet
        reports values around 0.99+ for periodic flows, dropping to
        ~0.4-0.8 for the turbulent von Kármán case.
    """
    mask_b = mask[None, None].astype(bool)        # (1, 1, H, W)
    sq_err = ((pred_z - truth_z) ** 2) * mask_b   # zeros outside mask
    sq_truth = (truth_z ** 2) * mask_b

    # Per-snapshot spatial sums over (channel, valid-cell). Shape: (N,).
    num = sq_err.reshape(sq_err.shape[0], -1).sum(axis=1)
    den = sq_truth.reshape(sq_truth.shape[0], -1).sum(axis=1)

    # Per-snapshot ratio, then mean over snapshots.
    ratio = num / np.maximum(den, eps)
    return float(1.0 - ratio.mean())


def plot_latent_sweep(
    latent_dims: Sequence[int],
    model_E_d: Sequence[float],
    pod_E_d: Sequence[float] | None = None,
    model_rmse_celsius: Sequence[float] | None = None,
    model_label: str = "model",
    save_path: str | None = None,
) -> plt.Figure:
    """Bousquet-style latent-dim sweep.

    Primary (left) panel: ``E_d`` vs latent dim with optional POD
    overlay (the headline AE-vs-POD comparison, Bousquet Fig 3).
    Secondary (right) panel: ``rmse_celsius`` vs latent dim, model
    curve only (POD °C-RMSE is omitted; the head-to-head story lives
    on the E_d panel).

    Log-2 x-axis with explicit tick labels at the sweep points.
    ``model_label`` is shown in the legend; defaults to ``"model"``.
    """
    has_celsius = model_rmse_celsius is not None
    n_panels = 2 if has_celsius else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels + 0.5, 4),
        squeeze=False, constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(latent_dims, model_E_d, "o-", color="C0", label=model_label, lw=1.6)
    if pod_E_d is not None:
        ax.plot(latent_dims, pod_E_d, "s--", color="C3", alpha=0.7, label="POD truncation")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(latent_dims))
    ax.set_xticklabels([str(d) for d in latent_dims])
    ax.set_xlabel("latent dim")
    ax.set_ylabel(r"$E_d$ (Bousquet)")
    ax.set_title("Latent-dim sweep — $E_d$ (head-to-head with POD)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if has_celsius:
        ax = axes[0, 1]
        ax.plot(latent_dims, model_rmse_celsius, "o-", color="C2", lw=1.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(latent_dims))
        ax.set_xticklabels([str(d) for d in latent_dims])
        ax.set_xlabel("latent dim")
        ax.set_ylabel("RMSE (°C)")
        ax.set_title("Latent-dim sweep — °C (secondary, human-readable)")
        ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Bousquet Layer-2 primitives.
# ---------------------------------------------------------------------------
# These functions implement the latent-space analysis Bousquet performs
# at low latent dimension (his §3.2). Kept model-agnostic so the same
# code can drive any model with a 2D latent space.
# ---------------------------------------------------------------------------
def compute_pod_time_coefficients(
    cube_z: np.ndarray,
    mask: np.ndarray,
    pod_basis: dict,
) -> np.ndarray:
    """Bousquet's POD time-coefficients ``a_k(t)`` (his Eq. 22).

    Each snapshot is centred on the POD fit-set mean and projected onto
    each POD mode, yielding a single scalar coefficient per (snapshot,
    mode). The basis is assumed orthonormal (sklearn's ``TruncatedSVD``
    components are), so the projection is a simple inner product.

    Parameters
    ----------
    cube_z : (N_ages, 2, H, W) float
        Z-scored cube; every age is projected (no slicing) so the full
        ``a_k(t)`` time series is available.
    mask : (H, W) bool
        ``safe_valid`` mask. Only valid cells participate, matching the
        column convention used in :func:`pod_fit`.
    pod_basis : dict
        Output of :func:`pod_fit`. Provides ``mu``, ``V_max``,
        ``keep_mask``.

    Returns
    -------
    np.ndarray of shape ``(N_ages, max_k)``
        ``a_k(t)`` for each (snapshot, POD-mode index) pair.
    """
    V_max = pod_basis["V_max"]                    # (max_k, D_valid)
    mu = pod_basis["mu"]                          # (1, D_valid)
    keep = pod_basis["keep_mask"]

    X = cube_z.reshape(cube_z.shape[0], -1)[:, keep]
    X_c = X - mu
    # (N_ages, D_valid) @ (D_valid, max_k) = (N_ages, max_k).
    return X_c @ V_max.T


def per_mode_learning_accuracy(
    pod_a_k: np.ndarray,
    ae_a_k_per_epoch: np.ndarray,
) -> np.ndarray:
    """Bousquet's per-mode learning accuracy ``e_k`` (his Eq. 23).

    For each (epoch, POD-mode) pair:

    .. math::

        e_k = \\max\\left(0,\\ 1 -
              \\frac{\\sum_t (a_k(t) - \\tilde a_k(t))^2}
                   {\\sum_t a_k(t)^2}\\right)

    where ``a_k`` is the true POD coefficient and ``\\tilde a_k`` is the
    coefficient obtained by projecting the **model's reconstruction**
    onto the same POD mode at that epoch. The ``max(0, ·)`` clip keeps
    pre-training noise from producing nonsense negative accuracies.

    Parameters
    ----------
    pod_a_k : (N_ages, max_k) float
        Output of :func:`compute_pod_time_coefficients` on the truth.
    ae_a_k_per_epoch : (n_epochs, N_ages, max_k) float
        Per-epoch model-reconstruction coefficients, obtained by
        projecting the model's reconstruction at each epoch through
        :func:`compute_pod_time_coefficients`.

    Returns
    -------
    np.ndarray of shape ``(n_epochs, max_k)``
        ``e_k`` per epoch and per POD mode.
    """
    n_epochs = ae_a_k_per_epoch.shape[0]
    den = (pod_a_k ** 2).sum(axis=0)              # (max_k,)
    out = np.empty((n_epochs, pod_a_k.shape[1]), dtype=np.float64)
    for e in range(n_epochs):
        diff = pod_a_k - ae_a_k_per_epoch[e]
        num = (diff ** 2).sum(axis=0)             # (max_k,)
        out[e] = np.maximum(0.0, 1.0 - num / np.maximum(den, 1e-12))
    return out


def plot_per_mode_learning_curves(
    per_mode_acc: np.ndarray,
    ks_to_show: Sequence[int] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bousquet Fig 11 / 14: per-POD-mode learning curves vs training epoch.

    The x-axis is log-spaced training epoch (Bousquet's convention); the
    y-axis is the per-mode learning accuracy ``e_k`` from
    :func:`per_mode_learning_accuracy`. One coloured line per mode tells
    you which modes the model picks up first.

    ``ks_to_show`` defaults to all modes the array contains. Pass a
    short list to highlight specific modes.
    """
    n_epochs, max_k = per_mode_acc.shape
    if ks_to_show is None:
        ks_to_show = list(range(max_k))
    epochs = np.arange(1, n_epochs + 1)            # log-x starts at 1.

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for k in ks_to_show:
        if not (0 <= k < max_k):
            continue
        color = cmap(k / max(max_k - 1, 1))
        ax.plot(epochs, per_mode_acc[:, k], lw=1.4, color=color, label=f"k={k}")
    ax.set_xscale("log")
    ax.set_xlabel("training epoch (log scale)")
    ax.set_ylabel(r"$e_k$  (per-mode learning accuracy)")
    ax.set_title("Per-POD-mode learning curves")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower right")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def plot_latent_2d(
    latents: np.ndarray,
    color_values: np.ndarray,
    color_label: str,
    save_path: str | None = None,
    title: str = "Latent space (2D)",
    discrete: bool | None = None,
) -> plt.Figure:
    """Bousquet Fig 9 / 10: 2D scatter of the AE's ``(Z_1, Z_2)`` latent.

    ``color_values`` is typically the integer D-O event label (output
    of :func:`paleoreco.splits.assign_event_label`) or the age in yr
    BP. ``discrete`` controls colour-bar style; default auto-detects
    based on whether ``color_values`` is integer-typed with a small
    cardinality.
    """
    if latents.ndim != 2 or latents.shape[1] != 2:
        raise ValueError(
            f"plot_latent_2d expects (N, 2) latents, got {latents.shape}"
        )

    if discrete is None:
        is_int = np.issubdtype(np.asarray(color_values).dtype, np.integer)
        discrete = bool(is_int and len(np.unique(color_values)) <= 16)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    if discrete:
        # Use a categorical-ish colormap so values like 0..8 read as
        # event indices rather than a continuum.
        levels = np.unique(color_values)
        cmap = plt.get_cmap("tab10", max(len(levels), 1))
        sc = ax.scatter(
            latents[:, 0], latents[:, 1],
            c=color_values, cmap=cmap, s=14, alpha=0.85, edgecolor="none",
        )
        cbar = plt.colorbar(sc, ax=ax, ticks=levels)
        cbar.set_label(color_label)
    else:
        sc = ax.scatter(
            latents[:, 0], latents[:, 1],
            c=color_values, cmap="viridis", s=14, alpha=0.85, edgecolor="none",
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_label)
    ax.set_xlabel(r"$Z_1$")
    ax.set_ylabel(r"$Z_2$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def partition_latent_2d(
    latents: np.ndarray,
    criterion: str | int,
    random_state: int = 0,
) -> np.ndarray:
    """Partition snapshots in latent space; returns integer cluster labels.

    Two criteria, both used by Bousquet:

    * ``criterion="z2_sign"``: binary split by the sign of ``Z_2``.
      Cheap, interpretable, and a useful first cut whenever the 2D
      latent looks two-lobed (Bousquet Fig 13). Labels: 0 for
      ``Z_2 < 0``, 1 for ``Z_2 >= 0``.
    * ``criterion=int n``: KMeans with ``n`` clusters
      (Bousquet's Fig 15 / 20 partition). ``random_state`` is fixed so
      cluster IDs are reproducible across runs.
    """
    if latents.ndim != 2 or latents.shape[1] != 2:
        raise ValueError(
            f"partition_latent_2d expects (N, 2) latents, got {latents.shape}"
        )

    if criterion == "z2_sign":
        return (latents[:, 1] >= 0).astype(np.int64)
    if isinstance(criterion, int):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=criterion, n_init=10, random_state=random_state)
        return km.fit_predict(latents).astype(np.int64)
    raise ValueError(
        f"unknown criterion {criterion!r}; expected 'z2_sign' or an int."
    )


def plot_per_cluster_pod_distributions(
    pod_a_k: np.ndarray,
    cluster_labels: np.ndarray,
    ks_to_show: Sequence[int],
    save_path: str | None = None,
) -> plt.Figure:
    """Bousquet Fig 15: per-cluster densities of POD time-coefficients.

    One row per POD mode in ``ks_to_show``, one panel per row showing
    the overlaid per-cluster densities of ``a_k(t)``. This is the
    Bousquet-style diagnostic for *what each cluster encodes*: if
    cluster 0 concentrates the negative tail of ``a_1`` and cluster 1
    the positive tail, the latent has split along the ``a_1`` axis.
    """
    if pod_a_k.shape[0] != cluster_labels.shape[0]:
        raise ValueError(
            "pod_a_k and cluster_labels must have the same first axis "
            f"(got {pod_a_k.shape[0]} vs {cluster_labels.shape[0]})."
        )
    levels = np.unique(cluster_labels)
    n_rows = len(ks_to_show)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(7.5, 2.0 * n_rows + 1),
        squeeze=False, constrained_layout=True,
    )

    for row, k in enumerate(ks_to_show):
        ax = axes[row, 0]
        a_k = pod_a_k[:, k]
        lo, hi = float(a_k.min()), float(a_k.max())
        bins = np.linspace(lo, hi, 50)
        for lev in levels:
            sub = a_k[cluster_labels == lev]
            if sub.size == 0:
                continue
            ax.hist(
                sub, bins=bins, alpha=0.55, density=True,
                label=f"cluster {int(lev)} (n={sub.size})",
            )
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel(rf"$a_{{{k}}}(t)$")
        ax.set_ylabel("density")
        ax.set_title(rf"POD mode $k={k}$")
        ax.legend(fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


def plot_per_cluster_reconstructions(
    cube: np.ndarray,
    cluster_labels: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    mask: np.ndarray,
    var_names: Sequence[str] = ("mtco", "mtwa"),
    unit_label: str = "°C anomaly",
    save_path: str | None = None,
) -> plt.Figure:
    """Bousquet Fig 20: per-cluster mean physical-space field.

    For each cluster, the snapshots assigned to that cluster are
    averaged and plotted as a map per channel. Useful for asking *what
    the latent partition actually corresponds to in physical space*:
    e.g. cluster 0 = "warm interstadial composite", cluster 1 = "cold
    stadial composite".

    ``cube`` is plotted as-is: the caller decides which units. Typical
    inputs are ``cube_z * std`` (°C anomaly) or ``cube_z * std + mean``
    (absolute °C). Cells outside ``mask`` are rendered NaN.

    Layout: one row per cluster, one column per channel.
    """
    if cube.shape[0] != cluster_labels.shape[0]:
        raise ValueError(
            "cube and cluster_labels must have the same first axis "
            f"(got {cube.shape[0]} vs {cluster_labels.shape[0]})."
        )
    levels = np.unique(cluster_labels)
    n_channels = cube.shape[1]
    if len(var_names) != n_channels:
        raise ValueError("var_names must match cube channel count.")

    fig, axes = plt.subplots(
        len(levels), n_channels,
        figsize=(4.5 * n_channels + 0.5, 2.6 * len(levels) + 0.5),
        squeeze=False, constrained_layout=True,
    )
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Symmetric colour scale per channel from the per-cluster means
    # (so warm/cold composites are directly comparable across rows).
    per_cluster_means = []
    for lev in levels:
        sub = cube[cluster_labels == lev]
        if sub.size == 0:
            mean_field = np.full_like(cube[0], np.nan, dtype=np.float64)
        else:
            mean_field = sub.mean(axis=0)
        mean_field = np.where(mask[None], mean_field, np.nan)
        per_cluster_means.append(mean_field)
    stacked = np.stack(per_cluster_means)  # (n_levels, C, H, W)
    vmax = [
        float(np.nanpercentile(np.abs(stacked[:, c]), 99))
        for c in range(n_channels)
    ]

    for r, lev in enumerate(levels):
        n_in_cluster = int((cluster_labels == lev).sum())
        for c in range(n_channels):
            ax = axes[r, c]
            im = ax.imshow(
                per_cluster_means[r][c], origin="lower", extent=extent,
                cmap="RdBu_r", vmin=-vmax[c], vmax=vmax[c], aspect="auto",
            )
            if r == 0:
                ax.set_title(f"{var_names[c]} ({unit_label})")
            if c == 0:
                ax.set_ylabel(f"cluster {int(lev)}\n(n={n_in_cluster})")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.8)

    fig.suptitle("Per-cluster mean field (Bousquet Fig 20-style)", fontsize=11)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig


# ---------------------------------------------------------------------------
# Reconstruction distribution.
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
