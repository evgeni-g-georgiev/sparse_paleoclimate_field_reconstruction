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

    Note on ``fit_indices``
    -----------------------
    In v1 (Bousquet-style in-sample probe) pass ``np.arange(N_ages)``:
    the AE is also fit on every age, so the fair POD baseline is fit
    on the same data.

    Parameters
    ----------
    cube_z : (N_ages, 2, H, W) float
        Z-scored cube (output of :func:`paleoreco.data.apply_zscore`).
    fit_indices : int array
        Indices into the ``N_ages`` axis used for the SVD.
    mask : (H, W) bool
        ``safe_valid`` mask.
    max_k : int
        Largest latent dimension we'll ever want to evaluate. The
        returned basis can be sliced down to any ``k <= max_k``.
    random_state : int
        Passed to sklearn's TruncatedSVD for reproducibility.

    Returns
    -------
    dict with keys
        ``"mu"`` : (1, D_valid) array — fit-set mean over valid columns.
        ``"V_max"`` : (max_k, D_valid) array — top-``max_k`` right
            singular vectors (rows are POD modes).
        ``"fit_indices_used"`` : ndarray copy of ``fit_indices``.
        ``"keep_mask"`` : (2*H*W,) bool — column-keep mask used to
            extract valid columns from a flattened cube.
        ``"shape"`` : ``(2, H, W)`` — original spatial shape, used by
            :func:`pod_predict` to scatter reconstructions back.
    """
    from sklearn.decomposition import TruncatedSVD

    n_channels, H, W = cube_z.shape[1:]
    keep = _valid_column_mask(mask, n_channels=n_channels)
    X_all = cube_z.reshape(cube_z.shape[0], -1)[:, keep]
    X_fit = X_all[fit_indices]

    mu = X_fit.mean(axis=0, keepdims=True)  # (1, D_valid)
    X_fit_c = X_fit - mu

    # Fit once at max_k and slice for smaller k - randomised SVD is
    # much cheaper that way than separate fits per k.
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
    """Back-compat wrapper: per-``k`` test RMSE in z-score units.

    Thin wrapper around :func:`pod_fit` + :func:`pod_predict`. Kept
    for code that still wants the original convenience signature. New
    code should call ``pod_fit`` / ``pod_predict`` directly and use
    :func:`compute_E_d` as the headline metric.

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

    * The ratio is computed **per snapshot** and then averaged across
      snapshots - *not* a single global numerator/denominator ratio.
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
        Guards a snapshot with vanishing ``||u||^2`` (impossible in
        practice for our z-scored cube, but cheap to defend against).

    Returns
    -------
    float
        ``E_d`` - values near 1 mean near-perfect compression. Bousquet
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
    overlay — the headline AE-vs-POD comparison (Bousquet Fig 3).
    Secondary (right) panel: ``rmse_celsius`` vs latent dim - the
    same diagnostic in °C units, model curve only
    (POD °C-RMSE is omitted; the head-to-head story lives on the
    E_d panel).

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
