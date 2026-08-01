"""Principal-component projections of a field stack, and plots of their distributions.

Each principal component is a 1-D linear projection of the states. By the
Cramer-Wold device a distribution is multivariate Gaussian only if every such
projection is univariate Gaussian, so a visibly non-Gaussian leading component
falsifies a Gaussian model of the field distribution.

Comparing two stacks means fitting each in its own basis, so component ``k`` is a
different spatial pattern in each: what the plots compare is the shape of a score
distribution, not one quantity measured twice.
:func:`pc_pattern_correlation` says how far apart the two patterns are.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA


def pca_scores(fields: np.ndarray, n_components: int) -> dict:
    """Leading principal-component scores of a stack of fields.

    ``fields`` is ``(n_states, ...)``; the trailing axes (``(H, W)`` or
    ``(C, H, W)``) are flattened, so one call serves single- and multi-channel
    state vectors. The stack is demeaned across states into anomalies before
    projection. Cells are not area-weighted, so the high-latitude variance that
    carries the D-O signal keeps its weight.

    Returns ``scores`` ``(n_states, n_components)``, ``explained_variance_ratio``,
    ``components`` and the removed ``mean_field``, the last two in the original
    field shape. sklearn picks a randomized solver at these sizes, so the seed is
    fixed to keep repeated fits identical.
    """
    n = fields.shape[0]
    flat = np.asarray(fields, dtype=np.float64).reshape(n, -1)
    mean = flat.mean(axis=0)
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(flat - mean)
    return {
        "scores": scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_.reshape(n_components, *fields.shape[1:]),
        "mean_field": mean.reshape(fields.shape[1:]),
    }


def orient_pcs(result: dict) -> dict:
    """Sign-fix a :func:`pca_scores` result so a positive score means a warmer state.

    An eigenvector is defined only up to sign, so two stacks can come out mirrored
    and their score histograms then compare left-to-right rather than shape to
    shape. Flipping each component to a positive spatial mean is the same rule as
    making its score correlate positively with the field's global mean, since
    ``cov(x . v_k, mean(x)) = lambda_k mean(v_k)``.

    Adds ``flipped``, a per-component bool, so a caller can report what moved.
    """
    components = np.asarray(result["components"])
    means = components.reshape(len(components), -1).mean(axis=1)
    flipped = means < 0.0
    sign = np.where(flipped, -1.0, 1.0)
    return {
        **result,
        "scores": result["scores"] * sign[None, :],
        "components": components * sign.reshape(-1, *([1] * (components.ndim - 1))),
        "flipped": flipped,
    }


def pc_pattern_correlation(a: dict, b: dict, n_components: int = 4) -> np.ndarray:
    """Spatial correlation between two results' matching components, ``(n_components,)``.

    Only defined when both were fitted on the same grid, so mismatched component
    shapes are an error rather than a broadcast.
    """
    ca, cb = np.asarray(a["components"]), np.asarray(b["components"])
    if ca.shape[1:] != cb.shape[1:]:
        raise ValueError(f"components live on different grids: {ca.shape[1:]} against "
                         f"{cb.shape[1:]}")
    return np.array([np.corrcoef(ca[k].ravel(), cb[k].ravel())[0, 1]
                     for k in range(n_components)])


# ---------------------------------------------------------------------------
# Score-distribution plots.
# ---------------------------------------------------------------------------
def _standardised(result: dict, k: int) -> np.ndarray:
    """Component ``k``'s scores rescaled to unit variance, comparable across stacks."""
    z = np.asarray(result["scores"][:, k], dtype=float)
    return z / z.std()


def _summary(z: np.ndarray) -> str:
    """One-line moment summary for a panel title."""
    return (f"n={z.size}  mean={z.mean():.3f}  sd={z.std():.3f}  "
            f"skew={stats.skew(z):.3f}  exkurt={stats.kurtosis(z):.3f}")


def _grid(results: dict[str, dict], n_components: int, height: float):
    """Component-by-stack panel grid, rows labelled by component."""
    labels = list(results)
    fig, axes = plt.subplots(n_components, len(labels),
                             figsize=(5.5 * len(labels), height * n_components),
                             squeeze=False, constrained_layout=True)
    return fig, axes, labels


def plot_pc_score_distributions(
    results: dict[str, dict],
    n_components: int = 4,
    bins: int = 30,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Standardised PC score histograms against N(0,1): one row per component, one
    column per stack.

    ``results`` maps a label to a :func:`pca_scores` result, ideally through
    :func:`orient_pcs` so the two columns share a sign convention. Column order
    follows insertion order. Each panel's x-label carries that component's share of
    the stack's variance, which is what says whether a visibly non-Gaussian
    component matters.
    """
    fig, axes, labels = _grid(results, n_components, 2.6)
    grid = np.linspace(-5, 5, 400)

    for k in range(n_components):
        for col, label in enumerate(labels):
            ax = axes[k, col]
            z = _standardised(results[label], k)
            ax.hist(z, bins=bins, density=True, color="steelblue", alpha=0.7)
            ax.plot(grid, stats.norm.pdf(grid), color="black", lw=1.5, label="N(0,1)")
            ax.set_xlim(-5, 5)
            evr = float(results[label]["explained_variance_ratio"][k])
            ax.set_xlabel(f"PC{k + 1} score (standardised)   explained variance {evr:.3f}",
                          fontsize=9)
            ax.set_title(f"{label}  PC{k + 1}\n{_summary(z)}", fontsize=9)
            if col == 0:
                ax.set_ylabel("density")
            ax.legend(loc="upper right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_pc_score_qq(
    results: dict[str, dict],
    n_components: int = 4,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Normal QQ of each standardised PC score, on the layout of
    :func:`plot_pc_score_distributions`.

    A QQ resolves the tails that a histogram flattens, so the two read together.
    """
    fig, axes, labels = _grid(results, n_components, 2.9)

    for k in range(n_components):
        for col, label in enumerate(labels):
            ax = axes[k, col]
            stats.probplot(_standardised(results[label], k), dist="norm", plot=ax)
            ax.set_title(f"{label}  PC{k + 1}: normal QQ", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig
