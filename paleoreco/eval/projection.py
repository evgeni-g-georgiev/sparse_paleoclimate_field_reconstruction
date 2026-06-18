"""Principal-component projections of a field stack.

Each principal component is a 1-D linear projection of the states. By the
Cramer-Wold device a distribution is multivariate Gaussian only if every such
projection is univariate Gaussian, so a visibly non-Gaussian leading component
falsifies a Gaussian model of the field distribution.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_scores(fields: np.ndarray, n_components: int) -> dict:
    """Leading principal-component scores of a stack of fields.

    ``fields`` is ``(n_states, ...)``; the trailing axes (``(H, W)`` or
    ``(C, H, W)``) are flattened, so one call serves single- and multi-channel
    state vectors. The stack is demeaned across states into anomalies before
    projection. Cells are not area-weighted, so the high-latitude variance that
    carries the D-O signal keeps its weight.

    Returns ``scores`` ``(n_states, n_components)``, ``explained_variance_ratio``,
    and the removed ``mean_field`` in the original field shape.
    """
    n = fields.shape[0]
    flat = np.asarray(fields, dtype=np.float64).reshape(n, -1)
    mean = flat.mean(axis=0)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(flat - mean)
    return {
        "scores": scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "mean_field": mean.reshape(fields.shape[1:]),
    }
