"""Data assimilation: observations, coordinate scorings, background error
covariance, and innovation diagnostics.

The submodules are array-level building blocks shared by the assimilation
methods built on top of them:

* :mod:`paleoreco.assim.observations` - proxy network loading and per-site stats.
* :mod:`paleoreco.assim.scoring`      - raw / anomaly / normalised transforms.
* :mod:`paleoreco.assim.background`   - background state and B from the Prior cube.
* :mod:`paleoreco.assim.innovation`   - H on the grid, innovations, standardisation.
* :mod:`paleoreco.assim.joint`        - pairwise whitened innovations for the 2D test.
"""

from paleoreco.assim.observations import (
    attach_site_stats,
    collapse_to_samples,
    load_observations,
    observation_site_stats,
    observations_at_age,
)
from paleoreco.assim.scoring import (
    ANOMALY,
    NORMALISED,
    RAW,
    SCORINGS,
    score,
)
from paleoreco.assim.background import (
    background_covariance,
    background_state,
    background_variance,
)
from paleoreco.assim.innovation import (
    innovation,
    nearest_age_index,
    nearest_lat_index,
    nearest_lon_index,
    obs_cell_index,
    obs_operator_scale,
    predicted_sd,
    standardise,
)
from paleoreco.assim.joint import (
    rank_pairs,
    whitened_pair,
)

__all__ = [
    "attach_site_stats",
    "collapse_to_samples",
    "load_observations",
    "observation_site_stats",
    "observations_at_age",
    "ANOMALY",
    "NORMALISED",
    "RAW",
    "SCORINGS",
    "score",
    "background_covariance",
    "background_state",
    "background_variance",
    "innovation",
    "nearest_age_index",
    "nearest_lat_index",
    "nearest_lon_index",
    "obs_cell_index",
    "obs_operator_scale",
    "predicted_sd",
    "standardise",
    "rank_pairs",
    "whitened_pair",
]
