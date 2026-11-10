"""Data assimilation: observations, background error covariance, gains, the estimators
built on them, and the lanes that score them.

Array-level building blocks:

* :mod:`paleoreco.assim.observations` - proxy network loading, per-site stats, error model.
* :mod:`paleoreco.assim.background`   - B and the temporal structure function.
* :mod:`paleoreco.assim.innovation`   - H on the grid: nearest-cell lookup.
* :mod:`paleoreco.assim.ensrf`        - ensemble square-root gains for one covariance.
* :mod:`paleoreco.assim.analog`       - choosing which prior states form an ensemble.

Methods and drivers, reached by submodule path rather than re-exported:

* :mod:`paleoreco.assim.method`       - the Method / AnalysisResult contract.
* :mod:`paleoreco.assim.priors`       - the Prior container and its tapers.
* :mod:`paleoreco.assim.threedvar`    - gain-form 3DVar.
* :mod:`paleoreco.assim.hgaoenkf`     - HGAOEnKF and MTA-HGAOEnKF.
* :mod:`paleoreco.assim.lanes`        - the snapshot, time-series and withholding lanes.
* :mod:`paleoreco.assim.experiments`  - tuning grids, estimators and run drivers.
* :mod:`paleoreco.assim.reconstruction` - full age axis assimilated from a real network.
"""

from paleoreco.assim.observations import (
    FULL_MODEL,
    ObservationModel,
    attach_site_stats,
    collapse_to_samples,
    load_observations,
    observation_site_stats,
    observations_at_age,
    representativeness_variance,
    sample_block_centres,
)
from paleoreco.assim.background import background_covariance, temporal_structure_function
from paleoreco.assim.innovation import (
    nearest_age_index,
    nearest_lat_index,
    nearest_lon_index,
    obs_cell_index,
)

__all__ = [
    "FULL_MODEL",
    "ObservationModel",
    "attach_site_stats",
    "collapse_to_samples",
    "load_observations",
    "observation_site_stats",
    "observations_at_age",
    "representativeness_variance",
    "sample_block_centres",
    "background_covariance",
    "temporal_structure_function",
    "nearest_age_index",
    "nearest_lat_index",
    "nearest_lon_index",
    "obs_cell_index",
]
