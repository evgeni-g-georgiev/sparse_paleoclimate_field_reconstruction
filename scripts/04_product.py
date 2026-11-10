"""The MIS3 reconstruction.

The lanes exist to choose an estimator. That choice is made, so this stops measuring and
produces the product: one MTA-HGAOEnKF analysis per age of the whole archive, at the taper
and analog settings the snapshot lane selected, with the real network at each age and the
full observation model. The prior is every state of the archive.

Nothing here can score an amplitude, so the product is published at several side by side.
"""

from __future__ import annotations

import _common as C

from paleoreco import paths
from paleoreco.assim import experiments as ex
from paleoreco.assim.reconstruction import run_reconstruction

ESTIMATOR = ex.ESTIMATOR_MTA
STAGES = ["product"]


def main() -> None:
    smoke = C.smoke()
    stages = C.Stages("04_product", STAGES)
    cube, ages, lats, lons, valid = C.load_prior(C.SMOKE_AGES if smoke else None)
    _, long = C.load_networks(ages if smoke else None)

    if stages.run("product"):
        taper, point = C.operating_point(ESTIMATOR)
        run_reconstruction(
            cube, ages, lats, lons, valid, long, C.clear_dir(paths.PRODUCT), taper=taper,
            make_method=point.factory(cube, ages, lats, lons),
            b_scales=ex.PRODUCT_B_SCALES,
            record={"estimator": ESTIMATOR, "point": point.record()},
            progress_every=100)
    stages.done()


if __name__ == "__main__":
    main()
