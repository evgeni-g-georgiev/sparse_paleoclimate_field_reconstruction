"""Model training: losses, the AE/VAE trainers, and the CV fold loop.

* :mod:`paleoreco.training.losses`      - masked reconstruction + VAE ELBO losses.
* :mod:`paleoreco.training.trainer_ae`  - ConvAE training loop.
* :mod:`paleoreco.training.trainer_vae` - ConvBetaVAE training loop (KL warmup).
* :mod:`paleoreco.training.cv`          - per-fold anomaly centring, fit/eval loop, refit.

``set_seed`` is re-exported as the one cross-trainer convenience.
"""

from __future__ import annotations

from ._common import set_seed
from .losses import masked_mse, masked_rmse, vae_elbo_loss

__all__ = ["set_seed", "masked_mse", "masked_rmse", "vae_elbo_loss"]
