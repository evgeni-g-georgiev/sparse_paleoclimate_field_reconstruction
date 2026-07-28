"""Diffusion-trainer evaluation helpers.

:func:`plot_loss_curves_diffusion` reads the ``history`` dict shape produced by
:mod:`paleoreco.training.trainer_diffusion`. Its held-out and training-probe
curves all share one fixed noise-level grid, so their levels are comparable and
the gap between them is a generalisation gap; ``train_loss`` is the random-sigma
optimisation target and sits on a different scale, so it gets its own axis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _mark(ax, best_epoch: int | None, stop_epoch: int | None, label: bool) -> None:
    """Best-epoch and early-stopping markers, drawn only where the value is real."""
    if stop_epoch is not None and stop_epoch >= 0:
        ax.axvline(stop_epoch, color="r", lw=1.2,
                   label=f"early stop ep {stop_epoch}" if label else None)
    if best_epoch is not None and best_epoch >= 0:
        ax.axvline(best_epoch, color="k", lw=1, alpha=0.4, ls="--",
                   label=f"best / refit ep {best_epoch}" if label else None)


def plot_loss_curves_diffusion(
    history: dict,
    best_epoch: int | None = None,
    stop_epoch: int | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Held-out and training-probe scores on the left, the optimisation target on the right.

    The left panel is skipped when ``history`` carries no held-out curve, so a
    fixed-length run still renders. Log y throughout the left panel because the EDM
    loss spans a wide range over training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False,
                             constrained_layout=True)
    has_val = "val_loss" in history and len(history["val_loss"]) > 0

    ax = axes[0, 0]
    if has_val:
        epochs = np.arange(len(history["val_loss"]))
        if len(history.get("train_probe_epoch", [])):
            ax.plot(history["train_probe_epoch"], history["train_probe_loss"],
                    label="train probe", lw=1.5)
        ax.plot(epochs, history["val_loss"], label="held out (EMA)", lw=1.5, ls="--")
        if len(history.get("val_loss_live", [])):
            ax.plot(epochs, history["val_loss_live"], label="held out (live)",
                    lw=1.0, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("fixed-grid EDM loss")
        ax.set_title("Generalisation (one shared noise-level grid)", fontsize=10)
        ax.grid(True, alpha=0.3)
        _mark(ax, best_epoch, stop_epoch, label=True)
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()

    ax = axes[0, 1]
    ax.plot(np.arange(len(history["train_loss"])), history["train_loss"],
            label="train", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("EDM loss (random sigma)")
    ax.set_title("Optimisation target", fontsize=10)
    ax.grid(True, alpha=0.3)
    _mark(ax, best_epoch, stop_epoch, label=False)
    ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
    return fig
