"""Loss utility functions."""

from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor


def to_numpy(x: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def stratify_loss_by_time(
    batch_t: Union[Tensor, np.ndarray],
    batch_loss: Union[Tensor, np.ndarray],
    loss_name: Optional[str] = None,
    num_bins: int = 4,
) -> Dict[str, float]:
    """
    Stratify loss by binning timesteps.

    Args:
        batch_t: Timesteps for each sample in batch (B,) or (B * multiplicity,)
        batch_loss: Loss values for each sample (B,) or (B * multiplicity,)
        loss_name: Name of the loss for key generation. If None, uses 'loss'.
        num_bins: Number of time bins to stratify into

    Returns:
        Dictionary with stratified loss values:
            - "{loss_name} t=[{bin_start:.2f},{bin_end:.2f})": mean loss for each bin
    """
    batch_t = to_numpy(batch_t)
    batch_loss = to_numpy(batch_loss)

    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()

    # Define bin edges (add small epsilon to include t=1.0 in last bin)
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)

    # Compute bin indices for each sample
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1

    # Aggregate losses by bin
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses, minlength=num_bins)
    t_binned_n = np.bincount(bin_idx, minlength=num_bins)

    if loss_name is None:
        loss_name = "loss"

    stratified_losses = {}
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"

        if t_binned_n[t_bin] > 0:
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        else:
            range_loss = float("nan")

        stratified_losses[t_range] = range_loss

    return stratified_losses
