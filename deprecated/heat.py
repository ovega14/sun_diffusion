"""Deprecated module for heat kernel utilities."""
import torch
import numpy as np
import itertools

from torch import Tensor
from typing import Optional


def eucl_log_hk(x: Tensor, *, width: Tensor) -> Tensor:
    """Log density of Euclidean heat kernel, ignoring normalization."""
    dims = tuple(range(1, x.ndim))
    return -(x**2).sum(dims) / (2 * width**2)


def _sun_hk_unwrapped(xs, *, width, eig_meas=True):
    """Computes the SU(N) Heat Kernel over the unwrapped eigenangles."""
    xn = -torch.sum(xs, dim=-1, keepdims=True)
    xs = torch.cat([xs, xn], dim=-1)

    # Compute pariwise differences between eigenangles
    delta_x = torch.stack([
        xs[..., i] - xs[..., j]
        for i in range(xs.shape[-1]) for j in range(i+1, xs.shape[-1])
    ], dim=-1)

    # Include / exclude Haar measure J^2 factor
    if eig_meas:
        meas = torch.prod(_sun_hk_meas_D(delta_x) * _sun_hk_meas_J(delta_x), dim=-1)
    else:
        meas = torch.prod(_sun_hk_meas_D(delta_x) / _sun_hk_meas_J(delta_x), dim=-1)

    # Gaussian (Euclidean) heat kernel weight
    weight = torch.exp(eucl_log_hk(xs, width=width))
    return meas * weight


def sun_hk_old(thetas, *, width, n_max=3, eig_meas=True):  # (ovega): the original implementation
    """Computes the SU(N) heat kernel over the wrapped eigenangles."""
    total = 0
    lattice_shifts = itertools.product(range(-n_max, n_max), repeat=thetas.shape[-1])
    # Sum over periodic lattice shifts to account for pre-images
    for ns in lattice_shifts:
        ns = torch.tensor(ns)
        xs = thetas + 2*np.pi * ns
        total = total + _sun_hk_unwrapped(xs, width=width, eig_meas=eig_meas)
    return total
