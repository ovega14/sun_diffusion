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


def eucl_score_hk(x: Tensor, *, width: Tensor):
    """Analytical score function for the Euclidean heat kernel."""
    return -x / width[..., None]**2


def _sun_score_hk_unwrapped_old(xs, *, width):  # (ovega): the original implementation
    """
    Computes the analytical score function for the SU(Nc) heat kernel over the 
    unwrapped (non-compact) space of eigenangles.
    """
    K = _sun_hk_unwrapped(xs, width=width, eig_meas=False)

    xn = -torch.sum(xs, dim=-1, keepdims=True)
    xs = torch.cat([xs, xn], dim=-1)  # enforce tr(X) = 0
    Nc = xs.shape[-1]

    delta = xs[..., :, None] - xs[..., None, :]
    delta += 0.1 * torch.eye(Nc).to(xs)  # avoid division by zero

    # Gradient of measure term
    grad_meas = 1 / delta - 0.5 / torch.tan(delta/2)
    grad_meas = grad_meas * (1 - torch.eye(Nc)).to(xs)  # mask diagonal
    grad_meas = grad_meas.sum(-1)

    # Gradient of Gaussian weight term
    grad_weight = eucl_score_hk(xs, width=width)

    return (grad_meas + grad_weight) * K[..., None]


def sun_score_hk_old(thetas, *, width, n_max=3):  # (ovega): the original implementation
    """
    Computes the analytical score function for the wrapped
    SU(N) heat kernel of width `width` as a function of
    wrapped SU(N) eigenangles.
    """
    total = 0
    lattice_shifts = itertools.product(range(-n_max, n_max), repeat=thetas.shape[-1])
    K = sun_hk_old(thetas, width=width, eig_meas=False)
    # Sum over periodic lattice shifts to account for pre-images
    for ns in lattice_shifts:
        ns = torch.tensor(ns)
        xs = thetas + 2*np.pi * ns
        total = total + _sun_score_hk_unwrapped_old(xs, width=width)
    #return total / K[..., None]
    return total / (K[..., None] + 1e-12)


def sun_score_hk_stable(  # (ovega): this was my alternative 'stable' implementation
    thetas: Tensor, 
    *, 
    width: Tensor, 
    n_max: Optional[int] = 3
) -> Tensor:
    """
    Computes the exact score function for the SU(Nc) heat kernel over the
    wrapped space of eigenangles. Implementation is done using `logsumexp` for 
    more numerical stability.

    .. note:: Assumes that `thetas` only includes the Nc-1 independent 
    eigenangles.

    Args:
        thetas (Tensor): Batch of wrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Std deviation of the heat kernel, shaped `[B]`
        n_max (int): Max number of pre-image sum terms to include. Default: 3

    Returns:
        (Tensor) Gradient of the log SU(Nc) heat kernel w.r.t eigenangles
    """
    # Build all lattice shifts
    Nc = thetas.size(-1) + 1
    lattice_shifts = itertools.product(range(-n_max, n_max+1), repeat=Nc-1)
    shifts = torch.tensor(list(lattice_shifts))
    theta_exp = thetas.unsqueeze(1) + 2*np.pi*shifts.unsqueeze(0)
    xN = -theta_exp.sum(dim=-1, keepdim=True)
    xs = torch.cat([theta_exp, xN], dim=-1)

    # Compute unwrapped scores and log K_unwrapped
    grad_unwrapped = []
    logK_unwrapped = []
    for s in range(len(shifts)):
        xn = xs[:, s, :]
        g = _sun_score_hk_unwrapped(xn, width=width)
        grad_unwrapped.append(g)
        
        # Log measure term
        ix, jx = torch.triu_indices(Nc, Nc, offset=1)
        delta = torch.abs(xn[:, ix] - xn[:, jx])
        logM = torch.log(delta) - torch.log(2*torch.sin(delta/2).abs())
        logM = logM.sum(-1)  # sum over positive roots
        
        # Log K_unwrapped = log measure + log Gaussian weight
        logG = eucl_log_hk(xn, width=width)
        logK = logM + logG
        logK_unwrapped.append(logK)
        
    grad_unwrapped = torch.stack(grad_unwrapped, dim=1)
    logK_unwrapped = torch.stack(logK_unwrapped, dim=1)
    
    # logsumexp for K_wrapped
    a_max = logK_unwrapped.max(dim=1, keepdim=True).values
    sum_exp = torch.exp(logK_unwrapped - a_max).sum(dim=1, keepdim=True)
    logK_wrapped = (a_max + torch.log(sum_exp + 1e-12)).squeeze(-1)

    # softmax probs; weighted sum over shifts -> wrapped score
    probs = torch.exp(logK_unwrapped - logK_wrapped.unsqueeze(-1))
    probs = probs.unsqueeze(-1)
    score_full = (probs * grad_unwrapped).sum(dim=1)
    return score_full
