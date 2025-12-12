"""Utilities for canonicalizing SU(N) eigenangles."""
import torch
import numpy as np

from .utils import wrap


__all__ = [
    'canonicalize_su2',
    'canonicalize_su3',
    'canonicalize_sun'
]


def canonicalize_su2(thetas: torch.Tensor) -> torch.Tensor:
    r"""
    Canonicalizes a set of :math:`{\rm SU}(2)` eigenangles 
    :math:`(\theta_1, \theta_2)` by

        1.) Set :math:`\theta_1 = {\rm wrap}(|\theta|)`
        2.) Set :math:`\theta_2 = -\theta_1`

    Args:
        thetas (Tensor): Batch of :math:`{\rm SU}(2)` eigenangles

    Returns:
        Canonicalized batch of eigenangles summing to zero
    """
    thetas[..., 0] = wrap(thetas[..., 0]).abs()
    thetas[..., 1] = -thetas[..., 0]
    return thetas


def canonicalize_su3(thetas: torch.Tensor) -> torch.Tensor:
    r"""
    Canonicalizes a set of :math:`{\rm SU}(3)` eigenangles `thetas`.

    Given eigenangles :math:`(\theta_1, theta_2, \theta_3)`, the algorithm for
    canonicalization is

        1.) Project onto hyperplane defined by :math:`\sum_i \theta_i = 0`
        2.) Map into coordinates :math:`(a, b, c)`
        3.) Wrap onto canonical hexagon centered at the identity
        4.) Impose hexagonal constraints by wrapping into [-0.5, 0.5]
        5.) Round and shift into the centered hexagon

    Args:
        thetas (Tensor): Batch of :math:`{\rm SU}(3)` eigenangles

    Returns:
        Canonicalized batch of eigenangles summing to zero
    """
    thetas[..., -1] -= torch.sum(thetas, dim=-1)  # sum_i theta_i = 0

    U_inv = torch.tensor([  # maps v -> (a, b, c)
        [1, 0, -1],
        [0, -1, 1],
        [-1, 1, 0]
    ]) / (6*np.pi)
    v = thetas.reshape(-1, 3)
    kappa = U_inv @ torch.transpose(v, 0, 1)  # ij, jb -> ib
    a, b, c, = kappa[0], kappa[1], kappa[2]

    k = (b + c) / 2
    a -= k
    b -= k
    c -= k
    a -= torch.round(a)

    k = torch.round(b)
    b -= k
    c += k
    b -= torch.round(b - (a + c)/2)

    k = (b + c) / 2
    a -= k
    b -= k
    c -= k
    a -= torch.round(a)
    c -= torch.round(c - (a + b)/2)

    U = 2*np.pi * torch.tensor([  # maps (a, b, c) -> v = (th1, th2, th3)
        [1, 0, -1],
        [0, -1, 1],
        [-1, 1, 0]
    ])
    kappa = torch.stack([a, b, c], dim=0)
    return torch.transpose(U @ kappa, 0, 1).reshape(thetas.shape)


def canonicalize_sun(thetas: torch.Tensor) -> torch.Tensor:
    """Wrapper for SU(2) and SU(3) canonicalization."""
    Nc = thetas.shape[-1]
    if Nc == 2:
        return canonicalize_su2(thetas)
    if Nc == 3:
        return canonicalize_su3(thetas)
    raise NotImplementedError(f'SU({Nc}) canonicalization not supported')
