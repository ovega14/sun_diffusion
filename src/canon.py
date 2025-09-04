import numpy as np
import torch
from .utils import wrap


__all__ = [
    'canonicalize_su2',
    'canonicalize_su3',
    'canonicalize_sun'
]


def canonicalize_su2(thW):
    r"""
    Canonicalizes a set of SU(2) eigenangles :math:`(\theta_1, \theta_2)` by

        1.) Set :math:`\theta_1 = {\rm wrap}(|\theta|)`,
        2.) Set :math:`\theta_2 = -\theta_1`.

    Args:
        thW: Batch of non-canonicalized SU(2) eigenangles.

    Returns:
        Canonicalized batch of eigenangles that sum to zero
    """
    thW[..., 0] = wrap(thW[..., 0]).abs()  # map to [0, pi]
    thW[..., 1] = -thW[..., 0]  # second angle is negative first
    return thW


def canonicalize_su3(thW):
    r"""
    Canonicalizes a set of SU(3) eigenangles :math:`(\theta_1, theta_2, \theta_3)` by

        1.) Project onto hyperplane defined by :math:`\sum_i \theta_i = 0`,
        2.) Wrap onto canonical hexagon centered at the identity
        3.) Map into coordinates :math:`(a, b, c)` from angles :math:`\theta_i`
        4.) Impose hexagonal constraints by wrapping :math:`a, b, c` into [-0.5, 0.5]
        5.) Round and shift into the centered hexagon

    Args:
        thW: Batch of non-canonicalzied SU(3) eigenangles

    Returns:
        Canonicalized batch of SU(3) eigenangles summing to zero
    
    """
    thW[..., -1] -= torch.sum(thW, dim=-1)  # sum_i theta_i = 0
    v = thW.reshape(-1, 3)

    U = 2*np.pi * torch.tensor([  # map (a,b,c) -> v = (th1,th2,th3)
        [1, 0, -1],
        [0, -1, 1],
        [-1, 1, 0]
    ])
    U_inv = torch.tensor([  # map v -> (a,b,c)
        [1, 0, -1],
        [0, -1, 1],
        [-1, 1, 0]
    ]) / (6*np.pi)

    kappa = U_inv @ torch.transpose(v, 0, 1)
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

    kappa = torch.stack([a, b, c], dim=0)
    return torch.transpose(U @ kappa, 0, 1).reshape(thW.shape)


def canonicalize_sun(thW):
    Nc = thW.shape[-1]
    if Nc == 2:
        return canonicalize_su2(thW)
    elif Nc == 3:
        return canonicalize_su3(thW)
    else:
        raise NotImplementedError(f'{Nc =} not supported')
