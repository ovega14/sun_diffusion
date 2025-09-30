"""
A note on conventions: For now, we use `width` to denote what was
previously called `sigma` since the heat kernel standard deviation
can sometimes be a function of time or depend directly on time itself.
Sigma can be a related or unrelated parameter elsewhere.
"""
import torch
import numpy as np
import itertools

from torch import Tensor
from typing import Optional

from .utils import grab
from .canon import canonicalize_sun


# =======================================================================
#  Euclidean Heat Kernel
# =======================================================================
def eucl_log_hk(x: Tensor, *, width: Tensor):
    """Log density of Euclidean heat kernel with width `width`."""
    return -(x**2).sum(-1) / (2 * width**2)


def eucl_score_hk(x: Tensor, *, width: Tensor):
    """Analytical score function for the Euclidean heat kernel with width `width`."""
    return -x / width[...,None]**2


# =======================================================================
#  SU(N) Heat Kernel
# =======================================================================
def _sun_hk_meas_J(delta: Tensor):
    """Measure term Jij on Hermitian matrix eigenvalue differences."""
    return 2 * torch.sin(delta / 2)


def _sun_hk_meas_D(delta: Tensor):
    """Measure term Dij on Hermitian matrix eigenvalue differences."""
    return delta


def _sun_hk_unwrapped(xs, *, width, eig_meas=True):
    """Computes the SU(N) Heat Kernel over the unwrapped space of eigenangles."""
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


def _sun_score_hk_unwrapped(xs, *, width):
    """
    Computes the analytical score function of the SU(N)
    heat kernel over the unwrapped (non-compact) space of
    eigenangles.
    """
    K = _sun_hk_unwrapped(xs, width=width, eig_meas=False)
    xn = -torch.sum(xs, dim=-1, keepdims=True)
    xs = torch.cat([xs, xn], dim=-1)

    Nc = xs.shape[-1]
    delta = xs[..., :, None] - xs[..., None, :]
    delta += 0.1 * torch.eye(Nc).to(xs)  # avoid division by zero

    # Gradient of measure term
    grad_meas = 1 / delta - 0.5 / torch.tan(delta/2)
    grad_meas = grad_meas * (1 - torch.eye(Nc)).to(xs)  # mask diagonal
    grad_meas = grad_meas.sum(-1)

    # Gradient of Gaussian weight term
    grad_weight = eucl_score_hk(xs, width=width)
    return (grad_meas + grad_weight) * K[...,None]


def sun_hk(
    thetas: Tensor,
    *,
    width: Tensor,
    n_max: Optional[int] = 3,
    eig_meas: Optional[bool] = True
) -> Tensor:
    """
    Computes the SU(N) heat kernel of width `width` over
    the wrapped space of eigenangles.

    Args:
        thetas (Tensor): Wrapped SU(N) eigenangles
        width (Tensor): Standard deviation of the heat kernel, batched
        n_max (int): Max number of pre-image sum correction terms to include
        eig_meas (bool): Weather to include the measure term over the eigenangles

    Returns:
        SU(N) heat kernel evaluated at the input angles `thetas`
    """
    total = 0
    lattice_shifts = itertools.product(range(-n_max, n_max), repeat=thetas.shape[-1])
    # Sum over periodic lattice shifts to account for pre-images
    for ns in lattice_shifts:
        ns = torch.tensor(ns)
        xs = thetas + 2*np.pi * ns
        total = total + _sun_hk_unwrapped(xs, width=width, eig_meas=eig_meas)
    return total


def sun_score_hk(
    thetas: Tensor,
    *,
    width: float,
    n_max: Optional[int] = 3
) -> Tensor:
    """
    Computes the analytical score function for the wrapped
    SU(N) heat kernel of width `width` as a function of
    wrapped SU(N) eigenangles.

    Args:
        thetas (Tensor): Wrapped SU(N) eigenangles
        width (float): Standard deviation of the heat kernel
        n_max (int): Max number of pre-image sum correction terms to include

    Returns:
        Analytical gradient of the SU(N) heat kernel log-density
    """
    total = 0
    lattice_shifts = itertools.product(range(-n_max, n_max), repeat=thetas.shape[-1])
    K = sun_hk(thetas, width=width, eig_meas=False)
    # Sum over periodic lattice shifts to account for pre-images
    for ns in lattice_shifts:
        ns = torch.tensor(ns)
        xs = thetas + 2*np.pi * ns
        total = total + _sun_score_hk_unwrapped(xs, width=width)
    return total / (K[...,None] + 1e-12)


def sun_score_hk_autograd(
    thetas: Tensor,
    *,
    width: float,
    n_max: Optional[int] = 3
) -> Tensor:
    """
    Computes the score function for the wrapped SU(N) heat kernel
    of width `width` by automatic differentiation of the log density
    in `thetas`.

    Args:
        thetas (Tensor): Wrapped SU(N) eigenangles
        width (float): Standard deviation of the heat kernel
        n_max (int): Max number of pre-image sum correction terms to include

    Returns:
        Autograd derivative of the SU(N) heat kernel log-density
    """
    if len(thetas.shape) != 2:
        raise ValueError('Expects batched thetas')
    Nc = thetas.shape[-1] + 1
    f = lambda ths: sun_hk(ths, width=width, n_max=n_max, eig_meas=False)
    def gradf(ths):
        g = torch.func.grad(f)(ths) / f(ths)
        gn = -g.sum(-1) / Nc
        return torch.cat([g + gn, gn[..., None]], dim=-1)
    return torch.func.vmap(gradf)(thetas)


def _test_sun_score_hk():
    print('[Testing sun_score_hk]')
    torch.manual_seed(1234)
    batch_size = 128
    Nc = 3
    thetas = 3*np.pi*torch.rand((batch_size, Nc-1))
    width = torch.ones((batch_size,))

    a = sun_score_hk(thetas, width=width, n_max=1)
    b = sun_score_hk_autograd(thetas, width=1.0, n_max=1)

    assert torch.allclose(a, b), f'{a=} {b=} {a/b=}'
    print('[PASSED test_sun_score]')


if __name__ == '__main__': _test_sun_score_hk()


def sample_sun_hk(
    batch_size: int,
    Nc: int,
    *,
    width: Tensor,
    n_iter: Optional[int] = 3,
    n_max: Optional[int] = 3
):
    """
    Generates `batch_size` many samples from the SU(Nc) heat
    kernel of width `width` using importance sampling by

        1.) Sampling eigenangles with `n_iter` iterations of IS
        2.) Sample random eigenvectors and recomposing with eigenvals

    Args:
        batch_size (int): Number of samples to generate
        Nc (int): Dimension of fundamental rep. of SU(Nc)
        width (float) Standard deviation of heat kernel
        n_iter (int): Number of IS iterations
        n_max (int): Max number of terms to include in HK pre-image sum
    """
    def propose():
        """Samples proposal eigenangles from uniform dist."""
        xs = 2*np.pi*np.random.random(size=(batch_size, Nc))
        xs[...,-1] = -np.sum(xs[...,:-1])
        return grab(canonicalize_sun(torch.tensor(xs)))
    
    # Sample eigenangles
    assert width.shape == (batch_size,), 'width should be batched'
    xs = propose()
    for i in range(n_iter):
        xps = propose()
        # ratio b/w new, old points
        p = grab(sun_hk(torch.tensor(xps[..., :-1]), width=width, n_max=n_max))
        p /= grab(sun_hk(torch.tensor(xs[..., :-1]), width=width, n_max=n_max))
        u = np.random.random(size=p.shape)
        xs[u < p] = xps[u < p]  # accept / reject step

    # Sample eigenvectors
    # V = grab(random_sun_haar_element(batch_size, Nc))
    # D = np_embed_diag(xs)  # embed diagonal
    # A = V @ D @ adjoint(V)
    # return xs, A
    return xs
