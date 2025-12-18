"""
Module for heat kernel evaluations and analytical score functions.

Notation: We use `width` to denote the standard deviation of the heat kernel
instead of `sigma`, since this is used as a parameter label elsewhere. We often
abbreviate the heat kernel to HK.
"""
import torch
import numpy as np
import itertools

from torch import Tensor

from .utils import grab, logsumexp_signed
from .canon import canonicalize_sun
from .irreps import (
    casimir,
    weyl_dimension,
    weyl_character,
    grad_character,
    generate_partitions
)


__all__ = [
    'log_sun_hk',
    'sun_hk',
    'sun_dual_hk',
    'sun_score_hk',
    'sun_score_dual_hk',
    'sun_score_hk_autograd',
    'sun_score_hk_autograd_v2',
    'sample_sun_hk'
]


def eucl_log_hk(x: Tensor, *, width: Tensor) -> Tensor:
    """Log density of Euclidean heat kernel, ignoring normalization."""
    dims = tuple(range(1, x.ndim))
    return -(x**2).sum(dims) / (2 * width**2)


def eucl_score_hk(x: Tensor, *, width: Tensor):
    """Analytical score function for the Euclidean heat kernel."""
    return -x / width[..., None]**2


def _sun_hk_meas_J(delta):
    """Measure term :math:`J_{ij}` on eigenvalue differences `delta`."""
    return 2 * torch.sin(delta / 2)


def _sun_hk_meas_D(delta):
    """Measure term :math:`D_{ij}` on eigenvalue differences `delta`."""
    return delta


def _log_sun_hk_unwrapped(xs: Tensor, *, width: Tensor, eig_meas: bool = True):
    r"""Computes the :math:`{\rm SU}(N)` log HK over unwrapped eigenangles."""
    xn = -torch.sum(xs, dim=-1, keepdims=True)
    xs = torch.cat([xs, xn], dim=-1)

    # Compute pariwise differences between eigenangles
    delta_x = torch.stack([
        xs[..., i] - xs[..., j]
        for i in range(xs.shape[-1]) for j in range(i+1, xs.shape[-1])
    ], dim=-1)

    # Include / exclude Haar measure J^2 factor
    J_sign = 1 if eig_meas else -1
    log_meas = torch.sum(
        _sun_hk_meas_D(delta_x).abs().log() +
        J_sign * _sun_hk_meas_J(delta_x).abs().log(), dim=-1)
    sign = torch.prod(
        _sun_hk_meas_D(delta_x).sign() *
        _sun_hk_meas_J(delta_x).sign(), dim=-1)

    # Gaussian (Euclidean) HK weight
    log_weight = eucl_log_hk(xs, width=width)
    return log_meas + log_weight, sign


def log_sun_hk(
    thetas: Tensor,
    *,
    width: Tensor,
    n_max: int = 3,
    eig_meas: bool = True
) -> Tensor:
    r"""Computes the :math:`{\rm SU}(N)` log HK over wrapped eigenangles."""
    log_values = []
    signs = []
    
    # Sum over pre-images
    shifts = itertools.product(range(-n_max, n_max+1), repeat=thetas.shape[-1])
    for ns in shifts:
        ns = torch.tensor(ns)
        xs = thetas + 2*np.pi * ns
        log_value, sign = _log_sun_hk_unwrapped(xs, width=width, eig_meas=eig_meas)
        log_values.append(log_value)
        signs.append(sign)

    log_total, signs = logsumexp_signed(torch.stack(log_values), torch.stack(signs), axis=0)
    # assert torch.all(signs > 0)
    return log_total


def sun_hk(
    thetas: Tensor,
    *,
    width: Tensor,
    n_max: int = 3,
    eig_meas: bool = True
) -> Tensor:
    r"""
    Evaluates the :math:`{\rm SU}(N)` heat kernel on wrapped eigenangles.

    .. note:: This function assumes the input only includes the :math:`N-1`
    independent eigenangles.

    Args:
        thetas (Tensor): Wrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Standard deviation of the heat kernel, batched
        n_max (int): Max number of pre-image sum terms to include. Default: 3
        eig_meas (bool): Weather to include Haar measure term. Default: `True`

    Returns:
        :math:`{\rm SU}(N)` heat kernel evaluated on the angles `thetas`
    """
    return log_sun_hk(thetas, width=width, n_max=n_max, eig_meas=eig_meas).exp()


def sun_dual_hk(
    thetas: torch.Tensor, 
    *, 
    width: torch.Tensor, 
    max_weight: int = 5,
    eig_meas: bool = True
) -> torch.Tensor:
    r"""
    Evaluates the :math:`{\rm SU}(N)` dual heat kernel over wrapped eigenangles
    as a character expansion.

    .. note:: This function assumes the input only includes the :math:`N-1`
    independent eigenangles.    

    Args:
        thetas (Tensor): Wrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Standard deviation of the heat kernel, batched
        max_weight (int): Max total weight of irreps to sum. Default: 5
        eig_meas (bool): Whether to include Haar measure term. Default: `True`

    Returns:
        :math:`{\rm SU}(N)` dual heat kernel evaluated on the angles `thetas`
    """
    thn = -torch.sum(thetas, dim=-1, keepdim=True)
    thetas = torch.cat([thetas, thn], dim=-1)
    Nc = thetas.shape[-1]
    
    K = 0
    partitions = generate_partitions(Nc, max_weight)
    for mu in partitions:
        d_mu = weyl_dimension(mu)
        c_mu = casimir(mu)
        chi_mu = weyl_character(thetas, mu)
        K += d_mu * torch.exp(-c_mu * width**2) * chi_mu

    # Haar measure factor
    if eig_meas:
        delta = torch.stack([
            thetas[..., i] - thetas[..., j]
            for i in range(Nc) for j in range(i+1, Nc)
        ], dim=-1)
        K = K * torch.prod(_sun_hk_meas_J(delta)**2, dim=-1)
    return K


def _sun_score_hk_unwrapped(xs: Tensor, *, width: Tensor) -> Tensor:
    r"""
    Computes the analytical score function for the :math:`{\rm SU}(N)` HK over
    unwrapped eigenangles.

    .. note:: Assumes `xs` only includes the :math:`N-1` independent angles.

    Args:
        xs (Tensor): Batch of unwrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Standard deviation of the heat kernel, shaped `[B]`

    Returns:
        Gradient of the log HK w.r.t. `xs`
    """
    xn = -torch.sum(xs, dim=-1, keepdims=True)
    xs = torch.cat([xs, xn], dim=-1)  # tr(X) = 0
    Nc = xs.shape[-1]
    
    delta = xs[..., :, None] - xs[..., None, :]
    delta += 0.1 * torch.eye(Nc).to(xs)  # avoid division by zero

    # Gradient of measure term
    grad_meas = 1 / delta - 0.5 / torch.tan(delta/2)
    grad_meas = grad_meas * (1 - torch.eye(Nc)).to(xs)  # mask diagonal
    grad_meas = grad_meas.sum(-1)

    # Gradient of Gaussian weight term
    grad_weight = eucl_score_hk(xs, width=width)
    return grad_meas + grad_weight


def sun_score_hk(thetas: Tensor, *, width: Tensor, n_max: int = 3) -> Tensor:
    r"""
    Computes the analytical score function for the wrapped :math:`{\rm SU}(N)`
    heat kernel over eigenangles `thetas`.

    .. note:: This function assumes the input only includes the :math:`N-1` 
    independent eigenangles.

    Args:
        thetas (Tensor): Wrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Standard deviation of the heat kernel
        n_max (int): Max number of pre-image sum terms to include. Default: 3

    Returns:
        Analytical gradient of the wrapped log HK
    """
    shifts = itertools.product(range(-n_max, n_max+1), repeat=thetas.shape[-1])
    logK = log_sun_hk(thetas, width=width, eig_meas=False)
    
    # Sum over pre-images
    total = 0
    for ns in shifts:
        xs = thetas + 2*np.pi * torch.tensor(ns)
        logKi, si = _log_sun_hk_unwrapped(xs, width=width, eig_meas=False)
        exp_factor = (si * torch.exp(logKi - logK))[..., None]
        total = total + exp_factor * _sun_score_hk_unwrapped(xs, width=width)
    return total


def sun_score_dual_hk(
    thetas: torch.Tensor, 
    *, 
    width: torch.Tensor, 
    max_weight: int = 5
) -> torch.Tensor:
    r"""
    Computes the analytical score function for the dual :math:`{\rm SU}(N)`
    heat kernel over eigenangles `thetas`.

     .. note:: This function assumes the input only includes the :math:`N-1` 
    independent eigenangles.

    Args:
        thetas (Tensor): Wrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Standard deviation of the heat kernel, batched
        max_weight (int): Max total weight of irreps to sum. Default: 5

    Returns:
        Analytical gradient of the character expansion HK
    """
    K = sun_dual_hk(thetas, width=width, max_weight=max_weight, eig_meas=False)
    thetas = torch.cat(
        [thetas, -torch.sum(thetas, dim=-1, keepdim=True)]
    , dim=-1)
    Nc = thetas.shape[-1]

    gradK = 0
    partitions = generate_partitions(Nc, max_weight)
    for mu in partitions:
        d_mu = weyl_dimension(mu)
        C_mu = casimir(mu)
        grad_chi_mu = grad_character(thetas, mu)
        gradK += d_mu * torch.exp(-C_mu * width**2)[:, None] * grad_chi_mu
    return gradK / K[:, None]


def sun_score_hk_autograd(
    thetas: Tensor,
    *,
    width: float,
    n_max: int = 3
) -> Tensor:
    r"""
    Computes the score function for the wrapped :math:`{\rm SU}(N)` heat kernel
    via automatic differentiation of :math:`K` in `thetas`, followed by 
    division by :math:`K`.

    This function uses autograd to compute the score **indirectly** as

    .. math::

        s(\theta) = \frac{\nabla K(\theta)}{K(\theta)}

    Args:
        thetas (Tensor): Wrapped eigenangles, shaped `[B, Nc-1]`
        width (float): Standard deviation of the heat kernel
        n_max (int): Max number of pre-image sum terms to include. Default: 3

    Returns:
        Autograd derivative of the HK divided by :math:`K`
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


def sun_score_hk_autograd_v2(
    thetas: Tensor,
    *,
    width: float,
    n_max: int = 3
) -> Tensor:
    r"""
    Computes the score function for the wrapped :math:`{\rm SU}(N)` heat kernel
    via automatic differentiation of :math:`\log K` in `thetas`.

    This function uses autograd to compute the score **directly** as

    .. math::

        s(\theta) = \nabla \log K(\theta)

    Args:
        thetas (Tensor): Wrapped eigenangles, shaped `[B, Nc-1]`
        width (float): Standard deviation of the heat kernel
        n_max (int): Max number of pre-image sum terms to include. Default: 3

    Returns:
        Autograd derivative of the HK log-density
    """
    if len(thetas.shape) != 2:
        raise ValueError('Expects batched thetas')
    Nc = thetas.shape[-1] + 1
    f = lambda ths: log_sun_hk(ths, width=width, n_max=n_max, eig_meas=False)
    def gradf(ths):
        g = torch.func.grad(f)(ths)
        gn = -g.sum(-1) / Nc
        return torch.cat([g + gn, gn[..., None]], dim=-1)
    return torch.func.vmap(gradf)(thetas)


def _test_sun_score_hk():
    print('[Testing sun_score_hk]')
    torch.manual_seed(1234)
    batch_size = 16
    Nc = 3
    thetas = 4*np.pi*(2*torch.rand((batch_size, Nc))-1)
    thetas = canonicalize_sun(thetas)
    thetas_in = thetas[:,:-1]
    # NOTE(gkanwar): Making the width much smaller results in the autograd impls
    # giving nan while sun_score_hk remains stable.
    width = 0.5
    width_batch = width * torch.ones((batch_size,))

    a = sun_score_hk(thetas_in, width=width_batch, n_max=1)
    b = sun_score_hk_autograd_v2(thetas_in, width=width, n_max=1)
    c = sun_score_hk_autograd(thetas_in, width=width, n_max=1)

    assert torch.allclose(b, c), f'{b=} {c=} {b/c=}'

    inds = (torch.sum(~torch.isclose(a, b), dim=-1) != 0)
    ratio = a/b
    thetas_ratio = grab(torch.stack([ratio[inds], thetas[inds]/np.pi], dim=-2))
    assert torch.allclose(a, b), f'{a[inds]=} {b[inds]=}\n{thetas_ratio=}'
    print('[PASSED test_sun_score]')


if __name__ == '__main__': _test_sun_score_hk()


def sample_sun_hk(
    batch_size: int,
    Nc: int,
    *,
    width: Tensor,
    n_iter: int = 3,
    n_max: int = 3
):
    r"""
    Samples from the :math:`{\rm SU}(N)` heat kernel with importance sampling.

    Args:
        batch_size (int): Number of samples to generate
        Nc (int): Dimension of fundamental rep. of SU(Nc)
        width (float) Standard deviation of heat kernel
        n_iter (int): Number of IS iterations. Default: 3
        n_max (int): Max number of pre-image sum terms to include. Default: 3

    Returns:
        xs: Batch of eigenangles shaped `[B, Nc]` from the heat kernel
    """
    def propose():
        """Samples proposal eigenangles from patched measure."""
        sigma_cut = 0.5
        xa = 2*np.pi*torch.rand(size=(batch_size, Nc))
        xa[..., -1] = -torch.sum(xa[..., :-1], dim=-1)
        xb = width[..., None] * torch.randn(size=(batch_size, Nc))
        xb -= torch.mean(xb, dim=-1, keepdim=True)
        assert torch.all((width[..., None] > sigma_cut) | (xb.abs() < np.pi))
        xs = torch.where(width[..., None] < sigma_cut, xb, xa)
        xs = canonicalize_sun(xs)
        # NOTE(gkanwar): logq is not normalized. This is okay given the fixed
        # width over sampling iterations.
        logqb = -torch.sum(xb**2, dim=-1)/(2*width**2)
        logq = torch.where(width < sigma_cut, logqb, 0.0)
        return xs, logq

    # Sample eigenangles
    assert width.shape == (batch_size,), 'width should be batched'
    xs, old_logq = propose()
    old_logp = log_sun_hk(xs[..., :-1], width=width, n_max=n_max)
    for i in range(n_iter):
        xps, new_logq = propose()
        # ratio b/w new, old points
        new_logp = log_sun_hk(xps[..., :-1], width=width, n_max=n_max)
        log_acc = new_logp - new_logq + old_logq - old_logp
        # do comparison in F64 just to be safe
        u = torch.rand(size=log_acc.shape, dtype=torch.float64).log()
        acc = u < log_acc.to(u)
        xs[acc] = xps[acc]  # accept / reject step
        old_logq[acc] = new_logq[acc]
        old_logp[acc] = new_logp[acc]

    # Sample eigenvectors
    # V = grab(random_sun_haar_element(batch_size, Nc))
    # D = np_embed_diag(xs)  # embed diagonal
    # A = V @ D @ adjoint(V)
    # return xs, A
    # TODO(gkanwar): Convert signature to return torch.Tensor?
    return grab(xs)
