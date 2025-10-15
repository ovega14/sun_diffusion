"""
A note on conventions: For now, we use `width` to denote what was previously
called `sigma` since the heat kernel standard deviation can sometimes be a
function of time or depend directly on time itself. Sigma can be a related or
unrelated parameter elsewhere.
"""
import torch
import numpy as np
import itertools

from torch import Tensor
from typing import Optional

from .utils import grab, logsumexp_signed
from .canon import canonicalize_sun


# =============================================================================
#  Euclidean Heat Kernel
# =============================================================================
def eucl_log_hk(x: Tensor, *, width: Tensor) -> Tensor:
    """Log density of Euclidean heat kernel, ignoring normalization."""
    dims = tuple(range(1, x.ndim))
    return -(x**2).sum(dims) / (2 * width**2)


def eucl_score_hk(x: Tensor, *, width: Tensor):
    """Analytical score function for the Euclidean heat kernel."""
    return -x / width[..., None]**2


# =============================================================================
#  SU(N) Heat Kernel
# =============================================================================
def _sun_hk_meas_J(delta):
    """Measure term Jij on Hermitian matrix eigenvalue differences."""
    return 2 * torch.sin(delta / 2)


def _sun_hk_meas_D(delta):
    """Measure term Dij on Hermitian matrix eigenvalue differences."""
    return delta


def _log_sun_hk_unwrapped(xs, *, width, eig_meas=True):
    """Computes the SU(N) log Heat Kernel over the unwrapped eigenangles."""
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

    # Gaussian (Euclidean) heat kernel weight
    log_weight = eucl_log_hk(xs, width=width)
    return log_meas + log_weight, sign


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


def log_sun_hk(
    thetas: Tensor,
    *,
    width: Tensor,
    n_max: Optional[int] = 3,
    eig_meas: Optional[bool] = True
) -> Tensor:
    """Computes the SU(N) log Heat Kernel over wrapped eigenangles."""
    log_values = []
    signs = []
    
    # Sum over periodic lattice shifts to account for pre-images
    lattice_shifts = itertools.product(range(-n_max, n_max+1), repeat=thetas.shape[-1])
    for ns in lattice_shifts:
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
    return log_sun_hk(thetas, width=width, n_max=n_max, eig_meas=eig_meas).exp()


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


def _sun_score_hk_unwrapped(xs: Tensor, *, width: Tensor) -> Tensor:
    """
    Computes the analytical score function for the SU(Nc) heat kernel over the 
    unwrapped (non-compact) space of eigenangles.

    .. note:: Assumes that `xs` only includes the Nc-1 independent eigenangles.

    .. note:: This implementation returns the product of the score with the 
    heat kernel, dK/dx = s(x) * K. 

    Args:
        xs (Tensor): Batch of unwrapped eigenangles, shaped `[B, Nc-1]`
        width (Tensor): Std deviation of the heat kernel, shaped `[B]`

    Returns:
        (Tensor) Gradient of the SU(Nc) heat kernel w.r.t. eigenangles
    """

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
    return grad_meas + grad_weight


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
    lattice_shifts = itertools.product(range(-n_max, n_max+1), repeat=thetas.shape[-1])
    # K = sun_hk(thetas, width=width, eig_meas=False)
    logK = log_sun_hk(thetas, width=width, eig_meas=False)
    # Sum over periodic lattice shifts to account for pre-images
    for ns in lattice_shifts:
        ns = torch.tensor(ns)
        xs = thetas + 2*np.pi * ns
        # Ki = _sun_hk_unwrapped(xs, width=width, eig_meas=False)
        logKi, si = _log_sun_hk_unwrapped(xs, width=width, eig_meas=False)
        total = total + (si * (logKi-logK).exp())[...,None] * _sun_score_hk_unwrapped(xs, width=width)
    return total


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


def sun_score_hk_autograd(
    thetas: Tensor,
    *,
    width: float,
    n_max: Optional[int] = 3
) -> Tensor:
    """
    Computes the score function for the wrapped SU(N) heat kernel by automatic
    differentiation of the log density in the eigenangles `thetas`.

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


def sun_score_hk_autograd_v2(
    thetas: Tensor,
    *,
    width: float,
    n_max: Optional[int] = 3
) -> Tensor:
    """
    Computes the score function for the wrapped SU(N) heat kernel by automatic
    differentiation of the log density in the eigenangles `thetas`.

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
