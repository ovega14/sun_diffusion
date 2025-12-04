"""Miscellaneous utilities."""
import torch
import numpy as np
import math

from torch import Tensor
from numpy.typing import NDArray
from typing import Any


def grab(x: Tensor | Any) -> NDArray | Any:
    """
    Detaches a `torch.Tensor` from the computational graph and loads it into 
    the CPU memory as a numpy `NDArray`. Otherwise returns the input as is.

    Args: 
        x (Tensor): PyTorch tensor to be detached

    Returns:
        (NDArray) Detached NumPy array
    """
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    else:
        return x


def wrap(theta: Tensor | NDArray) -> Tensor | NDArray:
    r"""
    Wraps a non-compact input variable into the
    compact interval :math:`[-\pi, \pi]`.

    Args:
        theta (Tensor, NDArray): Non-compact, real-valued input

    Returns:
        Input wrapped around [-pi, pi]
    """    
    return (theta + np.pi) % (2*np.pi) - np.pi


def _test_wrap():
    print('[Testing wrap...]')
    batch_size = 5
    x = 20 * torch.randn((batch_size))
    wx = wrap(x)
    pi = np.pi * torch.ones_like(x)
    assert torch.all((-pi < wx) & (wx < pi)), \
        '[FAILED: Output must be within (-pi, pi)]'
    print('[PASSED]')


if __name__ == '__main__': _test_wrap()


def roll(
    x: NDArray | Tensor,
    shifts: int | tuple[int, ...],
    dims: int | tuple[int, ...]
) -> NDArray | Tensor:
    """
    Bi-compatible wrapper for the `roll` function in NumPy and PyTorch.
    
    Rolls a NumPy array or PyTorch tensor around a given dimension or set of
    dimensions given in `dims` by corresponding amount(s) specified in `shifts`.

    Args:
        x (Tensor, NDArray): Array or tensor object to roll
        shifts (int, tuple): Shift amount(s), where negative values shift right
        dims (int, tuple): Axes or dims along which to shift

    Returns:
        Rolled array / tensor
    """
    if isinstance(x, torch.Tensor):
        return torch.roll(x, shifts=shifts, dims=dims)
    elif isinstance(x, np.ndarray):
        return np.roll(x, shift=shifts, axis=dims)
    else:
        raise TypeError(f'Unsupported type {type(x)}')


def logsumexp(x, axis: int = 0):
    """Bi-compatible wrapper for `logsumexp` in NumPy and PyTorch."""
    if isinstance(x, torch.Tensor):
        return torch.logsumexp(x, dim=axis)
    elif isinstance(x, np.ndarray):
        return np.logaddexp.reduce(x, axis=axis)
    raise TypeError(f'Unsupported type {type(x)}')


def logsumexp_signed(
    x: NDArray | Tensor,
    signs: NDArray | Tensor,
    axis: int
) -> (NDArray | Tensor, NDArray | Tensor):
    ind_pos = (signs > 0)
    ind_neg = ~ind_pos
    if isinstance(x, torch.Tensor):
        x_pos = torch.where(ind_pos, x, -float('inf'))
        x_neg = torch.where(ind_neg, x, -float('inf'))
        x_pos = torch.logsumexp(x_pos, dim=axis)
        x_neg = torch.logsumexp(x_neg, dim=axis)
        out_signs = torch.where(x_pos > x_neg, torch.ones_like(x_pos), -torch.ones_like(x_pos))
        out_logs = torch.where(
            x_pos > x_neg, x_pos + torch.log(1.0 - torch.exp(x_neg-x_pos)),
            x_neg + torch.log(1.0 - torch.exp(x_pos-x_neg)))
        return out_logs, out_signs
    elif isinstance(x, np.ndarray):
        raise NotImplementedError()
    else:
        raise TypeError(f'Unsupported type {type(x)}')


def _test_logsumexp_signed():
    print('[Testing logsumexp_signed]')
    torch.manual_seed(1234)
    batch_size = 5
    Nd = 128
    x = 4*torch.rand((batch_size,Nd)) - 2
    log_x = x.abs().log()
    sign_x = x.sign()
    log_a, sign_a = logsumexp_signed(log_x, sign_x, axis=-1)
    b = torch.sum(x, axis=-1)
    log_b = b.abs().log()
    sign_b = b.sign()
    assert torch.allclose(log_a, log_b), f'{log_a=} {log_b=}'
    assert torch.allclose(sign_a, sign_b)
    print('[PASSED logsumexp_score]')


if __name__ == '__main__': _test_logsumexp_signed()


def compute_kl_div(logp, logq):
    """Computes the reverse KL divergence, assuming model samples x~q."""
    return (logq - logp).mean()  # OV: torch/np bicompatible


def compute_ess(logp, logq):
    logw = logp - logq
    log_ess = 2 * logsumexp(logw) - logsumexp(2 * logw)
    return math.exp(log_ess) / len(logw)
