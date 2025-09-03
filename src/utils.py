"""Miscellaneous utilities."""
import torch
import numpy as np

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
    """
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
