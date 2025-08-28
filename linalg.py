"""Utilities for basic linear algebra operations."""
import torch
import numpy as np

from torch import Tensor
from numpy.typing import NDArray
from typing import Optional


__all__ = [
    'trace',
    'transpose',
]


def trace(
    mtrx: Tensor | NDArray,
    dim1: Optional[int] = -1,
    dim2: Optional[int] = -2
) -> Tensor | NDArray:
    """
    Computes the trace of a batched matrix `mtrx` along a chosen pair of 
    dimensions `(dim1, dim2)`.

    .. warning:: By default, `trace` does NOT preserve the dimensionality of the 
    matrix.

    Args:
        mtrx (Tensor, NDArray): Matrix to be traced
        dim1 (int): First dimension over which to trace
        dim2 (int): Second dimension over which to trace

    Returns:
        Array with elements along the diagonal of `(dim1, dim2)` summed
    """
    len1, len2 = mtrx.shape[dim1], mtrx.shape[dim2]
    if len1 != len2:
        raise ValueError('Sizes of matrix must match along dim1 and dim2')

    if isinstance(mtrx, torch.Tensor):
        diag = torch.diagonal(mtrx, dim1=dim1, dim2=dim2)
    elif isinstance(mtrx, np.ndarray):
        diag = np.diagonal(mtrx, axis1=dim1, axis2=dim2)
    else:
        raise TypeError(f'Unsupported type {type(mtrx).__name__}')

    return diag.sum(-1)


def transpose(
    mtrx: Tensor | NDArray,
    dim1: Optional[int] = -1,
    dim2: Optional[int] = -2
) -> Tensor | NDArray:
    """
    Transposes a matrix `mtrx` by swapping a chosen pair of dimensions given by
    `(dim1, dim2)`.

    Args:
        mtrx (Tensor, NDArray): Matrix to transpose
        dim1 (int): First dimension in transpose pair
        dim2 (int): Dimension to swap with `dim1`

    Returns:
        Transposed matrix
    """
    if isinstance(mtrx, torch.Tensor):
        return torch.transpose(mtrx, dim1, dim2)
    elif isinstance(mtrx, np.ndarray):
        return np.swapaxes(mtrx, dim1, dim2)
    else:
        raise TypeError(f'Unsupported type {type(mtrx).__name__}')
