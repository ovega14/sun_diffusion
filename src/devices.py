"""Utilities for handling PyTorch devices and dtypes."""
import torch

from torch import Tensor
from typing import Optional, Any
from numpy.typing import NDArray


HAS_CUDA = torch.cuda.is_available()
_device = torch.device('cpu')
_cuda_id = 0
_default_dtype = torch.float64  # for real tensors
_default_complex_dtype = torch.complex128  # default 64 on GPU; complex128 on CPU


def set_device(device: Optional[str] = None, cuda_id: Optional[int]  = 0) -> None:
    """
    Set global device and default dtype for torch.

    Args:
        device (str): 'cpu' or 'cuda'. If None, defaults to 'cuda' if available.
        cuda_id (int): Which CUDA device to use.
    """
    global _device, _cuda_id, _default_dtype, _default_complex_dtype

    if device is None:
        device = 'cuda' if HAS_CUDA else 'cpu'

    if device == 'cuda':
        if not HAS_CUDA:
            raise RuntimeError('CUDA is not available.')
        torch.cuda.set_device(cuda_id)
        _device = torch.device(f'cuda:{cuda_id}')
        _cuda_id = cuda_id
        _default_dtype = torch.float32
        _default_complex_dtype = torch.complex64
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(f'cuda:{cuda_id}')
    else:
        _device = torch.device('cpu')
        _default_dtype = torch.float64
        _default_complex_dtype = torch.complex128
        torch.set_default_dtype(torch.float64)
        torch.set_default_device('cpu')


def get_device() -> torch.device:
    """Return the current torch.device."""
    return _device


def get_dtype(complex: bool = False) -> torch.dtype:
    """Return the current default dtype (complex or real)."""
    return _default_complex_dtype if complex else _default_dtype

def set_dtype(dtype: torch.dtype):
    global _default_dtype
    torch.set_default_dtype(dtype)
    _default_dtype = torch.get_default_dtype()
    _default_complex_dtype = torch.promote_types(_default_dtype, torch.complex64)

def device_name() -> str:
    """Get human-readable name for current device."""
    if _device.type == 'cuda':
        return torch.cuda.get_device_name(_cuda_id)
    return 'CPU'


def summary() -> str:
    """Return a summary string of the current device setup."""
    return f'Using device: {_device} ({device_name()}) with dtype: {torch.get_default_dtype()}'
