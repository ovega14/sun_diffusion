"""Miscellaneous utilities."""
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
