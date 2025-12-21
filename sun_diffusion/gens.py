"""Hard-coded generators for the SU(2) Lie algebra."""
import torch
import numpy as np


__all__ = [
    'pauli'
]


if __name__ == '__main__':
    from .devices import set_device, summary
    set_device('cpu')
    print(summary())


_su2_gens = {
    '0': 1j * np.array([
        [-1j, 0.],
        [0., -1j]]),
    '1': 1j * np.array([
        [0., -1j],
        [-1j, 0.]]),
    '2': 1j * np.array([
        [0., -1.],
        [1., 0.]]),
    '3': 1j * np.array([
        [-1j, 0.],
        [0., 1j]])
}


def pauli(i: int) -> torch.Tensor:
    r"""
    Retrieves the i\ :sup:`th` Pauli matrix as a PyTorch tensor.

    .. note::
        The pauli matrices are normalized by a factor of
        :math:`\frac{1}{\sqrt{2}}` so that they form an orthonormal basis for
        :math:`{\rm SU}(2)`, i.e., :math:`{\rm tr}[\sigma^a \sigma^b] = \delta^{ab}`.

    Args:
        i (int): Index of pauli matrix to get

    Returns:
        Pauli matrix `i` as a PyTorch tensor
    """
    if not isinstance(i, int):
        raise TypeError('Please input an integer between 0 and 3')
    if i not in list(range(4)):
        raise ValueError(f'Pauli matrix {i} not defined')
    pauli_i = _su2_gens[str(i)] / 2**0.5
    return torch.tensor(pauli_i)


def _test_su2_gens():
    print('[Testing SU(2) generators...]')
    for i in range(4):
        pauli_i = pauli(i)
        print(f'Pauli {i}:\n', pauli_i)
        assert torch.allclose(pauli_i @ pauli_i, torch.eye(2, dtype=pauli_i.dtype) / 2), \
            f'[FAILED: Pauli matrix {i} does not square to identity]'
    print('[PASSED]')


if __name__ == '__main__': _test_su2_gens()
