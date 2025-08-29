"""Utils for group/algbra operations with and between SU(N) variables."""
import torch
from linalg import trace, adjoint


__all__ = [
    'proj_to_algebra',
    'random_sun_element',
    'inner_prod',
    'matrix_exp',
    'matrix_log'
]


def proj_to_algebra(A):
    """
    Projects a complex-valued matrix `A` into the Lie algebra
    of the group :math:`{\rm SU}(N_c)` by converting it into
    a traceless, Hermitian matrix.

    Args:
        A: Complex-valued matrix

    Returns:
        projA: Projection of A into :math:`\mathfrak{su}(N_c)`
    """
    Nc = A.size(-1)
    trA = torch.eye(Nc)[None, ...] * trace(A)[..., None, None]
    A -= trA / Nc
    projA = (A + adjoint(A)) / 2
    return projA


def _test_proj_to_algebra():
    print('[Testing proj_to_algebra]')
    batch_size = 5
    Nc = 2
    M = (1 + 1j) * torch.randn((batch_size, Nc, Nc))

    A = proj_to_algebra(M)
    trA = torch.eye(Nc)[None, ...] * trace(A)[:, None, None]
    
    assert torch.allclose(trA, torch.zeros_like(trA), atol=1e-6), \
        '[FAILED: A is not traceless]'
    assert torch.allclose(adjoint(A), A), \
        '[FAILED: A is not hermitian]'
    print('[PASSED]')


if __name__ == '__main__': _test_proj_to_algebra()


def random_sun_element(batch_size: int, *, Nc: int) -> torch.Tensor:
    """
    Creates a random element of SU(N) whose elements are
    randomly sampled from a standard normal distribution.

    Args:
        batch_size (int): Number of samples to generate
        Nc (int): Matrix dimension

    Returns:
        Random SU(N) matrices as PyTorch tensors
    """
    A_re = torch.randn((batch_size, Nc, Nc))
    A_im = torch.randn((batch_size, Nc, Nc))
    A = A_re + 1j * A_im
    A = proj_to_algebra(A)
    return torch.matrix_exp(1j * A)


def _test_random_sun_element():
    print('[Testing random_sun_element]')
    batch_size = 5
    Nc = 2
    U = random_sun_element(batch_size, Nc=Nc)

    detU = torch.linalg.det(U)
    assert torch.allclose(detU, torch.ones_like(detU)), \
        '[FAILED: matrix determinant not unity]'
    I = torch.eye(Nc, dtype=U.dtype).repeat(batch_size, 1, 1)
    assert torch.allclose(adjoint(U) @ U, I, atol=1e-6), \
        '[FAILED: matrix not unitary]'
    print('[PASSED]')


if __name__ == '__main__': _test_random_sun_element()


def inner_prod(U, V):
    r"""
    Computes the inner product between two SU(N) Lie
    algebra-valued matrices `U` and `V`. Defines as

    .. math::

        \langle U, V \rangle := {\rm Tr}(U^\dagger V)

    Args:
        U: Square, hermitian matrix
        V: Square, hermitian matrix

    Returns:
        Inner product between `U` and `V` as a real scalar
    """
    return trace(adjoint(U) @ V)


def _test_inner_prod():
    print('[Testing inner_prod]')
    from gens import pauli

    for i in range(4):
        pauli_i = pauli(i)
        for j in range(4):
            pauli_j = pauli(j)
            assert torch.allclose(inner_prod(pauli_i, pauli_j), torch.tensor([i == j], dtype=pauli_j.dtype)), \
                f'[FAILED: pauli {i} not orthonormal to pauli {j}]'
    print('[PASSED]')


if __name__ == '__main__': _test_inner_prod()


def matrix_exp(A):
    """Applies the exponential map to a matrix `A`."""
    return torch.matrix_exp(1j * A)


def matrix_log(U):
    """Computes the matrix logarithm on an input matrix `U`."""
    D, V = torch.linalg.eig(U)
    logD = torch.diag_embed(torch.log(D))
    return -1j * (V @ logD @ adjoint(V))
