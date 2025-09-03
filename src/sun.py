"""Utils for group/algbra operations with and between SU(N) variables."""
import functools
import math
import numpy as np
import torch

from torch import Tensor

from .linalg import trace, adjoint
from .gens import pauli


__all__ = [
    'proj_to_algebra',
    'random_sun_element',
    'inner_prod',
    'matrix_exp',
    'matrix_log'
]


# Set device for tests
if __name__ == '__main__':
    from .devices import set_device, summary
    set_device('cpu')  # TODO: test group_to_coeffs failing on cuda
    print(summary())


def proj_to_algebra(A: Tensor) -> Tensor:
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


def random_sun_element(batch_size: int, *, Nc: int) -> Tensor:
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


def random_sun_lattice(batch_shape: tuple[int, ...], *, Nc: int) -> Tensor:
    """
    Creates a collection of random SU(N) matrices of 
    arbitrary batch dimension specified by `batch_shape`.

    Args:
        batch_shape (tuple): Desired shape of batch to generate
        Nc (int): Matrix dimension

    Returns:
        Random tensor of SU(N) matrices with shape `[*batch_shape, Nc, Nc]`
    """
    B = np.prod(batch_shape)
    U = random_sun_element(B, Nc=Nc)
    return U.reshape(*batch_shape, Nc, Nc)


def _test_random_sun_lattice():
    print('[Testing random_sun_lattice]')
    batch_size = 5
    lattice_shape = (8, 8)
    batch_shape = (batch_size, *lattice_shape)
    Nc = 2
    U = random_sun_lattice(batch_shape, Nc=Nc)

    detU = torch.linalg.det(U)
    assert torch.allclose(detU, torch.ones_like(detU)), \
        '[FAILED: matrix determinant not unity]'
    I = torch.eye(Nc, dtype=U.dtype).repeat((1,) * (len(batch_shape) + 2))
    assert torch.allclose(adjoint(U) @ U, I, atol=1e-6), \
        '[FAILED: matrix not unitary]'
    print('[PASSED]')


if __name__ == '__main__': _test_random_sun_lattice()


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
    from .gens import pauli

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


@functools.cache
def sun_gens(Nc):
    """
    Generators of the SU(N) Lie algebra.

    Normalized to satisfy Tr[T^a T^b] = delta^{ab}, so that the Nc=2 case equals
    the Pauli matrices over sqrt(2) and the Nc=3 case equals the Gell-Mann
    matrices over sqrt(2).
    """
    gens = []
    for j in range(1,Nc):
        for i in range(j):
            gens.append(torch.zeros((Nc,Nc)) + 0j)
            gens[-1][i,j] = 1
            gens[-1][j,i] = 1
            gens.append(torch.zeros((Nc,Nc)) + 0j)
            gens[-1][i,j] = -1j
            gens[-1][j,i] = 1j
        gens.append(torch.zeros((Nc,Nc)) + 0j)
        norm = np.sqrt(j*(j+1)/2)
        for k in range(j):
            gens[-1][k,k] = 1/norm
        gens[-1][j,j] = -j/norm
    # unit normalization
    return torch.stack(gens) / np.sqrt(2)

def embed_sun_algebra(omega, Nc):
    gens = sun_gens(Nc)
    return torch.einsum('...x,xab->...ab', omega.to(gens), gens)

def extract_sun_algebra(A):
    Nc, Nc_ = A.shape[-2:]
    assert Nc == Nc_
    gens = sun_gens(Nc)
    return torch.einsum('...ab,xba->...x', A.to(gens), gens)

def group_to_coeffs(U):
    """
    Decomposes an SU(N) group element into the coefficients
    on the generators in the algebra su(N).

    Args:
        U: Batch of SU(N) matrices

    Returns:
        Batch of N^2 - 1 generator coefficients
    """
    return extract_sun_algebra(matrix_log(U))


def _test_group2coeffs():
    print('[Testing group_to_coeffs]')
    batch_size = 1
    Nc = 2
    U = random_sun_element(batch_size, Nc=Nc)
    coeffs = group_to_coeffs(U)
    assert coeffs.shape == (batch_size, Nc**2 - 1), \
        '[FAILED: incorrect output shape]'
    assert torch.allclose(coeffs.imag, torch.zeros((batch_size, Nc**2 - 1)), atol=1e-5), \
        '[FAILED: generator coefficients should be real]'
    print('[PASSED]')


if __name__ == '__main__': _test_group2coeffs()


def coeffs_to_group(coeffs):
    """
    Recomposes an SU(N) group element given the 
    coefficients of the generators in the Lie algebra su(N)
    by forming the linear combination with the group generators.

    Args:
        coeffs: Batch of N^2 - 1 generator coefficients

    Returns:
        Batch of SU(N) group elements
    """
    Nc = math.isqrt(coeffs.shape[-1]+1)
    return matrix_exp(embed_sun_algebra(coeffs, Nc))


def _test_coeffs2group():
    print('[Testing coeffs_to_group]')
    batch_size = 1
    Nc = 2
    coeffs = torch.randn((batch_size, Nc**2 - 1))
    U = coeffs_to_group(coeffs)
    assert U.shape == (batch_size, Nc, Nc), \
        '[FAILED: incorrect output shape]'
    I =  torch.eye(Nc, dtype=U.dtype).repeat(batch_size, 1, 1)
    assert torch.allclose(U @ adjoint(U), I, atol=1e-6), \
        '[FAILED: result not unitary]'
    assert torch.allclose(torch.linalg.det(U), torch.ones((batch_size,), dtype=U.dtype)), \
        '[FAILED: result does not have unit determinant]'
    print('[PASSED]')


if __name__ == '__main__': _test_coeffs2group()


def mat_angle(U: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    Eigen-decomposes an input matrix `U` and retrives
    its eigenangles and eigenvectors.

    Args:
        U (Tensor): Input matrix to decompose

    Returns:
        th (Tensor): Eigengangles
        V (Tensor): Matrix of eigenvectors
        Vinv (Tensor): Inverse of matrix of eigenvectors
    """
    L, V = torch.linalg.eig(U)
    Vinv = torch.linalg.inv(V)
    th = torch.angle(L)
    return th, V, Vinv
