"""Utilities for group/algebra operations with and between SU(N) variables."""
import functools
import math
import numpy as np
import torch

from torch import Tensor

from .linalg import trace, adjoint
from .gens import pauli


__all__ = [
    'matrix_exp',
    'matrix_log',
    'proj_to_algebra',
    'random_sun_element',
    'random_un_haar_element',
    'random_sun_lattice',
    'inner_prod',
    'embed_sun_algebra',
    'extract_sun_algebra',
    'group_to_coeffs',
    'coeffs_to_group',
    'mat_angle',
    'extract_diag',
    'embed_diag'
]


# Set device for tests
if __name__ == '__main__':
    from .devices import set_device, summary
    set_device('cpu')  # TODO: test group_to_coeffs failing on cuda
    print(summary())


def matrix_exp(A: torch.Tensor) -> torch.Tensor:
    """Applies the complex exponential map to a matrix `A`."""
    return torch.matrix_exp(1j * A)


def matrix_log(U: torch.Tensor) -> torch.Tensor:
    """Computes the matrix logarithm on an input matrix `U`."""
    D, V = torch.linalg.eig(U)
    logD = torch.diag_embed(torch.log(D))
    return -1j * (V @ logD @ adjoint(V))


def proj_to_algebra(A: torch.Tensor) -> torch.Tensor:
    r"""
    Projects a complex-valued matrix `A` into the :math:`{\rm SU}(N)` Lie 
    algebra by converting it into a traceless, Hermitian matrix.

    Args:
        A (Tensor): Batch of complex-valued square matrices

    Returns:
        Projection of A into :math:`\mathfrak{su}(N)`
    """
    Nc = A.size(-1)
    trA = torch.eye(Nc)[None, ...] * trace(A)[..., None, None]
    A -= trA / Nc
    return (A + adjoint(A)) / 2


def _test_proj_to_algebra():
    print('[Testing proj_to_algebra...]')
    batch_size = 5
    Nc = 2
    M = (1 + 1j) * torch.randn((batch_size, Nc, Nc))

    A = proj_to_algebra(M)
    trA = torch.eye(Nc)[None, ...] * trace(A)[:, None, None]
    
    assert torch.allclose(adjoint(A), A), '[FAILED: result must be hermitian]'
    assert torch.allclose(trA, torch.zeros_like(trA), atol=1e-6), \
        '[FAILED: result must be traceless]'
    print('[PASSED]')


if __name__ == '__main__': _test_proj_to_algebra()


def random_sun_element(
    batch_size: int, 
    *, 
    Nc: int, 
    scale: float = 1.0
) -> torch.Tensor:
    r"""
    Creates a batch of random :math:`{\rm SU}(N)` matrices whose elements are
    randomly sampled from a standard normal distribution.

    Args:
        batch_size (int): Number of matrices to generate
        Nc (int): Dimension of each matrix
        scale (float): Width of the normal density. Default: 1.0

    Returns:
        Batch of random :math:`{\rm SU}(N)` matrices as PyTorch tensors
    """
    A_re = scale * torch.randn((batch_size, Nc, Nc))
    A_im = scale * torch.randn((batch_size, Nc, Nc))
    A = A_re + 1j*A_im
    A = proj_to_algebra(A)
    return matrix_exp(A)


def _test_random_sun_element():
    print('[Testing random_sun_element...]')
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


def random_un_haar_element(batch_size: int, *, Nc: int) -> torch.Tensor:
    r"""
    Creates a batch of Haar-random :math:`{\rm U}(N)` matrices.

    Args:
        batch_size (int): Number of matrices to generate
        Nc (int): Dimension of each matrix

    Returns:
        Batch of random :math:`{\rm U}(N)` matrices as PyTorch tensors
    """
    U_re = torch.randn((batch_size, Nc, Nc))
    U_im = torch.randn((batch_size, Nc, Nc))
    U = U_re + 1j*U_im
    
    Q, R = torch.linalg.qr(U)
    R *= torch.eye(R.shape[-1])
    R /= torch.abs(R) + (1-torch.eye(R.shape[-1]))
    return Q @ R


def random_sun_lattice(
    batch_shape: tuple[int, ...], 
    *, 
    Nc: int
) -> torch.Tensor:
    r"""
    Creates a collection of random :math:`{\rm SU}(N)` matrices with arbitrary
    batch dimension specified by `batch_shape`.

    Args:
        batch_shape (tuple): Desired shape of batch to generate
        Nc (int): Matrix dimension

    Returns:
        Tensor of matrices with shape `[*batch_shape, Nc, Nc]`
    """
    B = np.prod(batch_shape)
    U = random_sun_element(B, Nc=Nc)
    return U.reshape(*batch_shape, Nc, Nc)


def _test_random_sun_lattice():
    print('[Testing random_sun_lattice...]')
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


def inner_prod(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the inner product between two Lie algebra matrices `U` and `V`. 

    The inner product on :math:`\mathfrak{su}(N)` is defined as

    .. math::

        \langle U, V \rangle := {\rm Tr}(U^\dagger V)

    Args:
        U (Tensor): Batch of traceless, Hermitian matrices
        V (Tensor): Batch of traceless, Hermitian matrices

    Returns:
        Inner product between `U` and `V` as a batch of real scalars
    """
    return trace(adjoint(U) @ V)


def _test_inner_prod():
    print('[Testing inner_prod...]')
    from .gens import pauli

    for i in range(4):
        pauli_i = pauli(i)
        for j in range(4):
            pauli_j = pauli(j)
            assert torch.allclose(
                inner_prod(pauli_i, pauli_j), 
                torch.tensor([i == j], dtype=pauli_j.dtype)
            ), f'[FAILED: pauli {i} not orthonormal to pauli {j}]'
    print('[PASSED]')


if __name__ == '__main__': _test_inner_prod()


@functools.cache
def sun_gens(Nc: int) -> torch.Tensor:
    r"""
    Generators of the :math:`\mathfrak{su}(N)` Lie algebra.

    .. note:: The generators are normalized to satisfy 
    :math:`{\rm Tr}[T^a T^b] = \delta^{ab}`, so that the `Nc = 2` case equals
    the Pauli matrices over :math:`\sqrt{2}` and the `Nc = 3` case equals the
    Gell-Mann matrices over :math:`\sqrt{2}`.

    Args:
        Nc (int): Dimension of the matrices in the algebra

    Returns:
        Generators as stacked PyTorch tensors, shape `[Nc**2 - 1, Nc, Nc]`
    """
    gens = []
    for j in range(1,Nc):
        for i in range(j):
            gens.append(torch.zeros((Nc, Nc)) + 0j)
            gens[-1][i, j] = 1
            gens[-1][j, i] = 1
            gens.append(torch.zeros((Nc, Nc)) + 0j)
            gens[-1][i, j] = -1j
            gens[-1][j, i] = 1j
        gens.append(torch.zeros((Nc, Nc)) + 0j)
        norm = np.sqrt(j * (j + 1) / 2)
        for k in range(j):
            gens[-1][k, k] = 1 / norm
        gens[-1][j, j] = -j / norm
    # unit normalization
    return torch.stack(gens) / np.sqrt(2)


def embed_sun_algebra(omega: torch.Tensor, Nc: int) -> torch.Tensor:
    r"""Constructs a matrix in :math:`\mathfrak{su}(N)` from coeffs `omega`."""
    gens = sun_gens(Nc)
    return torch.einsum('...x, xab -> ...ab', omega.to(gens), gens)


def extract_sun_algebra(A: torch.Tensor) -> torch.Tensor:
    r"""Returns the coeffs from a matrix `A` in :math:`\mathfrak{su}(N)`."""
    Nc, Nc_ = A.shape[-2:]
    assert Nc == Nc_, 'input matrix must be square along final two dimensions'
    gens = sun_gens(Nc)
    return torch.einsum('...ab, xba -> ...x', A.to(gens), gens)


def group_to_coeffs(U: torch.Tensor) -> torch.Tensor:
    r"""
    Decomposes an :math:`{\rm SU}(N)` matrix into the coefficients on the
    generators in the algebra :math:`\mathfrak{su}(N)`.

    Args:
        U (Tensor): Batch of :math:`{\rm SU}(N)` matrices

    Returns:
        Batch of :math:`N^2 - 1` generator coefficients
    """
    return extract_sun_algebra(matrix_log(U))


def _test_group2coeffs():
    print('[Testing group_to_coeffs...]')
    batch_size = 1
    Nc = 2
    U = random_sun_element(batch_size, Nc=Nc)
    coeffs = group_to_coeffs(U)
    assert coeffs.shape == (batch_size, Nc**2 - 1), \
        '[FAILED: incorrect output shape]'
    assert torch.allclose(
        coeffs.imag, 
        torch.zeros((batch_size, Nc**2 - 1)), 
        atol=1e-5
    ), '[FAILED: generator coefficients should be real]'
    print('[PASSED]')


if __name__ == '__main__': _test_group2coeffs()


def coeffs_to_group(coeffs: torch.Tensor) -> torch.Tensor:
    r"""
    Recomposes an :math:`{\rm SU}(N)` matrix given generator coefficients.
    
    The group element is reconstructed by forming the linear combination with
    the generators in :math:`\mathfrak{su}(N)`, and then exponentiating onto
    the group manifold:

    ..math::

        U = \exp(\sum_a c_a T_a)

    Args:
        coeffs (Tensor): Batch of :math:`N^2 - 1` generator coefficients

    Returns:
        Batch of :math:`{\rm SU}(N)` matrices
    """
    Nc = math.isqrt(coeffs.shape[-1]+1)
    return matrix_exp(embed_sun_algebra(coeffs, Nc))


def _test_coeffs2group():
    print('[Testing coeffs_to_group...]')
    batch_size = 1
    Nc = 2
    coeffs = torch.randn((batch_size, Nc**2 - 1))
    U = coeffs_to_group(coeffs)
    assert U.shape == (batch_size, Nc, Nc), '[FAILED: incorrect output shape]'
    I =  torch.eye(Nc, dtype=U.dtype).repeat(batch_size, 1, 1)
    assert torch.allclose(U @ adjoint(U), I, atol=1e-6), \
        '[FAILED: result not unitary]'
    detU = torch.linalg.det(U)
    assert torch.allclose(detU, torch.ones((batch_size,), dtype=U.dtype)), \
        '[FAILED: result does not have unit determinant]'
    print('[PASSED]')


if __name__ == '__main__': _test_coeffs2group()


def mat_angle(
    U: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Eigen-decomposes a matrix `U` to get its eigenangles and eigenvectors.

    Args:
        U (Tensor): Input matrix to decompose

    Returns:
        thetas (Tensor): Eigengangles of `U`
        V (Tensor): Matrix of eigenvectors of `U`
        Vinv (Tensor): Inverse matrix of eigenvectors of `U`
    """
    eigs, V = torch.linalg.eig(U)
    Vinv = torch.linalg.inv(V)
    thetas = torch.angle(eigs)
    return thetas, V, Vinv


def extract_diag(M: torch.Tensor) -> torch.Tensor:
    """Extracts the diagonal entries of `M` as `(..., n, n) -> (..., n)`."""
    return torch.einsum('...ii->...i', M)


def embed_diag(d: torch.Tensor) -> torch.Tensor:
    """Embeds a batch of diagonal entries `d` as `(..., n) -> (..., n, n)`."""
    return torch.eye(d.shape[-1]) * d[...,None]


def np_extract_diag(M):
    return np.einsum('...ii->...i', M)


def np_embed_diag(d):
    return np.identity(d.shape[-1]) * d[...,None]
