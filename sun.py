"""Utils for group/algbra operations with and between SU(N) variables."""
import torch


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


def _test_proj_to_alg():
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


if __name__ == '__main__': _test_proj_to_alg()
