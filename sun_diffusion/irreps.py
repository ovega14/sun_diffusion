""""Utilities for handling the irreducible representations of SU(N)."""
import torch
import itertools


__all__ = [
    'weyl_dimension',
    'casimir',
    'weyl_character',
    'generate_partitions'
]


def weyl_dimension(mu: torch.Tensor) -> float:
    r"""
    Computes the dimension of the :math:`{\rm SU}(N)` irrep labeled by the 
    partition `mu`.

    The formula is given by

    .. math::

        {\rm dim}(\mu) = \prod_{1 \leq i < j \leq N}
            \frac{\mu_i - \mu_j + j - i}{j - i},

    and we conventionally take :math:`\mu_N \equiv 0`.
    
    Args:
        mu (Tensor): Partition for the irrep as tensor of decreasing integers
    
    Returns:
        Weyl dimension for the irrep corresponding to `mu`.
    """
    Nc = len(mu)
    upper_tri = torch.triu_indices(Nc, Nc, offset=1)
    ix, jx = upper_tri
    
    delta_mu = mu[:, None] - mu[None, :]
    mu_ij = delta_mu[ix, jx]
    
    num = mu_ij + jx - ix
    den = jx - ix
    return torch.prod(num / den).item()


def _test_weyl_dimension():
    print('[Testing weyl_dimension...]')
    
    # SU(2)
    Nc = 2
    j = 1  # Lorentz vector (spin-1)
    mu = torch.tensor([2, 0])
    dim = weyl_dimension(mu)
    true_dim = 2*j + 1
    print(f'SU({Nc}) dim =', dim)
    assert dim == true_dim, f'[FAILED: Incorrect Weyl dimension for SU({Nc})]'

    # SU(3)
    Nc = 3
    p, q = 1, 1
    mu = torch.tensor([2, 1, 0])
    dim = weyl_dimension(mu)
    true_dim = (p + 1) * (q + 1) * (p + q + 2) / 2
    print('SU(3) dim =', dim)
    assert dim == true_dim, f'[FAILED: Incorrect Weyl dimension for SU({Nc})]'
    
    print('[PASSED]')
    

if __name__ == '__main__': _test_weyl_dimension()


def casimir(mu: torch.Tensor) -> float:
    r"""
    Computes the value of the :math:`{\rm SU}(N)` quadratic Casimir operator
    for the irrep specified by the partition `mu`.

    The formula is given by

    .. math::

        C_2(\mu) = \sum_{i=1}^N \mu_i (\mu_i - 2i + N + 1)
            - \frac{1}{N} \left(\sum_{i=1}^N \mu_i\right)^2

    Args:
        mu (Tensor): Tensor of :math:`N` decreasing integers

    Returns:
        Quadratic Casimir for irrep `mu`
    """
    Nc = len(mu)
    ns = torch.arange(1, Nc+1)
    
    term1 = torch.sum(mu * (mu - 2*ns + Nc + 1))
    term2 = torch.sum(mu)**2 / Nc
    return (term1 - term2).item() / 2  # norm depends on generator conventions


def _test_casimir():
    print('[Testing casimir...]')
    
    # SU(2)
    j = 3
    mu_su2 = torch.tensor([2*j, 0.])
    C2 = casimir(mu_su2)
    print('SU(2) C2 =', C2)
    assert C2 == j * (j + 1), \
        '[FAILED: Incorrect Casimir value for SU(2)]'

    # SU(3)
    p, q = 1, 1  # adjoint
    mu_su3 = torch.tensor([p + q, q, 0.])
    C2 = casimir(mu_su3)
    print('SU(3) C2 =', C2)
    assert C2 == (p**2 + q**2 + p*q + 3*p + 3*q) / 3, \
        '[FAILED: Incorrect Casimir value for SU(3)]'

    print('[PASSED]')


if __name__ == '__main__': _test_casimir()


def weyl_character(thetas: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    r"""
    Evaluates the Weyl character :math:`\chi_\mu(\theta)` on 
    :math:`{\rm SU}(N)` eigenangles `thetas`.

    The general formula, as in Drouffe and Zuber (1983), is 

    .. math::

        \chi_\mu(U) = \frac{
            \det_{i,j}\left[\lambda_i^{\mu_j + N - j}\right]}{
            \det_{i,j}\left[\lambda_i^{N - j}\right]}

    where the eigenvalues are :math:`\lambda_j = e^{i \theta_j}`.

    Args:
        thetas (Tensor): Batch of eigenangles, shaped `[B, Nc]`
        mu (Tensor): Partition corresponding to SU(N) irrep

    Returns:
        Batch of character values as floats
    """
    Nc = thetas.shape[-1]
    inds = torch.arange(1, Nc+1)
    
    num_coeffs = (mu + Nc - inds)[None, :, None]
    den_coeffs = (Nc - inds)[None, :, None]

    # Expand to [B, Nc, Nc] for determinant
    theta_k = thetas.unsqueeze(1)
    A = torch.exp(1j * num_coeffs * theta_k)
    B = torch.exp(1j * den_coeffs * theta_k)
    
    detA = torch.linalg.det(A)
    detB = torch.linalg.det(B)
    return (detA / detB).real


def _test_character():
    print('[Testing weyl_character...]')
    # SU(2) spin-1/2
    j = 1 / 2
    mu = torch.tensor([2*j, 0])

    th = 0.5
    thetas = torch.tensor([[th, -th]])
    th = torch.tensor(th)
    
    chi = weyl_character(thetas, mu)
    chi_true = torch.sin((2*j + 1) * th) / torch.sin(th)
    assert torch.allclose(chi, chi_true), \
        '[FAILED: Incorrect character value]'    
    print('[PASSED]')


if __name__ == '__main__': _test_character()


def generate_partitions(Nc: int, max_weight: int) -> list[torch.Tensor]:
    r"""
    Generates a list of partitions labeling :math:`{\rm SU}(N)` irreps.

    Each partition :math:`\mu` is a list of decreasing integers

    .. math::

        \mu = (\mu_1, ..., \mu_N), 
            \quad \mu_1 \geq \mu_2 \geq \cdots \geq \mu_N = 0.

    The infinite sum is truncated by specifying the `max_weight` parameter,
    which only allows partitions of weight less than the maximum.

    Args:
        Nc (int): Dimension of the fundamental rep
        max_weight (int): Highest weight of irreps to include

    Returns:
        ptns (list): List containing integer tuples labeling irreps
    """
    ptns = []
    for total in range(max_weight + 1):
        for mu in itertools.product(range(total+1), repeat=Nc-1):
            if sum(mu) == total and list(mu) == sorted(mu, reverse=True):
                ptns.append(torch.tensor(list(mu) + [0]))
    return ptns
