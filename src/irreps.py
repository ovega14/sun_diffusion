""""Utilities for handling the irreducible representations of SU(N)."""
import torch


__all__ = [
    'weyl_dimension',
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
