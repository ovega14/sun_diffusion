"""Physical actions as functions of field configurations."""
import torch

from linalg import trace, adjoint
from sun import random_sun_element


__all__ = [
    'SUNToyAction'
]


class SUNToyAction:
    """
    Toy matrix action defined in terms of a single
    SU(N) degree of freedom on a single lattice site.

    Args:
        beta (float): Coupling strength
    """
    def __init__(self, beta: float):
        self.__beta = beta

    @property
    def beta(self) -> float:
        return self.__beta

    def __call__(self, U):
        assert len(U.shape) == 3, \
            'U must have shape [B, Nc, Nc]'
        Nc = U.shape[-1]
        return -self.beta * trace(U).real / Nc


def apply_gauge_transform(U, V=None):
    """Applies a unitary conjugation to an input element `U`."""
    if V is None:
        V = random_sun_element(U.size(0), Nc=U.size(-1))
    return V @ U @ adjoint(V)


def _test_toy_action():
    print('[Testing SUNToyAction]')
    batch_size = 2
    Nc = 2
    U = random_sun_element(batch_size, Nc=Nc)
    gU = apply_gauge_transform(U)

    action = SUNToyAction(1.0)
    S = action(U)
    Sg = action(gU)
    assert torch.allclose(S, Sg), \
        '[FAILED: Action not gauge-invariant]'
    print('[PASSED]')
    

if __name__ == '__main__': _test_toy_action()
