"""Physical actions as functions of field configurations."""
import torch
from abc import ABC, abstractmethod
from typing import Optional

from .linalg import trace, adjoint
from .sun import random_sun_element, random_sun_lattice, mat_angle
from .utils import roll


__all__ = [
    'SUNToyAction',
    'SUNToyPolynomialAction',
    'SUNPrincipalChiralAction'
]


class Action(ABC):
    """Abstract base class for all physics actions."""    
    @abstractmethod
    def __call__(self, cfgs):
        """Evaluates the action on a batch of configurations `cfgs`."""
        raise NotImplementedError()


class SUNToyAction(Action):
    r"""
    Toy action on a single :math:`{\rm SU}(N)` matrix defined by
    
    .. math::

        S(U) = -\frac{\beta}{N}{\rm Re}{\rm Tr}(U)

    Args:
        beta (float): Coupling strength
    """
    def __init__(self, beta: float):
        self.__beta = beta

    @property
    def beta(self) -> float:
        """Retrieves the coupling strength `beta`."""
        return self.__beta

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        """Computes the action as a function of group elements `U`."""
        assert len(U.shape) == 3, 'U must have shape [B, Nc, Nc]'
        Nc = U.shape[-1]
        return -(self.beta / Nc) * trace(U).real


def _test_toy_action():
    print('[Testing SUNToyAction]')
    batch_size = 2
    Nc = 2
    U = random_sun_element(batch_size, Nc=Nc)
    V = random_sun_element(batch_size, Nc=Nc)

    action = SUNToyAction(1.0)
    S = action(U)
    Sp = action(V @ U @ adjoint(V))
    assert torch.allclose(S, Sp), '[FAILED: Action not conjugation-invariant]'
    print('[PASSED]')
    

if __name__ == '__main__': _test_toy_action()


class SUNToyPolynomialAction(Action):
    r"""
    Toy polynomial action over a single :math:`{\rm SU}(N)` matrix.

    The action is defined as in Eqn. (18) of 2008.05456_.

    .. _2008.05456: https://arxiv.org/abs/2008.05456

    .. math::

        S(U) = -\frac{\beta}{N}{\rm Re}{\rm Tr}\left[\sum_{n} c_n U^n\right],

    where :math:`U^n` denotes repeated matrix multiplication.

    Args:
        beta (float): Inverse coupling strength
        coeffs (list): List of polynomial coefficients. Default: `[1,0,0]`
    """
    def __init__(self,
        beta: float,
        coeffs: list[float] = [1., 0., 0.]
    ):
        self.__beta = beta
        self.coeffs = coeffs

    @property
    def beta(self) -> float:
        """Retrieves the coupling strength `beta`."""
        return self.__beta

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        """Computes the action as a function of group elements `U`."""
        Nc = U.shape[-1]
        action = 0
        for i, c in enumerate(self.coeffs):
            action += c * torch.matrix_power(U, i+1)
        return -(self.beta / Nc) * trace(action).real

    def value_eigs(self, thetas: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluates the action in terms of the matrix eigenangles `thetas`.

        In the angular representation, the action is given by

        .. math::

            S(\theta) = -\frac{\beta}{N} \sum_n c_n \sum_j\cos(n \theta_j)

        Args:
            thetas (Tensor): Batch of eigenangles, shaped `[B, Nc]`

        Returns:
            Batch of action values
        """
        Nc = thetas.shape[-1]
        S = 0
        for i, c in enumerate(self.coeffs):
            S += c * torch.cos((i+1) * thetas).sum(-1)
        return -(self.beta / Nc) * S

    def force_eigs(self, thetas: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluates the force as a function of the matrix eigenangles `thetas`.

        In the angular representation, the force is given by

        .. math::

            F(\theta) = -\frac{\beta}{N} \sum_n n c_n \sin(n\theta)

        Args:
            thetas (Tensor): Batch of eigenangles, shaped `[B, Nc]`

        Returns:
            Batch of force values
        """
        Nc = thetas.shape[-1]
        F = 0
        for i, c in enumerate(self.coeffs):
            F += (i+1) * c * torch.sin((i+1) * thetas)
        return -(self.beta / Nc) * F


def _test_toy_polynomial_action():
    batch_size = 5
    Nc = 2
    U = random_sun_element(batch_size, Nc=2)
    ths, _, _ = mat_angle(U)
    ths.requires_grad_()

    beta = 1.0
    coeffs = [1.0, 1.0, 1.0]
    action = SUNToyPolynomialAction(beta, coeffs)

    # Check the action
    print('[Testing SU(N) toy polynomial action on eigs vs U...]')
    S = action(U)
    S2 = action.value_eigs(ths)
    assert torch.allclose(S, S2), \
        print('[FAILED: Actions evaluated on U vs. eigenangles do not match]')
    print('[PASSED]')

    # Check the force vs autograd
    print('[TESTING SU(N) toy polynomial force vs autograd force...]')
    F = action.force_eigs(ths)
    grad = torch.autograd.grad(
        outputs=S2,
        inputs=ths,
        grad_outputs=torch.ones_like(S2),
        create_graph=True
    )[0]
    assert torch.allclose(F, -grad), \
        '[FAILED: Analytic force does not match autograd force]'
    print('[PASSED]')
    

if __name__ == '__main__': _test_toy_polynomial_action()


class SUNPrincipalChiralAction(Action):
    r"""
    Action for the :math:`{\rm SU}(N) \times {\rm SU}(N)` Principle Chiral
    model on the lattice.

    The action is defined as

    .. math::

        S[U] = -\beta N \sum_x \sum_\mu {\rm Re}{\rm Tr}\left[
                U^\dagger(x) U(x + \hat{\mu})
            \right]

    This theory enjoys two noteworthy symmetries:
        - local charge conjugation invariance: :math:`U(x) \to U^\dagger(x)`
        - global left-right multiplication invariance: 
            :math:`U(x) \to V_L U(X) V_R^\dagger`

    Args:
        beta (float): Inverse temperature parameter
    """
    def __init__(self, beta: float):
        self.__beta = beta

    @property
    def beta(self) -> float:
        """Retrieves the coupling strength `beta`."""
        return self.__beta

    def __call__(self, U):
        Nc = U.shape[-1]
        Nd = len(U.shape) - 3
        assert Nd == 2, 'Only implemented for Nd = 2 so far'

        action_density = 0
        for mu in range(Nd):
            action_density += trace(adjoint(U) @ roll(U, -1, mu+1)).real

        dims = tuple(range(1, Nd+1))
        return -self.beta * Nc * action_density.sum(dims)


def _test_pcm_action():
    print('[Testing SUNPrincipalChiralAction]')
    batch_size = 5
    Nc = 2
    lattice_shape = (4, 4)
    U = random_sun_lattice((batch_size, *lattice_shape), Nc=Nc)

    beta = 1.0
    action = SUNPrincipalChiralAction(beta)
    S = action(U)

    # Charge conjugation symmetry
    S_adj = action(adjoint(U))
    assert torch.allclose(S, S_adj), \
        '[FAILED: action not invariant under local charge conjugation]'

    # Global SU(N)_L x SU(N)_R symmetry
    V_L = random_sun_lattice((1, 1, 1), Nc=2)
    V_R = random_sun_lattice((1, 1, 1), Nc=2)
    S_lr = action(V_L @ U @ adjoint(V_R))
    assert torch.allclose(S, S_lr), \
        '[FAILED: action not invariant under global left-right conjugation]'
    
    print('[PASSED]')


if __name__ == '__main__': _test_pcm_action()
