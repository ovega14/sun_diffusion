"""Forward processes and noise schedules for diffusion."""
import torch
import math
from abc import abstractmethod, ABC

from .heat import sample_sun_hk
from .sun import random_un_haar_element, embed_diag, matrix_exp, adjoint


__all__ = [
    'VarianceExpandingDiffusion',
    'VarianceExpandingDiffusionSUN',
    'PowerDiffusionSUN'
]


class DiffusionProcess(torch.nn.Module, ABC):
    """
    Abstract base class for diffusion processes.
    """
    @abstractmethod
    def diffuse(self, x_0, t):
        raise NotImplementedError()

    #@abstractmethod
    def denoise(self, x_1):
        # TODO
        # For now: keep samplers / O(S)DEsolve in the frontend
        raise NotImplementedError()

    def forward(self, x_0, t):
        """Noises input data samples `x_0` to the noise level at time `t`."""
        return self.diffuse(x_0, t)

    @torch.no_grad()
    def reverse(self, x_1):
        """De-noises prior data samples `x_1` back to new target samples."""
        return self.denoise(x_1)


class VarianceExpandingDiffusion(DiffusionProcess):
    r"""
    Variance-expading diffusion process.

    Noise schedule is given by :math:`g(t) = \sigma^t`, which yields a
    diffusivity of

    .. math::

        \sigma(t) = \sqrt{\frac{\kappa^{2t} - 1}{2 \log\kappa}}.

    Args:
        kappa (float): Noise scale
    """
    def __init__(self, kappa: float):
        super().__init__()
        self.kappa = kappa

    def noise_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the noise coefficient :math:`g(t)` at time `t`."""
        return self.kappa ** t

    def sigma_func(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the width of the heat kernel at time `t`."""
        kappa = torch.tensor(self.kappa)
        num = kappa ** (2*t) - 1
        den = 2 * math.log(kappa)
        return (num / den) ** 0.5

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""
        Diffuses input `x_0` forward to time `t`, where

        .. math::

            x_0 \rightarrow x_t = x_0 + \sigma_t \eta

        and :math:`\eta \sim \mathcal{N}(0, \mathbb{I})`.

        Args:
            x_0 (Tensor): Input data
            t (Tensor): Time step to which to diffuse
        """
        sigma_t = self.sigma_func(t)[:, None]
        eta = torch.randn_like(x_0)
        x_t = x_0 + sigma_t * eta
        return x_t
    

class DiffusionSUN(DiffusionProcess):
    r"""
    Provides :math:`{\rm SU}(N)` diffusion sampling assuming subclasses define
    `sigma_func(t)`.

    Abstract base class.
    """
    def diffuse(self, 
        U_0: torch.Tensor, 
        t: torch.Tensor, 
        n_iter: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Diffuses input data `U_0` to time `t`, where

        .. math::

            U_0 \rightarrow U_t = exp(1j A(\sigma_t)) U_0

        and the noising matrix :math:`A(\sigma_t)` is constructed from samples
        from the :math:`{\rm SU}(N)` spectral heat kernel at time `t`.

        Args:
            U_0 (Tensor): Input data as :math:`{\rm SU}(N)` matrices
            t (Tensor): Time step to which to diffuse
            n_iter (int): Number of iterations for HK sampling. Default: 3

        Returns:
            U_t (Tensor): Matrices diffused to time `t`
            xs (Tensor): Eigenangles sampled from the heat kernel
            V (Tensor): Random Haar-uniform eigenvectors in :math:`{\rm U}(N)`
        """
        batch_size = U_0.size(0)
        Nc, Nc_ = U_0.shape[-2:]
        assert Nc == Nc_, \
            f'U_0 must be a Nc x Nc matrix; got {Nc} x {Nc_}'
        
        sigma_t = self.sigma_func(t)
        xs = sample_sun_hk(batch_size, Nc, width=sigma_t, n_iter=n_iter)
        V = random_un_haar_element(batch_size, Nc=Nc)
        A = V @ embed_diag(torch.tensor(xs)).to(V) @ adjoint(V)
        U_t = matrix_exp(A) @ U_0
        return U_t, xs, V


class VarianceExpandingDiffusionSUN(DiffusionSUN):
    r"""
    Variance-expanding diffusion on the :math:`{\rm SU}(N)` group manifold.

    Args:
        kappa (float): Noise scale
    """
    def __init__(self, kappa: float):
        super().__init__()
        self.kappa = kappa

    def noise_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the noise coefficient :math:`g(t)` at time `t`."""
        return self.sigma ** t

    def sigma_func(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the width of the heat kernel at time `t`."""
        kappa = torch.tensor(self.kappa)
        num = kappa ** (2*t) - 1
        den = 2 * math.log(kappa)
        return (num / den) ** 0.5


class PowerDiffusionSUN(DiffusionSUN):
    r"""
    Power-law diffusion on the :math:`{\rm SU}(N)` group manifold.

    The noise schedule is defined as :math:`g(t) = \kappa t^\alpha`, which
    results in a diffusivity of

    .. math::

        \sigma(t) = \kappa \sqrt{\frac{t^{2\alpha + 1}}{2\alpha + 1}}.

    Args:
        kappa (float): Noise scale
        alpha (float): Time exponent
    """
    def __init__(self, kappa: float, alpha: float):
        super().__init__()
        self.kappa = kappa
        self.alpha = alpha

    def noise_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the noise coefficient :math:`g(t)` at time `t`."""
        return t**self.alpha * self.kappa

    def sigma_func(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the width of the heat kernel at time `t`."""
        p = 2*self.alpha + 1
        return self.kappa * (t**p / p)**0.5
