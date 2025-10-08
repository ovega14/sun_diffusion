import torch
from abc import abstractmethod, ABC

from .heat import sample_sun_hk
from .sun import random_un_haar_element, embed_diag, matrix_exp, adjoint


__all__ = [
    'VarianceExpandingDiffusion',
    'VarianceExpandingDiffusionSUN'
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
    """
    Variance-expading diffusion process.

    Args:
        sigma (float): Noise scale
    """
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def noise_coeff(self, t):
        """Returns the noise (diffusion) coefficient g(t) at time `t`."""
        return self.sigma ** t

    def sigma_func(self, t):
        """Returns the std dev of the Euclidean heat kernel at time `t`."""
        sigma = torch.tensor(self.sigma)
        numerator = sigma ** (2*t) - 1
        denominator = 2 * torch.log(sigma)
        return (numerator / denominator) ** 0.5

    def diffuse(self, x_0, t):
        r"""
        Diffuses input data `x_0` to a noise level projected
        forward to time step `t` according to the variance-expanding
        framework, where

        .. math::

            x_0 \rightarrow x_t = x_0 + \sigma_t \epsilon

        Args:
            x_0 (Tensor): Input data
            t (Tensor): Time step to which to diffuse
        """
        sigma_t = self.sigma_func(t)[:, None]
        eps = torch.randn_like(x_0)
        x_t = x_0 + sigma_t * eps
        return x_t


class VarianceExpandingDiffusionSUN(DiffusionProcess):
    """
    Variance-expanding diffusion on SU(N) group manifold.

    Args:
        sigma (float): Noise scale
    """
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def noise_coeff(self, t):
        """Returns the noise (diffusion) coefficient g(t) at time `t`."""
        return self.sigma ** t

    def sigma_func(self, t):
        """Returns the std dev of the Euclidean heat kernel at time `t`."""
        sigma = torch.tensor(self.sigma)
        numerator = sigma ** (2*t) - 1
        denominator = 2 * torch.log(sigma)
        return (numerator / denominator) ** 0.5

    def diffuse(self, U_0, t, n_iter=3):
        r"""
        Diffuses input data `U_0` to a noise level projected
        forward to time step `t` according to the variance-expanding
        framework, where

        .. math::

            U_0 \rightarrow U_t = exp(1j A(\sigma_t)) U_0

        Args:
            U_0 (Tensor): Input data
            t (Tensor): Time step to which to diffuse
        """
        batch_size = U_0.size(0)
        Nc, Nc_ = U_0.shape[-2:]
        assert Nc == Nc_, \
            f'U_0 must be a Nc x Nc matrix; got {Nc} x {Nc_}'
        sigma_t = self.sigma_func(t)
        xs = torch.tensor(sample_sun_hk(batch_size, Nc, width=sigma_t, n_iter=n_iter))
        V = random_un_haar_element(batch_size, Nc=Nc)
        A = V @ embed_diag(xs).to(V) @ adjoint(V)
        return matrix_exp(A) @ U_0, xs, V
