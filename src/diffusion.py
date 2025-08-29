import torch
from abc import abstractmethod, ABC


__all__ = [
    'VarianceExpandingDiffusion'
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
