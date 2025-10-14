"""Utilities for ODE integration."""
from typing import Callable
import torch


def estimate_divergence(
    func: Callable[[torch.Tensor], torch.Tensor], 
    x: torch.Tensor, 
    *, 
    num_estimates: int = 1, 
    requires_grad: bool = True, 
    return_stats: Optional[bool] = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""
    Estimates the Jacobian trace of `func` at `x` using the
    Skilling-Hutchinson method:

    .. math::

        {\rm div}(f) = {\rm Tr}(J_f) = \mathbb{E}_{z}\left[z^\top J_f z\right],

    where :math:`z \in \{+1, -1\}^n` are Rademacher random vectors.

    Args:
        func (Callable): Function whose Jacobian trace is estimated
        x (Tensor): Input tensor (assumed to include batch axis)
        num_estimates (int): Number of random probe vectors. Larger values
            improve accuracy at higher computational cost
        requires_grad (bool): If True, keep graph for higher-order derivatives
        return_stats (bool): If True, return mean and standard error instead
            of a single tensor

    Returns:
        Tensor:
        - If `num_estimates == 1`: a single trace estimate
        - If `num_estimates > 1` and `return_stats == False`: mean estimate
        - If `return_stats == True`: both `mean` and `error` (standard error of the mean)
    """
    if x.ndim == 1:
        x = x.unsqueeze(-1)

    with torch.enable_grad():
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        y = func(x)

    grad_kwargs = {
        'retain_graph': (num_estimates > 1),
        'create_graph': requires_grad,
    }

    dims = tuple(range(1, x.ndim))
    batch_size = x.size(0)
    estimates = torch.zeros((num_estimates, batch_size), 
                            device=x.device, 
                            dtype=x.dtype)

    for n in range(num_estimates):
        v = 2*torch.randint_like(x, low=0, high=2).float() - 1  # Rademacher
        norm_sq = (v**2).mean(dim=dims)

        vjp = torch.autograd.grad(y, x, v, **grad_kwargs)[0]
        estimates[n] = (vjp * v).sum(dims) / norm_sq

    if num_estimates == 1:
        return estimates[0]

    mean = estimates.mean(dim=0)
    if return_stats:
        error = estimates.std(dim=0) / (num_estimates - 1)**0.5
        return mean, error
    return mean


def _test_trace_estimator():
    print('[Testing estimate_divergence...]')
    def example_function(x):
        return x ** 2
    def exact_divergence(x):
        return 2 * x.sum(dim=1)

    x = torch.randn(5, 3, requires_grad=True)
    est_div = estimate_divergence(example_function, x, num_estimates=10)
    exact_div = exact_divergence(x)

    assert torch.allclose(est_div, exact_div), \
        '[FAILED: Estimated and true divergences do not match]'
    print('[PASSED]')


if __name__ == '__main__': _test_trace_estimator()
