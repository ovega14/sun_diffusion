"""Analysis library from github.com:gkanwar/lattlib.git"""
import numpy as np

from typing import Optional, Any, Union, Callable, Iterable
from numpy.typing import ArrayLike, NDArray

from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer


def add_errorbar(
    trace: tuple[ArrayLike, ArrayLike],
    *, 
    ax: Axes, 
    xs: Optional[ArrayLike] = None, 
    off: float = 0.0, 
    flip: bool = False,
    **kwargs: Any
) -> ErrorbarContainer:
    """
    Plots a mean curve with error bars on a given axis.

    This function wraps around `matplotlib.axes.Axes.errorbar` with several
    added options for convenience.

    Args:
        trace (tuple): Tuple of arrays containing `(mean, err)`
        ax (Axes): Axis on which to plot the curve
        xs (Optional): X coordinates. If `None`, defaults to integer indices.
        off (float): Constant offset added to `xs`. Default: `0.0`
        flip (bool): Whether to swap the x and y axes. Default: `False`

    Returns:
        A Matplotlib `ErrorBarContainer`
    """
    mean, err = trace
    
    if xs is None:
        xs = np.arange(len(mean), dtype=np.float64)
    else:
        xs = np.array(xs).astype(np.float64)
    xs += off
    
    if flip:
        return ax.errorbar(mean, xs, xerr=err, **kwargs)
    return ax.errorbar(xs, mean, yerr=err, **kwargs)


def add_errorbar_fill(
    trace: tuple[ArrayLike, ArrayLike],
    *,
    ax: Axes,
    xs: Optional[ArrayLike] = None,
    off: float = 0.0,
    **kwargs: Any
) -> None:
    """
    Plots a mean curve filled between the error bands.

    Args:
        trace (tuple): Tuple of arrays containing `(mean, err)`
        ax (Axes): Axis on which to plot the curve
        xs (Optional): X coordinates. If `None`, defaults to integer indices.
        off (float): Constant offset added to `xs`. Default: `0.0`
    """
    mean, err = trace
    
    if xs is None:
        xs = np.arange(len(mean), dtype=np.float64)
    else:
        xs = np.array(xs).astype(np.float64)
    xs += off
    
    kwargs_stripped = {}
    if 'color' in kwargs:
        kwargs_stripped['color'] = kwargs['color']
    ax.fill_between(xs, mean-err, mean+err, alpha=0.8, **kwargs_stripped)
    ax.plot(xs, mean, **kwargs)


def lighten_color(
    color: Union[str, tuple[float, float, float]], 
    amount: float = 0.5
) -> tuple[float, float, float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    .. note: From SO: 37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:

    .. code-block:: python

        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)

    Args:
        color (str, tuple): Color to lighten, can be color/hex string or RGB 
        amount (float): Factor by which to lighten `color`. Default: `0.5`

    Returns:
        Signature of lightened color as an RGB tuple
    """
    import matplotlib.colors as mc
    import colorsys
    
    try:
        c = mc.cnames[color]
    except (KeyError, TypeError):
        c = color
    
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount*(1 - c[1]), c[2])


# =============================================================================
#  Standard Bootstrapping Functions
# =============================================================================
def mean(x):
    """Averages input array `x` over axis 0."""
    return np.mean(x, axis=0)


def rmean(x):
    """Gets the real part of the mean of `x` over axis 0."""
    return np.real(np.mean(x, axis=0))


def imean(x):
    """Gets the imaginary part of the mean of `x` over axis 0."""
    return np.imag(np.mean(x, axis=0))


def amean(x):
    """Gets the magnitude of the mean of `x` over axis 0."""
    return np.abs(np.mean(x, axis=0))


def log_meff(x: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""
    Computes the effective mass using the :math:`\log` estimator.

    Estimates the mass from adjacent correlator ratios, as in the infinite
    volume / early-time approximation. Effective mass is given by

    .. math::

        m_{\rm eff}^\log (t) = \log\left[\frac{C(t)}{C(t+1)}\right].

    Args:
        x (NDArray): Ensemble of correlation functions

    Returns:
        Estimate for the effective mass
    """
    corr = rmean(x)
    return np.log(corr[:-1] / corr[1:])


def acosh_meff(x: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""
    Computes the effective mass using the :math:`\cosh` estimator.

    Estimates the mass assuming :math:`\cosh` behavior from periodic boundary
    conditions. Effective mass is given by

    .. math::

        m_{\rm eff}^\cosh (t) = \arccosh\left[
            \frac{C(t-1) + C(t+1)}{2C(t)}\right].

    Args:
        x (NDArray): Ensemble of correlation functions

    Returns:
        Estimate for the effective mass
    """
    corr = rmean(x)
    return np.arccosh((corr[:-2] + corr[2:]) / (2*corr[1:-1]))


def make_stn_f(
    *, 
    N_inner_boot: int, 
    f: Callable[[NDArray[np.generic]], NDArray[np.generic]]
) -> Callable[[NDArray[np.generic]], NDArray[np.float64]]:
    """
    Builds a function that bootstraps input data using a function or estimator
    `f` and computes its signal-to-noise ratio.

    Args:
        N_inner_boot (int): Number of bootstrap samples
        f (Callable): Statistic or function applied inside the bootstrap

    Returns:
        stn (Callable): Function to compute signal-to-noise using `f`
    """
    def stn(x):
        mean, err = bootstrap(x, Nboot=N_inner_boot, f=f)
        stn = np.abs(mean) / np.abs(err)
        return stn
    return stn


# =============================================================================
#  Bootstrapping Framework
# =============================================================================
def bootstrap_gen(
    *samples: NDArray[np.generic], 
    Nboot: int, 
    seed: Optional[int] = None
) -> Iterable[tuple[NDArray[np.generic], ...]]:
    """
    Bootstrap resampling generator.

    Takes one or more aligned datasets (`samples`) and repeatedly resamples
    them *with replacement*, then yields resampled versions of *all* samples.

    Args:
        samples (NDArray): Arrays, assumed to have same length along axis 0
        Nboot (int): Number of bootstrap resamples
        seed (int, Optional): RNG seed

    Returns:
        Generator that yields bootstrap-resampled data when iterated
    """
    rng = np.random.default_rng(seed=seed)
    n = len(samples[0])
    for i in range(Nboot):
        inds = rng.integers(n, size=n)
        yield tuple(s[inds] for s in samples)


def bootstrap(
    *samples: NDArray[np.generic], 
    Nboot: int, 
    f: Callable[..., NDArray[np.generic]], 
    bias_correct: bool = False, 
    seed: Optional[int] = None
) -> tuple[NDArray[np.generic], NDArray[np.float64]]:
    r"""
    Computes a bootstrap estimate of a statistic `f`.

    Bias correction can optionally be applied to the estimate by computing 
    :math:`f - f_{\rm bias} = 2f - \langle f^* \rangle`.

    Args:
        samples (NDArray): Arrays, assumed to have same length along axis 0
        Nboot (int): Number of bootstrap resamples
        f (Callable): Statistic applied to each bootstrap sample
        bias_correct (bool): Whether to bias-correct result. Default: `False`
        seed (int, Optional): RNG seed

    Returns:
        boot_mean: Bootstrap estimate of :math:`\langle f \rangle`
        boot_err: Bootstrap uncertainty (standard deviation)
    """
    boots = []
    for x in bootstrap_gen(*samples, Nboot=Nboot, seed=seed):
        boots.append(f(*x))
    boot_mean, boot_err = np.mean(boots, axis=0), np.std(boots, axis=0)
    if bias_correct:
        full_mean = f(*samples)
        corrected_mean = 2*full_mean - boot_mean
        return corrected_mean, boot_err
    return boot_mean, boot_err


# TODO: Replace with np.cov? Only difference is that covar_from_boots allows
# higher-order shapes, but this is probably not useful.
def covar_from_boots(
    boots: Iterable[NDArray[np.generic]]
) -> NDArray[np.float64]:
    """Computes the sample covariance matrix of a set of bootstrap copies."""
    boots = np.array(boots)
    Nboot = boots.shape[0]
    means = np.mean(boots, axis=0, keepdims=True)
    deltas = boots - means
    return np.tensordot(deltas, deltas, axes=(0,0)) / (Nboot - 1)


def shrink_covar(
    covar: NDArray[np.float64], 
    *, 
    lam: float
) -> NDArray[np.float64]:
    """Applies linear shrinkage to a covariance matrix."""
    assert len(covar.shape) == 2 and covar.shape[0] == covar.shape[1],
        'covar must be a square 2D matrix'
    diag_covar = np.diag(covar) * np.identity(covar.shape[0])
    return (1-lam) * covar + lam * diag_covar


def bin_data(x, *, binsize, silent_trunc=True):
    x = np.array(x)
    if silent_trunc:
        x = x[:(x.shape[0] - x.shape[0]%binsize)]
    else:
        assert x.shape[0] % binsize == 0
    ts = np.arange(0, x.shape[0], binsize) # left endpoints of bins
    x = np.reshape(x, (-1, binsize) + x.shape[1:])
    return ts, np.mean(x, axis=1)


# Autocorrelations
def compute_autocorr(Os, *, tmax, vacsub=True):
    assert np.allclose(np.imag(Os), 0.0)
    if vacsub:
        dOs = Os - np.mean(Os)
    else:
        dOs = Os
    Gamma = np.array([np.mean((dOs[t:] - dOs[:-t])**2) for t in range(1,tmax)])
    Gamma = np.insert(Gamma, 0, np.mean(dOs**2))
    rho = Gamma / Gamma[0]
    return rho


def compute_tint(Os, *, tmax, vacsub=True):
    rho = compute_autocorr(Os, tmax=tmax, vacsub=vacsub)
    tint = 0.5 + np.cumsum(rho[1:])
    return tint


def self_consistent_tint(tints, *, W=4):
    after_W_tint = tints < np.arange(len(tints)) / W
    if not np.any(after_W_tint):
        # print('WARNING: self-consistent tint condition never satisfied, returning last tint')
        return tints[-1]
    i = np.argmax(after_W_tint)
    return tints[i]
