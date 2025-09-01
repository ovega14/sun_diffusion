from utils import wrap


def canonicalize_su2(thW):
    """
    Canonicalizes a set of SU(2) eigenangles :math:`(\theta_1, \theta_2)` by

        1.) Set :math:`\theta_1 = {\rm wrap}(|\theta|)`,
        2.) Set :math:`\theta_2 = -\theta_1`.

    Args:
        thW: Batch of non-canonicalized SU(2) eigenangles.

    Returns:
        Canonicalized batch of eigenangles that sum to zero
    """
    thW[..., 0] = wrap(thW[..., 0]).abs()  # map to [0, pi]
    thW[..., 1] = -thW[..., 0]  # second angle is negative first
    return thW
