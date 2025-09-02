def eucl_log_hk(x, *, width):
    """Log density of Euclidean heat kernel with width `width`."""
    return -(x**2).sum(-1) / (2 * sigma**2)


def eucl_score_hk(x, *, width):
    """Analytical score function for the Euclidean heat kernel with width `width`."""
    return -x / sigma**2
