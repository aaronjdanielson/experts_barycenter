"""
Dirichlet sampling and MLE.
"""

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize


def sample_dirichlet(alpha: np.ndarray, n: int) -> np.ndarray:
    """
    Draw n samples from Dirichlet(alpha).

    Returns (n, K) array where each row is a probability vector.
    """
    return np.random.dirichlet(alpha, size=n)


def dirichlet_mle(data: np.ndarray) -> np.ndarray:
    """
    Maximum likelihood estimate of Dirichlet concentration parameters.

    Parameters
    ----------
    data : (n, K) array
        Each row is a probability vector (compositional data).

    Returns
    -------
    alpha_hat : (K,) array
        MLE of the concentration parameter vector.
    """
    n, K = data.shape
    log_p_bar = np.mean(np.log(data), axis=0)

    def neg_log_likelihood(alpha):
        if np.any(alpha <= 0):
            return np.inf
        a0 = np.sum(alpha)
        return -(n * (gammaln(a0) - np.sum(gammaln(alpha)))
                 + np.sum((alpha - 1) * log_p_bar) * n)

    def grad(alpha):
        if np.any(alpha <= 0):
            return np.zeros_like(alpha)
        a0 = np.sum(alpha)
        return -(n * (digamma(a0) - digamma(alpha)) + n * log_p_bar)

    alpha0 = np.ones(K) * 0.5
    result = minimize(neg_log_likelihood, alpha0, jac=grad,
                      method="L-BFGS-B",
                      bounds=[(1e-6, None)] * K)
    return result.x
