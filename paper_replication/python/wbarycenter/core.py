"""
Discrete Wasserstein Barycenter for Probability Vectors
"""

import numpy as np
import cvxpy as cp


def dw_barycenter(data: np.ndarray, weights: np.ndarray, D: np.ndarray):
    """
    Compute the discrete Wasserstein barycenter of a set of probability vectors.

    Parameters
    ----------
    data : (n, K) array
        Row i is the i-th probability vector over K states.
    weights : (n,) array
        Non-negative weights summing to 1, one per expert.
    D : (K, K) array
        Distance matrix between the K states.

    Returns
    -------
    bcenter : (K,) array
        The barycenter probability vector.
    gamma : (n, K, K) array
        Optimal transport matrices; gamma[i] is the joint distribution
        coupling data[i] to bcenter.
    """
    n, K = data.shape
    assert weights.shape == (n,), "weights must have length n"
    assert D.shape == (K, K), "D must be K x K"

    b = cp.Variable(K, nonneg=True)
    gammas = [cp.Variable((K, K), nonneg=True) for _ in range(n)]

    objective = cp.Minimize(
        sum(weights[i] * cp.sum(cp.multiply(gammas[i], D)) for i in range(n))
    )

    constraints = [cp.sum(b) == 1]
    for i in range(n):
        constraints += [
            cp.sum(gammas[i], axis=1) == data[i],   # row marginals match expert i
            cp.sum(gammas[i], axis=0) == b,           # col marginals match barycenter
        ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver failed with status: {prob.status}")

    gamma_out = np.stack([g.value for g in gammas], axis=0)
    return b.value, gamma_out
