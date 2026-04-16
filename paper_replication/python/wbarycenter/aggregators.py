"""
Fast closed-form aggregators for probability vectors on the simplex.

For the indicator-cost (unordered K-category) setting, the W1 barycenter
equals the coordinatewise median (Theorem 2.1).  These functions implement
BC, AM, TM, and permutation-averaged AM without any LP solver.

All functions:
    Input:  probs  (n, K)  array of probability vectors, rows sum to 1
    Output: (K,) aggregate probability vector summing to 1
"""

import numpy as np
from itertools import permutations
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Core aggregators
# ────────────────────────────────────────────────────────────────────────────

def bc_l1(probs: np.ndarray) -> np.ndarray:
    """
    ℓ¹ Wasserstein barycenter (indicator cost) = coordinatewise median.

    For unordered K-category spaces, the W1 barycenter under the indicator
    cost is the coordinatewise sample median (Theorem 2.1).  O(Kn log n).

    Parameters
    ----------
    probs : (n, K) array, rows sum to 1

    Returns
    -------
    (K,) array, sums to 1 (up to floating-point rounding)
    """
    bc = np.median(probs, axis=0)
    s = bc.sum()
    if s > 0:
        bc = bc / s
    return bc


def am(probs: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Arithmetic mean (equal or weighted).

    Parameters
    ----------
    probs   : (n, K) array
    weights : (n,) non-negative weights; if None, use equal weights
    """
    if weights is None:
        return probs.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return w @ probs


def tm(probs: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """
    α-trimmed mean.

    Trimming is on the scalar s^(i) = Σ_k k·p^(i)_k (expected rank under
    the ordering 1, 2, …, K).  This is the standard trimmed-mean baseline
    for ordered spaces; for unordered spaces it is ill-defined because the
    rank ordering is arbitrary — which is exactly what Proposition C1 shows.

    Parameters
    ----------
    probs : (n, K) array
    alpha : trim fraction from each tail (default 0.10 = 10% each side)

    Returns
    -------
    (K,) trimmed mean; returns AM if too few forecasters remain after trim
    """
    n, K = probs.shape
    ranks = np.arange(1, K + 1, dtype=float)
    scalars = probs @ ranks                     # (n,) expected-rank scores
    order = np.argsort(scalars)
    n_trim = int(np.floor(alpha * n))
    if n - 2 * n_trim < 1:
        return probs.mean(axis=0)
    keep = order[n_trim: n - n_trim] if n_trim > 0 else order
    result = probs[keep].mean(axis=0)
    s = result.sum()
    return result / s if s > 0 else result


def perm_am(probs: np.ndarray, max_perms: int = 24) -> np.ndarray:
    """
    Permutation-averaged arithmetic mean.

    Averages the AM across all K! permutations of the column ordering.
    For K=4, this is 24 permutations (exact).  For K>4, uses a random
    sample of max_perms permutations.

    This is the Pezeshkpour & Hruschka (2023) debiasing heuristic.
    Under the contamination model, this still retains residual bias
    εc(1-1/K) — unlike BC which eliminates it entirely.

    Parameters
    ----------
    probs     : (n, K) array
    max_perms : max number of permutations to average over

    Returns
    -------
    (K,) array
    """
    n, K = probs.shape
    K_fact = 1
    for i in range(1, K + 1):
        K_fact *= i

    if K_fact <= max_perms:
        perms = list(permutations(range(K)))
    else:
        rng = np.random.default_rng(42)
        perms_set = set()
        while len(perms_set) < max_perms:
            p = tuple(rng.permutation(K).tolist())
            perms_set.add(p)
        perms = list(perms_set)

    agg = np.zeros(K)
    for perm in perms:
        perm = list(perm)
        permuted = probs[:, perm]        # reorder columns
        avg = permuted.mean(axis=0)      # AM under this permutation
        # Map back to original column ordering
        inv = np.empty(K, dtype=int)
        inv[perm] = np.arange(K)
        agg += avg[inv]
    agg /= len(perms)
    s = agg.sum()
    return agg / s if s > 0 else agg


def adaptive_blend(probs: np.ndarray, lam: float) -> np.ndarray:
    """
    Adaptive blend: λ·BC + (1-λ)·AM.

    Parameters
    ----------
    probs : (n, K) array
    lam   : blend weight in [0, 1]; 1 = pure BC, 0 = pure AM
    """
    lam = float(np.clip(lam, 0.0, 1.0))
    return lam * bc_l1(probs) + (1 - lam) * am(probs)


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────

def brier_score(forecast: np.ndarray, correct_idx: int) -> float:
    """Brier score: mean squared error vs. one-hot encoding of correct answer."""
    K = len(forecast)
    onehot = np.zeros(K)
    onehot[correct_idx] = 1.0
    return float(np.mean((forecast - onehot) ** 2))


def panel_dispersion(probs: np.ndarray) -> float:
    """
    Mean pairwise TV distance within the panel.

    TV(p, q) = 0.5 * ||p - q||_1.  Used as the dispersion measure D_t.
    """
    n, K = probs.shape
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0.5 * np.sum(np.abs(probs[i] - probs[j]))
            count += 1
    return total / count


def panel_dispersion_fast(probs: np.ndarray) -> float:
    """Vectorized version of panel_dispersion for large panels."""
    n = probs.shape[0]
    if n < 2:
        return 0.0
    # Mean pairwise L1 = mean_i mean_{j>i} ||p_i - p_j||_1 / 2
    # Use broadcasting: (n,1,K) - (1,n,K) gives (n,n,K)
    diff = np.abs(probs[:, None, :] - probs[None, :, :])   # (n, n, K)
    l1 = diff.sum(axis=2)                                    # (n, n)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(0.5 * l1[mask].mean())


# ────────────────────────────────────────────────────────────────────────────
# Equivariance test (unit-test helper)
# ────────────────────────────────────────────────────────────────────────────

def test_equivariance(probs: np.ndarray, perm: list[int],
                      tol: float = 1e-10) -> dict:
    """
    Verify BC(σ·probs) = σ·BC(probs) and check TM is NOT equivariant.

    Returns dict with 'bc_equivariant' (bool) and 'tm_equivariant' (bool).
    """
    K = probs.shape[1]
    perm = list(perm)

    # Permute columns
    probs_perm = probs[:, perm]

    # BC
    bc_orig = bc_l1(probs)
    bc_perm = bc_l1(probs_perm)
    bc_expected = bc_orig[perm]
    bc_eq = bool(np.allclose(bc_perm, bc_expected, atol=tol))

    # TM
    tm_orig = tm(probs)
    tm_perm = tm(probs_perm)
    tm_expected = tm_orig[perm]
    tm_eq = bool(np.allclose(tm_perm, tm_expected, atol=tol))

    return {
        "bc_equivariant": bc_eq,
        "tm_equivariant": tm_eq,
        "bc_perm": bc_perm,
        "bc_expected": bc_expected,
        "tm_perm": tm_perm,
        "tm_expected": tm_expected,
    }
