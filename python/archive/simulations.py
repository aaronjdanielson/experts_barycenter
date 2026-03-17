"""
Simulation study: compare arithmetic mean vs. Wasserstein barycenter.

Replicates the core analysis from paper_results.m.
"""

import numpy as np
from barycenter import dw_barycenter
from dirichlet import sample_dirichlet
from composition import composition_var


def run_robustness_simulation(
    K: int = 6,
    n_regular: int = 10,
    alpha_regular=None,
    alpha_outlier=None,
    n_outlier: int = 1,
    R: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compare arithmetic mean and Wasserstein barycenter under outlier contamination.

    Draws R datasets each containing n_regular experts from a uniform Dirichlet
    and n_outlier extreme experts, then computes the arithmetic mean and barycenter
    for each dataset.

    Returns
    -------
    dict with keys:
        ameans  : (R, K) array of arithmetic means
        bmeans  : (R, K) array of barycenters
    """
    rng = np.random.default_rng(seed)

    if alpha_regular is None:
        alpha_regular = np.ones(K)
    if alpha_outlier is None:
        alpha_outlier = np.array([1, 1, 1, 1, 1.5, 10.0]) ** 4

    n_total = n_regular + n_outlier
    weights = np.ones(n_total) / n_total
    D = np.ones((K, K)) - np.eye(K)  # indicator (Hamming) distance

    ameans = np.zeros((R, K))
    bmeans = np.zeros((R, K))

    for r in range(R):
        dat_regular = rng.dirichlet(alpha_regular, size=n_regular)
        dat_outlier = rng.dirichlet(alpha_outlier, size=n_outlier)
        dat = np.vstack([dat_regular, dat_outlier])

        ameans[r] = dat.mean(axis=0)
        bcenter, _ = dw_barycenter(dat, weights, D)
        bmeans[r] = bcenter

    return {"ameans": ameans, "bmeans": bmeans}


def summarize(results: dict) -> None:
    ameans = results["ameans"]
    bmeans = results["bmeans"]

    print("=== Arithmetic Mean ===")
    print(f"  Column means:    {ameans.mean(axis=0).round(4)}")
    print(f"  Column maxima:   {ameans.max(axis=0).round(4)}")
    print(f"  Column minima:   {ameans.min(axis=0).round(4)}")
    print(f"  Column variances:{ameans.var(axis=0).round(6)}")

    print("\n=== Wasserstein Barycenter ===")
    print(f"  Column means:    {bmeans.mean(axis=0).round(4)}")
    print(f"  Column maxima:   {bmeans.max(axis=0).round(4)}")
    print(f"  Column minima:   {bmeans.min(axis=0).round(4)}")
    print(f"  Column variances:{bmeans.var(axis=0).round(6)}")

    # fraction of samples where AM gives more extreme value on outlier state (K=6 → state 5)
    outlier_state = -1
    frac = np.mean(ameans[:, outlier_state] > bmeans[:, outlier_state])
    print(f"\nAM > BC on outlier state: {frac:.0%} of samples")


if __name__ == "__main__":
    results = run_robustness_simulation(R=100)
    summarize(results)
