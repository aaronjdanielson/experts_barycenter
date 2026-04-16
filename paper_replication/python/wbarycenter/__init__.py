"""
wbarycenter
===========
Robust aggregation of expert probability forecasts via Wasserstein barycenters.

Quick start
-----------
    import numpy as np
    from wbarycenter import dw_barycenter, cdf_median
    from wbarycenter import plot_aggregate, plot_cdfs, loo_influence, score_summary

    # Expert probability vectors: (n_experts, n_outcomes)
    data = np.array([...])
    labels = ["outcome A", "outcome B", ...]

    # Unordered outcomes (indicator distance)
    D = np.ones((K, K)) - np.eye(K)
    bc, _ = dw_barycenter(data, np.ones(n) / n, D)

    # Ordered outcomes — closed form, no solver needed
    bc = cdf_median(data)

    am = data.mean(axis=0)
    plot_aggregate(am, bc, labels)

References
----------
Danielson, A.J. and Amini, A.A. (2025).
"Robust Aggregation of Expert Probability Forecasts via Wasserstein Barycenters."
Journal of Forecasting.
"""

from wbarycenter.core import dw_barycenter
from wbarycenter.utils import (
    cdf_median,
    plot_aggregate,
    plot_cdfs,
    loo_influence,
    plot_loo,
    crps_ordered,
    brier_score,
    score_summary,
)

__all__ = [
    "dw_barycenter",
    "cdf_median",
    "plot_aggregate",
    "plot_cdfs",
    "loo_influence",
    "plot_loo",
    "crps_ordered",
    "brier_score",
    "score_summary",
]

__version__ = "0.1.0"
