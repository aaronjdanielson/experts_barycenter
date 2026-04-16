"""
Utilities for working with the Wasserstein barycenter aggregator.

Typical workflow
----------------
    from wbarycenter import dw_barycenter, plot_aggregate, plot_cdfs, loo_influence, crps_ordered

    data    = ...          # (n, K) array of expert probability vectors
    labels  = [...]        # K outcome labels
    D       = np.ones((K, K)) - np.eye(K)   # indicator distance (unordered)
    weights = np.ones(n) / n

    bc, _  = dw_barycenter(data, weights, D)
    am     = data.mean(axis=0)

    plot_aggregate(am, bc, labels)
    shifts = loo_influence(data, weights, D)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from wbarycenter.core import dw_barycenter


# ---------------------------------------------------------------------------
# 1. Visualize the aggregate distribution
# ---------------------------------------------------------------------------

def plot_aggregate(
    am: np.ndarray,
    bc: np.ndarray,
    labels: list[str],
    *,
    title: str = "Arithmetic mean vs. Wasserstein barycenter",
    savepath: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing the arithmetic mean and barycenter.

    Parameters
    ----------
    am, bc  : (K,) arrays
    labels  : K outcome labels
    savepath: if given, save PDF + PNG
    ax      : if given, draw into this Axes; otherwise create a new figure
    """
    K = len(labels)
    x = np.arange(K)
    w = 0.38

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, K), 4))
    else:
        fig = ax.get_figure()

    ax.bar(x - w / 2, am * 100, w, label="Arithmetic mean",
           color="#d62728", alpha=0.8)
    ax.bar(x + w / 2, bc * 100, w, label="Wasserstein barycenter",
           color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Probability (%)")
    ax.set_title(title)
    ax.legend(fontsize=9)

    if own_fig:
        fig.tight_layout()
    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(savepath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. CDF overlay (ordered outcome spaces only)
# ---------------------------------------------------------------------------

def plot_cdfs(
    data: np.ndarray,
    am: np.ndarray,
    bc: np.ndarray,
    labels: list[str],
    *,
    title: str = "Individual and aggregate CDFs",
    savepath: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot individual CDFs (grey) plus aggregate CDFs for AM (red) and BC (blue).

    Only meaningful when the outcome space is *ordered* (e.g. histogram bins).

    Parameters
    ----------
    data : (n, K) array of expert probability vectors
    am, bc : (K,) aggregate probability vectors
    labels : K bin labels, in ascending order
    """
    n, K = data.shape
    x = np.arange(K)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, K), 4))
    else:
        fig = ax.get_figure()

    for i in range(n):
        ax.step(x, np.cumsum(data[i]) * 100, where="post",
                color="gray", alpha=0.25, lw=0.8)

    ax.step(x, np.cumsum(am) * 100, where="post",
            color="#d62728", lw=2.5, label="Arithmetic mean")
    ax.step(x, np.cumsum(bc) * 100, where="post",
            color="#1f77b4", lw=2.5,
            label="Wasserstein barycenter\n(= component-wise CDF median)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative probability (%)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left")

    if own_fig:
        fig.tight_layout()
    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(savepath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Leave-one-out influence
# ---------------------------------------------------------------------------

def loo_influence(
    data: np.ndarray,
    weights: np.ndarray,
    D: np.ndarray,
    *,
    usecdf_median: bool = False,
) -> dict:
    """
    Compute the leave-one-out L1 shift in both aggregates for each expert.

    Parameters
    ----------
    data    : (n, K) array
    weights : (n,) weights (equal weights recommended)
    D       : (K, K) distance matrix
    usecdf_median : if True, use the closed-form CDF median for the BC
                     instead of the LP solver (only valid for ordered spaces
                     with D = absolute rank difference)

    Returns
    -------
    dict with keys:
        'am_shifts' : (n,) L1 shift in AM when expert i is removed
        'bc_shifts' : (n,) L1 shift in BC when expert i is removed
        'am_full'   : (K,) full-sample AM
        'bc_full'   : (K,) full-sample BC
    """
    n, K = data.shape

    am_full = data.mean(axis=0)
    if usecdf_median:
        bc_full = cdf_median(data)
    else:
        bc_full, _ = dw_barycenter(data, weights, D)

    am_shifts = np.zeros(n)
    bc_shifts = np.zeros(n)

    for i in range(n):
        loo = np.delete(data, i, axis=0)
        w_loo = np.ones(n - 1) / (n - 1)

        am_loo = loo.mean(axis=0)
        if usecdf_median:
            bc_loo = cdf_median(loo)
        else:
            bc_loo, _ = dw_barycenter(loo, w_loo, D)

        am_shifts[i] = np.sum(np.abs(am_loo - am_full))
        bc_shifts[i] = np.sum(np.abs(bc_loo - bc_full))

    return {
        "am_shifts": am_shifts,
        "bc_shifts": bc_shifts,
        "am_full": am_full,
        "bc_full": bc_full,
    }


def plot_loo(
    loo_result: dict,
    expert_labels: list[str] | None = None,
    *,
    title: str = "Leave-one-out influence",
    savepath: str | Path | None = None,
) -> plt.Figure:
    """
    Bar chart of LOO L1 shifts, sorted by AM influence.

    Parameters
    ----------
    loo_result : output of loo_influence()
    expert_labels : optional list of n expert names
    """
    am_shifts = loo_result["am_shifts"]
    bc_shifts = loo_result["bc_shifts"]
    n = len(am_shifts)
    order = np.argsort(am_shifts)[::-1]

    if expert_labels is None:
        expert_labels = [f"Expert {i + 1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 4))
    x = np.arange(n)
    ax.bar(x - 0.2, am_shifts[order] * 100, 0.38,
           label="Arithmetic mean", color="#d62728", alpha=0.8)
    ax.bar(x + 0.2, bc_shifts[order] * 100, 0.38,
           label="Wasserstein barycenter", color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([expert_labels[i] for i in order],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"$\ell^1$ shift in aggregate (pp)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(savepath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Scoring against a realized outcome
# ---------------------------------------------------------------------------

def crps_ordered(forecast: np.ndarray, outcome_bin: int) -> float:
    """
    Continuous Ranked Probability Score for an ordered categorical forecast.

    CRPS(F, y) = sum_{k=0}^{K-2} (F(k) - 1{y <= k})^2

    where F is the CDF of `forecast` and y is the realized bin index.

    Parameters
    ----------
    forecast    : (K,) probability vector (need not be the barycenter)
    outcome_bin : integer in {0, ..., K-1}, the realized bin (0-indexed)

    Returns
    -------
    float, lower is better
    """
    K = len(forecast)
    F = np.cumsum(forecast)          # CDF at bins 0, 1, ..., K-1
    indicator = np.arange(K) >= outcome_bin   # 1{y <= k} for k=0,...,K-1
    return float(np.sum((F[:-1] - indicator[:-1].astype(float)) ** 2))


def brier_score(forecast: np.ndarray, outcome_bin: int) -> float:
    """
    Brier score: mean squared error between forecast probabilities and
    the one-hot encoding of the realized outcome.

    Parameters
    ----------
    forecast    : (K,) probability vector
    outcome_bin : integer in {0, ..., K-1}

    Returns
    -------
    float, lower is better
    """
    K = len(forecast)
    onehot = np.zeros(K)
    onehot[outcome_bin] = 1.0
    return float(np.mean((forecast - onehot) ** 2))


def score_summary(
    am: np.ndarray,
    bc: np.ndarray,
    outcome_bin: int,
    ordered: bool = True,
) -> dict:
    """
    Return a dict of scores for the AM and BC against a realized outcome.

    Parameters
    ----------
    am, bc      : (K,) aggregate probability vectors
    outcome_bin : realized bin index (0-indexed)
    ordered     : if True, also compute CRPS (requires ordered outcome space)
    """
    result = {
        "brier_am": brier_score(am, outcome_bin),
        "brier_bc": brier_score(bc, outcome_bin),
        "log_score_am": -np.log(am[outcome_bin] + 1e-12),
        "log_score_bc": -np.log(bc[outcome_bin] + 1e-12),
    }
    if ordered:
        result["crps_am"] = crps_ordered(am, outcome_bin)
        result["crps_bc"] = crps_ordered(bc, outcome_bin)
    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def cdf_median(data: np.ndarray) -> np.ndarray:
    """Closed-form W1 barycenter for ordered spaces: component-wise CDF median."""
    cdf_med = np.median(np.cumsum(data, axis=1), axis=0)
    pmf = np.diff(np.concatenate([[0.0], cdf_med]))
    pmf = np.maximum(pmf, 0.0)
    return pmf / pmf.sum()
