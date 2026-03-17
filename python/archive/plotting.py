"""
Plotting utilities for barycenter transport matrices.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_transport_matrices(gamma: np.ndarray, figsize=None):
    """
    Plot each optimal transport matrix as a heatmap.

    Parameters
    ----------
    gamma : (n, K, K) array
        Transport matrices from dw_barycenter.
    """
    n = gamma.shape[0]
    if figsize is None:
        figsize = (3 * n, 3)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(gamma[i], aspect="auto")
        ax.set_title(f"Expert {i+1}")
        ax.set_xlabel("Barycenter state")
        ax.set_ylabel("Expert state")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig
