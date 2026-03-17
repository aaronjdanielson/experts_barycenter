"""
Compositional data analysis utilities (Aitchison geometry).
"""

import numpy as np


def composition_var(data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute Aitchison's variation matrix and total variance.

    Parameters
    ----------
    data : (n, K) array
        Each row is a probability vector.

    Returns
    -------
    var_matrix : (n, n) array
        var_matrix[i, j] = Var(log(data[i] / data[j]) / sqrt(2))
    total_var : float
        Mean of column sums of var_matrix.
    """
    n = data.shape[0]
    var_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ratio = np.log(data[i] / data[j]) / np.sqrt(2)
            var_matrix[i, j] = np.var(ratio)
    total_var = np.sum(var_matrix) / n
    return var_matrix, total_var
