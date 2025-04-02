"""
Parameter initialization functions for the SourceID-NMF algorithm.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("sourceid_nmf.initialization")


def standardization(matrix: np.ndarray) -> np.ndarray:
    """
    Standardize a matrix by column.

    Args:
        matrix: Input matrix to be standardized

    Returns:
        Standardized matrix where each column sums to 1
    """
    logger.debug("Standardizing matrix")
    matrix_sum = np.sum(matrix, axis=0)
    matrix = matrix / (matrix_sum[np.newaxis, :] + 1e-16)  # Add small epsilon to avoid division by zero
    return matrix


def unknown_initialize(sources: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """
    Initialize the unknown source component.

    Args:
        sources: Source data matrix
        sinks: Sink data matrix

    Returns:
        Initialized unknown source component
    """
    logger.debug("Initializing unknown source component")
    sinks = sinks.reshape((sinks.shape[0],))

    # Calculate the difference between sinks and sum of sources
    unknown_pre = sinks - np.sum(sources, axis=1)
    unknown_zero = np.zeros((unknown_pre.shape[0],))

    # Take the maximum of unknown_pre and zero to ensure non-negativity
    unknown_init = np.vstack((unknown_pre, unknown_zero))
    unknown_init = np.max(unknown_init, axis=0)
    unknown_init = unknown_init.reshape((-1, 1))

    # Standardize the unknown component
    unknown_init = standardization(unknown_init)
    return unknown_init


def parameters_initialize(
    sources: np.ndarray,
    sinks: np.ndarray,
    unknown_init: np.ndarray,
    weight_factor: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize parameters for the NMF model.

    Args:
        sources: Source data matrix
        sinks: Sink data matrix
        unknown_init: Initialized unknown source component
        weight_factor: Weighting matrix factor

    Returns:
        Tuple of initialized parameters: (x, y, w, a, i, w_plus, h_plus, alpha_w, alpha_h)
    """
    logger.debug(f"Initializing NMF parameters with weight factor {weight_factor}")

    y = sources.copy()
    n = sources.shape[0]  # Number of taxa
    k = sources.shape[1]  # Number of sources

    x = sinks  # Sink data
    w = y.copy()  # Initialize W with Y

    # Initialize H with uniform distribution
    h = np.zeros((k + 1, 1)) + 1 / (k + 1)

    # Create weighting matrix
    a = np.ones((n, k)) * weight_factor  # Previous K columns in the weighted matrix are one
    zeros = np.zeros((n, 1), dtype=y.dtype)
    a = np.hstack((a, zeros))  # The (K+1)-th column has zero values

    # Add unknown source to y and w
    y = np.hstack((y, zeros))
    w = np.hstack((w, unknown_init))

    # Initialize auxiliary variables
    w_plus = w.copy()
    h_plus = h.copy()

    # Initialize Lagrangian multipliers
    alpha_w = np.zeros((n, k + 1))
    alpha_h = np.zeros((k + 1, 1))

    # Identity matrix for updates
    i = np.identity(k + 1)

    return x, y, w, a, i, w_plus, h_plus, alpha_w, alpha_h