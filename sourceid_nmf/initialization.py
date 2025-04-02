"""
Parameter initialization functions for the SourceID-NMF algorithm.

This optimized version includes:
1. Faster matrix operations
2. Memory usage optimizations
3. Optimized data type selection
4. Fixed numerical stability issues
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

    # Get column sums
    matrix_sum = np.sum(matrix, axis=0)

    # Handle zero sums to avoid division by zero
    zero_cols = matrix_sum < 1e-16
    if np.any(zero_cols):
        # Create a copy to avoid modifying the original
        matrix = matrix.copy()
        # For zero-sum columns, set equal distribution
        n_rows = matrix.shape[0]
        for col_idx in np.where(zero_cols)[0]:
            matrix[:, col_idx] = 1.0 / n_rows
        # Recalculate sums
        matrix_sum = np.sum(matrix, axis=0)

    # Standardize
    return matrix / matrix_sum[np.newaxis, :]


def unknown_initialize(sources: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    """
    Initialize the unknown source component with optimized memory usage.

    Args:
        sources: Source data matrix
        sinks: Sink data matrix

    Returns:
        Initialized unknown source component
    """
    logger.debug("Initializing unknown source component")

    # Reshape sinks to vector
    if sinks.shape[1] == 1:
        sinks_flat = sinks.reshape(-1)
    else:
        sinks_flat = sinks.reshape((sinks.shape[0],))

    # Calculate the difference between sinks and sum of sources
    sources_sum = np.sum(sources, axis=1)
    unknown_pre = sinks_flat - sources_sum

    # Ensure non-negativity - more efficient than stacking and taking max
    unknown_init = np.maximum(0, unknown_pre).reshape(-1, 1)

    # Standardize the unknown component
    return standardization(unknown_init)


def parameters_initialize(
    sources: np.ndarray,
    sinks: np.ndarray,
    unknown_init: np.ndarray,
    weight_factor: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize parameters for the NMF model with memory optimizations.

    Args:
        sources: Source data matrix
        sinks: Sink data matrix
        unknown_init: Initialized unknown source component
        weight_factor: Weighting matrix factor

    Returns:
        Tuple of initialized parameters: (x, y, w, a, i, w_plus, h_plus, alpha_w, alpha_h)
    """
    logger.debug(f"Initializing NMF parameters with weight factor {weight_factor}")

    # Get dimensions
    n = sources.shape[0]  # Number of taxa
    k = sources.shape[1]  # Number of sources

    # Use copy only where needed
    y = sources.copy()
    x = sinks  # No need to copy
    w = y.copy()  # Need copy as we'll modify it

    # Use float64 for numerical stability
    dtype = np.float64

    # Initialize H with better than uniform distribution
    h = np.ones((k + 1, 1), dtype=dtype) / (k + 1)

    # Try to use source contributions as a hint for initial h values if appropriate
    sources_total = np.sum(sources)
    if sources_total > 0:  # Ensure we don't divide by zero
        try:
            source_sums = np.sum(sources, axis=0)
            # Only use this approach if we have meaningful contributions
            if np.any(source_sums > 0):
                # Calculate contribution of each source
                source_contributions = source_sums / sources_total
                # Scale to leave room for unknown source
                source_contributions = source_contributions * (1 - 1/(k+1))

                h = np.zeros((k + 1, 1), dtype=dtype)
                h[:-1, 0] = source_contributions
                h[-1, 0] = 1/(k+1)  # Give some weight to unknown source

                # Normalize to ensure sum is 1
                h_sum = np.sum(h)
                if h_sum > 0:
                    h = h / h_sum
                else:
                    # Fallback to uniform if something went wrong
                    h = np.ones((k + 1, 1), dtype=dtype) / (k + 1)
        except Exception as e:
            logger.warning(f"Error during H initialization, using uniform: {e}")
            # Fallback to uniform distribution
            h = np.ones((k + 1, 1), dtype=dtype) / (k + 1)

    # Create weighting matrix more efficiently
    a = np.ones((n, k), dtype=dtype) * weight_factor
    zeros = np.zeros((n, 1), dtype=dtype)
    a = np.hstack((a, zeros))  # The (K+1)-th column has zero values

    # Add unknown source to y and w
    y = np.hstack((y, zeros))
    w = np.hstack((w, unknown_init))

    # Initialize auxiliary variables - no need for deep copies
    w_plus = w.copy()
    h_plus = h.copy()

    # Initialize Lagrangian multipliers with zeros
    alpha_w = np.zeros((n, k + 1), dtype=dtype)
    alpha_h = np.zeros((k + 1, 1), dtype=dtype)

    # Identity matrix for updates - create once
    i = np.identity(k + 1, dtype=dtype)

    return x, y, w, a, i, w_plus, h_plus, alpha_w, alpha_h