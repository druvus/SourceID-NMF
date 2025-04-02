"""
Data clustering module for SourceID-NMF.

This module provides functionality for clustering similar source profiles
using Jensen-Shannon divergence.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger("sourceid_nmf.clustering")


def jsd_correlation_matrix(source_data: np.ndarray) -> np.ndarray:
    """
    Calculate the Jensen-Shannon divergence matrix between all pairs of sources.

    Args:
        source_data: Source data matrix of shape (n_taxa, n_sources)

    Returns:
        Matrix of Jensen-Shannon divergences between all pairs of sources
    """
    logger.debug("Calculating Jensen-Shannon divergence correlation matrix")

    num_sources = source_data.shape[1]
    jsd_matrix = np.zeros((num_sources, num_sources))

    # Calculate pairwise Jensen-Shannon divergences
    for i in range(num_sources - 1):
        for j in range(i + 1, num_sources):
            value = jensenshannon(source_data[:, i], source_data[:, j])
            jsd_matrix[i, j] = value
            jsd_matrix[j, i] = value  # Matrix is symmetric

    return jsd_matrix


def min_value(corr_matrix: np.ndarray) -> Tuple[float, int, int]:
    """
    Find the minimum value in the correlation matrix and its indices.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Tuple of (minimum value, row index, column index)
    """
    min_val = np.inf
    row_index, col_index = 0, 0

    # Find the minimum value in the matrix (excluding the diagonal)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i != j and corr_matrix[i, j] < min_val:
                min_val = corr_matrix[i, j]
                row_index = i
                col_index = j

    return min_val, row_index, col_index


def index_update(source_index: List, row_index: int, col_index: int) -> List:
    """
    Update the source indices after clustering two sources.

    Args:
        source_index: List of source indices
        row_index: Row index of the minimum value
        col_index: Column index of the minimum value

    Returns:
        Updated list of source indices
    """
    # Make a copy to avoid modifying the original list
    source_index_copy = source_index.copy()

    # Check if the source at row_index is a single source or already a cluster
    if isinstance(source_index_copy[row_index], int):
        # Create a new cluster
        source_index_update = [source_index_copy[row_index], source_index_copy[col_index]]
    else:
        # Add to an existing cluster
        source_index_copy[row_index].append(source_index_copy[col_index])
        source_index_update = source_index_copy[row_index]

    # Remove the individual sources (in correct order to avoid index shifting)
    if row_index > col_index:
        source_index_copy.pop(row_index)
        source_index_copy.pop(col_index)
    else:
        source_index_copy.pop(col_index)
        source_index_copy.pop(row_index)

    # Add the new cluster
    source_index_copy.append(source_index_update)

    return source_index_copy


def update_matrix(source_data: np.ndarray, row_index: int, col_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the source data matrix and correlation matrix after clustering.

    Args:
        source_data: Source data matrix
        row_index: Row index of the minimum value
        col_index: Column index of the minimum value

    Returns:
        Tuple of (updated source data matrix, updated correlation matrix)
    """
    # Sum the source profiles to create a new cluster profile
    cluster_data = np.sum(source_data.take([row_index, col_index], axis=1), axis=1)
    cluster_data = np.reshape(cluster_data, (cluster_data.shape[0], 1))

    # Remove the original source profiles
    delete_data = np.delete(source_data, [row_index, col_index], axis=1)

    # Add the new cluster profile
    source_data_update = np.concatenate((delete_data, cluster_data), axis=1)

    # Recalculate the correlation matrix
    corr_matrix_update = jsd_correlation_matrix(source_data_update)
    np.fill_diagonal(corr_matrix_update, 1)  # Set diagonal to 1 to avoid self-clustering

    return source_data_update, corr_matrix_update


def data_cluster(sources: np.ndarray, corr_matrix: np.ndarray, jsd_value: float) -> Tuple[List, np.ndarray]:
    """
    Cluster sources based on Jensen-Shannon divergence.

    Args:
        sources: Source data matrix
        corr_matrix: Correlation matrix of Jensen-Shannon divergences
        jsd_value: Threshold for clustering

    Returns:
        Tuple of (source indices after clustering, updated source data matrix)
    """
    logger.info(f"Clustering sources with JSD threshold {jsd_value}")

    # Initialize source indices as a list of integers
    source_index = list(range(sources.shape[1]))

    # Set diagonal to 1 to avoid self-clustering
    np.fill_diagonal(corr_matrix, 1)

    # Iteratively cluster sources until no pair has JSD below threshold
    for i in range(1, sources.shape[1]):
        # Find the pair with minimum JSD
        min_val, row_index, col_index = min_value(corr_matrix)

        logger.debug(f"Minimum JSD: {min_val:.4f} between sources {row_index} and {col_index}")

        # Stop if minimum JSD is above threshold
        if min_val > jsd_value:
            logger.info(f"Stopping clustering: minimum JSD ({min_val:.4f}) > threshold ({jsd_value})")
            break

        # Update source indices
        source_index = index_update(source_index, row_index, col_index)

        # Update source data and correlation matrix
        sources, corr_matrix = update_matrix(sources, row_index, col_index)

        logger.debug(f"After clustering step {i}: {len(source_index)} sources remain")

    logger.info(f"Clustering complete: {len(source_index)} sources after clustering")

    return source_index, sources