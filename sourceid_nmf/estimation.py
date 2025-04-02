"""
Performance estimation for SourceID-NMF results.

This module provides functions for evaluating the performance of the NMF source tracking
algorithm using Jensen-Shannon divergence and other metrics.
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger("sourceid_nmf.estimation")


def source_jsd_estimation(source_data: np.ndarray) -> float:
    """
    Calculate the average Jensen-Shannon divergence between all pairs of sources.

    Args:
        source_data: Source data matrix of shape (n_taxa, n_sources)

    Returns:
        Average Jensen-Shannon divergence
    """
    logger.debug("Calculating average JSD between sources")

    jsd_values = []

    # Calculate pairwise Jensen-Shannon divergences
    for i in range(source_data.shape[1] - 1):
        for j in range(i + 1, source_data.shape[1]):
            jsd = jensenshannon(source_data[:, i], source_data[:, j])
            jsd_values.append(jsd)

    # Calculate mean JSD
    jensen_shannon_mean = np.mean(jsd_values)
    logger.info(f"Average JSD between sources: {jensen_shannon_mean:.6f}")

    return jensen_shannon_mean


def jsd_estimation(noise_data: np.ndarray, source_data: np.ndarray) -> float:
    """
    Calculate the average Jensen-Shannon divergence between noise data and source data.

    Args:
        noise_data: Noise data matrix
        source_data: Source data matrix

    Returns:
        Average Jensen-Shannon divergence
    """
    jensen_shannon = []

    # Calculate Jensen-Shannon divergence for each pair of columns
    for i in range(noise_data.shape[1]):
        noise_profile = noise_data[:, i]
        source_profile = source_data[:, i]
        jsd = jensenshannon(noise_profile, source_profile)
        jensen_shannon.append(jsd)

    # Calculate mean JSD
    jensen_shannon_mean = np.mean(jensen_shannon)
    logger.debug(f"Average JSD between noise and source data: {jensen_shannon_mean:.6f}")

    return jensen_shannon_mean


def evaluate_performance(
    estimated_proportions_path: str,
    true_proportions_path: str
) -> Tuple[float, List[float], float, List[float]]:
    """
    Evaluate the performance of the estimated proportions against true proportions.

    Args:
        estimated_proportions_path: Path to the file with estimated proportions
        true_proportions_path: Path to the file with true proportions

    Returns:
        Tuple of (average JSD, JSD per sink, average difference, difference per sink)
    """
    logger.info(f"Evaluating performance: {estimated_proportions_path} vs {true_proportions_path}")

    try:
        # Load estimated and true proportions
        estimated_proportions = pd.read_csv(estimated_proportions_path, sep=" ", header=0, index_col=0)
        true_proportions = pd.read_csv(true_proportions_path, sep="\t", header=0, index_col=0)

        # Extract row labels (sink samples)
        row_labels = list(estimated_proportions.index.values)

        # Convert to numpy arrays for calculations
        estimated_proportions = np.array(estimated_proportions)
        true_proportions = np.array(true_proportions)

        # Calculate performance metrics
        jsd_ave, jsd, diff_ave, diff = perf_admm(estimated_proportions, true_proportions)

        logger.info(f"Average JSD between estimated and true proportions: {jsd_ave:.6f}")
        logger.info(f"Average absolute difference between estimated and true proportions: {diff_ave:.6f}")

        # Create a dataframe with the performance metrics
        perf_df = pd.DataFrame({
            'jsd': jsd,
            'difference': diff
        }, index=row_labels)

        return jsd_ave, jsd, diff_ave, diff

    except Exception as e:
        logger.error(f"Error during performance evaluation: {e}")
        raise


def jensen_shannon_divergence(
    estimated_proportions: np.ndarray,
    true_proportions: np.ndarray
) -> Tuple[float, List[float]]:
    """
    Calculate the Jensen-Shannon divergence between estimated and true proportions.

    Args:
        estimated_proportions: Matrix of estimated proportions
        true_proportions: Matrix of true proportions

    Returns:
        Tuple of (average JSD, JSD per sink)
    """
    jensen_shannon = []

    # Calculate Jensen-Shannon divergence for each sink
    for i in range(estimated_proportions.shape[0]):
        estimated_profile = estimated_proportions[i, :]
        true_profile = true_proportions[i, :]
        jsd = jensenshannon(estimated_profile, true_profile)
        jensen_shannon.append(jsd)

    # Calculate mean JSD
    jensen_shannon_mean = np.mean(jensen_shannon)

    return jensen_shannon_mean, jensen_shannon


def perf_admm(estimated_proportions: np.ndarray, true_proportions: np.ndarray) -> Tuple[float, List[float], float, List[float]]:
    """
    Calculate performance metrics for ADMM results.

    Args:
        estimated_proportions: Matrix of estimated proportions
        true_proportions: Matrix of true proportions

    Returns:
        Tuple of (average JSD, JSD per sink, average difference, difference per sink)
    """
    # Calculate Jensen-Shannon divergence
    jensen_shannon_mean, jensen_shannon = jensen_shannon_divergence(
        estimated_proportions, true_proportions
    )

    # Calculate absolute difference
    difference = np.sum(abs(estimated_proportions - true_proportions), axis=1)
    diff_mean = np.mean(difference)

    return jensen_shannon_mean, jensen_shannon, diff_mean, difference


def save_performance_results(
    jsd: List[float],
    diff: List[float],
    row_labels: List[str],
    output_path: str
) -> None:
    """
    Save performance results to a file.

    Args:
        jsd: Jensen-Shannon divergence values per sink
        diff: Absolute difference values per sink
        row_labels: Sink labels
        output_path: Path to save the performance file
    """
    logger.info(f"Saving performance results to {output_path}")

    try:
        # Create performance dataframe
        perf = np.array([jsd, diff])
        nmf_perf_label = ['jsd', 'difference']
        perf_df = pd.DataFrame(perf, index=nmf_perf_label, columns=row_labels)

        # Save to file
        perf_df.to_csv(output_path, sep=' ')
        logger.debug(f"Performance results saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Error saving performance results: {e}")
        raise