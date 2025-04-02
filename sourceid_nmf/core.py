"""
Core functionality for the SourceID-NMF algorithm.

This module contains the main functions for running the NMF-based source tracking algorithm.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sourceid_nmf.admm import admm_step, lagrangian
from sourceid_nmf.clustering import data_cluster, jsd_correlation_matrix
from sourceid_nmf.estimation import jsd_estimation, source_jsd_estimation
from sourceid_nmf.initialization import (
    parameters_initialize,
    standardization,
    unknown_initialize,
)
from sourceid_nmf.utils import load_data, save_results

logger = logging.getLogger("sourceid_nmf.core")


def run_source_tracking(
    data_path: str,
    name_path: str,
    output_path: str,
    mode: str = "normal",
    cutoff: float = 0.25,
    thread: int = 20,
    iteration: int = 2000,
    rho: int = 1,
    weight_factor: int = 1,
    threshold: float = 1e-06,
    perf_output: Optional[str] = None,
) -> np.ndarray:
    """
    Run the SourceID-NMF source tracking algorithm.

    Args:
        data_path: Path to input count table
        name_path: Path to sample name file
        output_path: Path to save the estimated proportions
        mode: Mode of operation ('normal' or 'cluster')
        cutoff: Threshold for clustering (if mode='cluster')
        thread: Number of threads for parallel processing
        iteration: Maximum number of iterations for NMF
        rho: Penalty parameter for ADMM
        weight_factor: Weighting matrix factor
        threshold: Convergence threshold
        perf_output: Optional path to save performance metrics

    Returns:
        Array of estimated proportions
    """
    logger.info("Starting SourceID-NMF source tracking")
    logger.info(f"Parameters: mode={mode}, threads={thread}, "
                f"iterations={iteration}, rho={rho}, weight_factor={weight_factor}, "
                f"threshold={threshold}")

    # Load data
    sources, sinks, sources_label, sinks_label = load_data(data_path, name_path)

    # Calculate average JSD between sources
    sources_jsd = source_jsd_estimation(sources)
    logger.info(f"Average Jensen-Shannon Divergence between sources: {sources_jsd:.6f}")

    # Cluster sources if requested
    if mode == 'cluster':
        logger.info(f"Clustering sources with cutoff={cutoff}")
        corr_matrix = jsd_correlation_matrix(sources)
        source_index, sources = data_cluster(sources, corr_matrix, cutoff)
        logger.info(f"Clustered sources: {source_index}")
        logger.info(f"Number of sources after clustering: {sources.shape[1]}")

        # Update source labels
        sources_label = [f"D{i+1}" for i in range(sources.shape[1])]
        sources_label.append("unknown")

    # Output arrays
    estimated_proportions = []  # Final estimated proportions
    jsd_wy = []  # JSD between W+ and Y
    diff_xwh = []  # Difference between X and W*H

    # Process each sink
    for i in range(sinks.shape[1]):
        sink_name = sinks_label[i]
        logger.info(f"Processing sink {i+1}/{sinks.shape[1]}: {sink_name}")

        # Remove non-contributing sources
        sources_sums = np.sum(sources, axis=0)
        zero_sum_columns = np.where(sources_sums == 0)[0]
        cleaned_sources = np.delete(sources, zero_sum_columns, axis=1)

        # Remove taxa with zero abundance in both sources and sinks
        merging_data = np.hstack((cleaned_sources, sinks[:, i].reshape((-1, 1))))
        remove_data = merging_data[[not np.all(merging_data[:, 0:merging_data.shape[1]][i] == 0)
                                     for i in range(merging_data.shape[0])], :]
        remove_data = standardization(remove_data)

        # Split back into source and sink
        source = remove_data[:, :-1]
        sink = remove_data[:, -1].reshape((-1, 1))

        # Initialize parameters
        logger.debug(f"Initializing NMF parameters for sink {i+1}")
        unknown_init = unknown_initialize(source, sink)
        x, y, w, a, identity, w_plus, h_plus, alpha_w, alpha_h = parameters_initialize(
            source, sink, unknown_init, weight_factor
        )

        # Run NMF model
        logger.info(f"Running NMF model for sink {i+1} (max {iteration} iterations)")
        loss = []

        # First iteration
        w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h = admm_step(
            w, x, y, a, identity, w_plus, h_plus, alpha_w, alpha_h, rho, thread
        )

        l_init = lagrangian(x, y, w_renew, h_renew, a, w_plus, h_plus, alpha_w, alpha_h, rho)
        loss.append(l_init)

        # Subsequent iterations
        for m in range(1, iteration):
            w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h = admm_step(
                w_renew, x, y, a, identity, w_plus, h_plus, alpha_w, alpha_h, rho, thread
            )

            l_update = lagrangian(x, y, w_renew, h_renew, a, w_plus, h_plus, alpha_w, alpha_h, rho)
            loss.append(l_update)

            # Check convergence
            if m == 1 and abs(l_update - l_init) / l_init < threshold:
                logger.info(f"Early convergence after 2 iterations for sink {i+1}")
                break

            if m > 1 and abs(l_update - loss[m - 1]) / loss[m - 1] < threshold:
                logger.info(f"Converged after {m+1} iterations for sink {i+1}")
                break

        # Calculate performance metrics
        jsd_wy_iter = jsd_estimation(w_plus[:, :-1], y[:, :-1])
        jsd_wy.append(jsd_wy_iter)

        diff_xwh_iter = np.sum(abs(x - np.dot(w_renew, h_renew)))
        diff_xwh.append(diff_xwh_iter)

        # Store the estimated proportions
        estimated_proportions.append(h_plus.reshape((h_plus.shape[0])))

        logger.info(f"Completed sink {i+1}: JSD={jsd_wy_iter:.6f}, diff={diff_xwh_iter:.6f}")

    # Convert to numpy array and create DataFrame
    estimated_proportions = np.array(estimated_proportions)

    # Save results
    save_results(estimated_proportions, sinks_label, sources_label, output_path)

    # Save performance metrics if requested
    if perf_output:
        logger.info(f"Saving performance metrics to {perf_output}")
        nmf_perf = [jsd_wy, diff_xwh]
        nmf_perf_label = ['jsd_wy', 'diff_xwh']
        nmf_perf = np.array(nmf_perf).T
        nmf_perf_df = pd.DataFrame(nmf_perf, index=sinks_label, columns=nmf_perf_label)
        nmf_perf_df.to_csv(perf_output, sep=' ')

    logger.info("SourceID-NMF source tracking completed successfully")
    return estimated_proportions