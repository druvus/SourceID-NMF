"""
Core functionality for the SourceID-NMF algorithm with advanced optimizations.

This version includes:
1. Active-set method integration
2. Adaptive rho parameter
3. Enhanced sink processing
4. Automatic data characteristic detection
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import concurrent.futures

import numpy as np
import pandas as pd

from sourceid_nmf.admm import (
    admm_step,
    lagrangian,
    clear_inv_cache,
    run_admm_with_optimizations
)
from sourceid_nmf.clustering import data_cluster, jsd_correlation_matrix
from sourceid_nmf.estimation import jsd_estimation, source_jsd_estimation
from sourceid_nmf.initialization import (
    parameters_initialize,
    standardization,
    unknown_initialize,
)
from sourceid_nmf.utils import load_data, save_results

logger = logging.getLogger("sourceid_nmf.core")

def detect_data_characteristics(sources, sinks):
    """
    Analyze data to automatically select optimal algorithm parameters.

    Args:
        sources: Source data matrix
        sinks: Sink data matrix

    Returns:
        Dictionary of recommended parameters
    """
    recommendations = {}

    # Determine data size category
    total_elements = sources.size + sinks.size
    if total_elements < 10000:
        recommendations['size_category'] = 'small'
    elif total_elements < 100000:
        recommendations['size_category'] = 'medium'
    else:
        recommendations['size_category'] = 'large'

    # Determine sparsity
    sources_sparsity = np.count_nonzero(sources == 0) / sources.size
    sinks_sparsity = np.count_nonzero(sinks == 0) / sinks.size
    recommendations['sparsity'] = (sources_sparsity + sinks_sparsity) / 2

    # Recommend whether to use active-set method
    recommendations['use_active_set'] = recommendations['sparsity'] > 0.5

    # Recommend initial rho value
    if recommendations['sparsity'] > 0.7:
        recommendations['initial_rho'] = 0.5  # Lower rho for sparser data
    elif recommendations['sparsity'] > 0.4:
        recommendations['initial_rho'] = 1.0  # Medium rho for medium sparsity
    else:
        recommendations['initial_rho'] = 2.0  # Higher rho for dense data

    # Recommend thread allocation
    available_cpus = os.cpu_count() or 4
    num_sinks = sinks.shape[1]

    if num_sinks > 1 and available_cpus >= 4:
        sink_threads = min(num_sinks, max(2, available_cpus // 2))
        admm_threads = max(1, available_cpus // sink_threads)
    else:
        sink_threads = 1
        admm_threads = max(1, available_cpus - 1)

    recommendations['sink_threads'] = sink_threads
    recommendations['admm_threads'] = admm_threads

    logger.info(f"Data characteristics: size={recommendations['size_category']}, "
                f"sparsity={recommendations['sparsity']:.2f}")
    logger.info(f"Recommended parameters: use_active_set={recommendations['use_active_set']}, "
                f"initial_rho={recommendations['initial_rho']}")

    return recommendations

def process_sink_advanced(args):
    """
    Process a single sink with all sources using advanced optimizations.

    Args:
        args: Tuple containing all necessary parameters for processing one sink
            (sink_index, sources, sink, iteration, rho, weight_factor, threshold, thread, use_active_set)

    Returns:
        Tuple of (sink_index, estimated_proportions, jsd_wy_value, diff_xwh_value)
    """
    sink_index, sources, sink, iteration, rho, weight_factor, threshold, thread, use_active_set = args

    logger.info(f"Processing sink {sink_index} with advanced optimizations")

    # Remove non-contributing sources
    sources_sums = np.sum(sources, axis=0)
    zero_sum_columns = np.where(sources_sums == 0)[0]
    cleaned_sources = np.delete(sources, zero_sum_columns, axis=1) if len(zero_sum_columns) > 0 else sources

    # Remove taxa with zero abundance in both sources and sinks
    sink_reshaped = sink.reshape((-1, 1))
    merging_data = np.hstack((cleaned_sources, sink_reshaped))

    # More efficient filtering of zero rows
    non_zero_rows = np.logical_or(
        np.any(merging_data[:, :-1] > 0, axis=1),
        merging_data[:, -1] > 0
    )
    remove_data = merging_data[non_zero_rows, :]

    # Skip standardization if possible
    col_sums = np.sum(remove_data, axis=0)
    if np.any(col_sums == 0):
        remove_data = standardization(remove_data)
    elif not np.allclose(col_sums, 1.0, rtol=1e-05):
        remove_data = standardization(remove_data)

    # Split back into source and sink
    source = remove_data[:, :-1]
    sink_data = remove_data[:, -1].reshape((-1, 1))

    # Initialize parameters
    unknown_init = unknown_initialize(source, sink_data)
    x, y, w, a, identity, w_plus, h_plus, alpha_w, alpha_h = parameters_initialize(
        source, sink_data, unknown_init, weight_factor
    )

    # Run NMF model with advanced optimizations
    try:
        # Use the run function with optimizations
        w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h, loss = run_admm_with_optimizations(
            w, x, y, a, identity, w_plus, h_plus, alpha_w, alpha_h, rho, thread, iteration, threshold
        )
        logger.info(f"Advanced optimization completed for sink {sink_index} in {len(loss)} iterations")
    except Exception as e:
        logger.warning(f"Advanced optimization failed for sink {sink_index}, "
                      f"falling back to standard method: {str(e)}")
        # Fall back to standard ADMM for this sink
        w_renew = w.copy()
        h_renew = h_plus.copy()
        loss = []

        try:
            # Initial step with standard ADMM
            w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h = admm_step(
                w, x, y, a, identity, w_plus, h_plus, alpha_w, alpha_h, rho, thread
            )

            l_init = lagrangian(x, y, w_renew, h_renew, a, w_plus, h_plus, alpha_w, alpha_h, rho)
            loss.append(l_init)

            # Run iterations with early stopping
            adaptive_threshold = threshold
            stagnation_count = 0

            for m in range(1, iteration):
                try:
                    w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h = admm_step(
                        w_renew, x, y, a, identity, w_plus, h_plus, alpha_w, alpha_h, rho, thread
                    )

                    l_update = lagrangian(x, y, w_renew, h_renew, a, w_plus, h_plus, alpha_w, alpha_h, rho)
                    loss.append(l_update)

                    # Check convergence with adaptive threshold
                    rel_improvement = abs(l_update - loss[m - 1]) / max(abs(loss[m - 1]), 1e-10)

                    if rel_improvement < adaptive_threshold:
                        stagnation_count += 1
                        if stagnation_count >= 3:
                            logger.info(f"Converged after {m+1} iterations for sink {sink_index}")
                            break
                    else:
                        stagnation_count = 0

                        # Reduce threshold as we make more iterations
                        if m > 10 and m % 10 == 0:
                            adaptive_threshold = max(threshold * 0.1, threshold / (1 + m/100))
                except Exception as iter_e:
                    logger.warning(f"Error at iteration {m}: {iter_e}, stopping early")
                    break
        except Exception as init_e:
            logger.error(f"Standard ADMM also failed: {init_e}")
            # In case both methods fail, return initial values

    # Calculate performance metrics
    try:
        jsd_wy_value = jsd_estimation(w_plus[:, :-1], y[:, :-1])
        diff_xwh_value = np.sum(abs(x - np.dot(w_renew, h_renew)))
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        jsd_wy_value = float('nan')
        diff_xwh_value = float('nan')

    logger.info(f"Completed sink {sink_index}: JSD={jsd_wy_value:.6f}, diff={diff_xwh_value:.6f}, "
                f"iterations={len(loss)}")

    # Return sink index with results to maintain order
    return (sink_index, h_plus.reshape((h_plus.shape[0])), jsd_wy_value, diff_xwh_value)


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
    use_active_set: bool = True,
    adaptive_rho: bool = True,
) -> np.ndarray:
    """
    Run the SourceID-NMF source tracking algorithm with advanced optimizations.

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
        use_active_set: Whether to use active-set method for optimization
        adaptive_rho: Whether to use adaptive rho parameter

    Returns:
        Array of estimated proportions
    """
    logger.info("Starting SourceID-NMF source tracking with advanced optimizations")
    logger.info(f"Parameters: mode={mode}, threads={thread}, "
                f"iterations={iteration}, rho={rho}, weight_factor={weight_factor}, "
                f"threshold={threshold}, use_active_set={use_active_set}, "
                f"adaptive_rho={adaptive_rho}")

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

    # Analyze data and get recommended parameters
    recommendations = detect_data_characteristics(sources, sinks)

    # Use recommendations if not explicitly set
    if rho == 1:  # Default value
        rho = recommendations['initial_rho']
        logger.info(f"Using recommended initial rho: {rho}")

    # Determine optimal thread allocation
    num_sinks = sinks.shape[1]
    available_cpus = min(thread, os.cpu_count() or thread)

    # For single sink or few sinks, allocate all threads to ADMM
    # For many sinks, process multiple sinks in parallel
    sink_parallel = num_sinks > 1 and available_cpus >= 4

    if sink_parallel:
        # Use recommended thread allocation if available
        sink_threads = recommendations['sink_threads']
        admm_threads = recommendations['admm_threads']
        logger.info(f"Using {sink_threads} threads for sink processing, {admm_threads} threads for ADMM")

        # Prepare tasks for parallel processing
        tasks = []
        for i in range(num_sinks):
            tasks.append((i, sources, sinks[:, i], iteration, rho, weight_factor, threshold,
                          admm_threads, use_active_set))

        # Process sinks in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=sink_threads) as executor:
            future_to_sink = {executor.submit(process_sink_advanced, task): i for i, task in enumerate(tasks)}
            for future in concurrent.futures.as_completed(future_to_sink):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed sink {result[0]+1}/{num_sinks}")
                except Exception as exc:
                    sink_idx = future_to_sink[future]
                    logger.error(f"Sink {sink_idx} generated an exception: {exc}")
                    raise

        # Sort results by sink index to maintain order
        results.sort(key=lambda x: x[0])

        # Extract results
        estimated_proportions = np.array([r[1] for r in results])
        jsd_wy = [r[2] for r in results]
        diff_xwh = [r[3] for r in results]

    else:
        # Process sinks sequentially with all threads for ADMM
        logger.info(f"Processing {num_sinks} sinks sequentially with {thread} threads for ADMM")

        # Output arrays
        estimated_proportions = []
        jsd_wy = []
        diff_xwh = []

        # Process each sink
        for i in range(num_sinks):
            # Clear cache before processing each sink to free memory
            clear_inv_cache()

            # Process sink with advanced optimizations
            result = process_sink_advanced((i, sources, sinks[:, i], iteration, rho,
                                           weight_factor, threshold, thread, use_active_set))
            estimated_proportions.append(result[1])
            jsd_wy.append(result[2])
            diff_xwh.append(result[3])

        # Convert to numpy array
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