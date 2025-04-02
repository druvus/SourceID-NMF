"""
Robust ADMM implementation for NMF with extra error handling.

This version handles numerical stability issues, singular matrices,
and function signature consistency.
"""

import concurrent.futures
import logging
from typing import List, Tuple
import numpy as np
from numpy.linalg import inv, LinAlgError
from scipy.optimize import root_scalar
from tqdm import tqdm

logger = logging.getLogger("sourceid_nmf.admm")

# Cache matrix inversions where possible
_inv_cache = {}

def clear_inv_cache():
    """Clear the inversion cache"""
    global _inv_cache
    _inv_cache = {}

def safe_inv(matrix, epsilon=1e-10):
    """
    Safely computes matrix inverse with regularization for singular matrices.

    Args:
        matrix: Input matrix to invert
        epsilon: Small value for regularization

    Returns:
        Inverted matrix
    """
    try:
        # Try normal inversion first
        return inv(matrix)
    except LinAlgError:
        # Add regularization for singular/near-singular matrix
        logger.warning(f"Matrix is singular, applying regularization (epsilon={epsilon})")
        reg_matrix = matrix + epsilon * np.eye(matrix.shape[0])
        return inv(reg_matrix)

def w_update(w_plus, alpha_w, h, x, i, rho):
    """
    Update W in the ADMM algorithm.

    Args:
        w_plus: Current W+ matrix
        alpha_w: Lagrangian multiplier for W
        h: Current H matrix
        x: Data matrix X
        i: Identity matrix
        rho: Penalty parameter

    Returns:
        Updated W matrix
    """
    # Calculate h.dot(h.T) + rho * i
    h_ht = np.dot(h, h.T)
    right_term = h_ht + rho * i

    # Use safe inversion
    try:
        right_inv = safe_inv(right_term)

        # Calculate x.dot(h.T) + rho * w_plus - alpha_w
        x_ht = np.dot(x, h.T)
        left_term = x_ht + rho * w_plus - alpha_w

        # Final multiplication
        return np.dot(left_term, right_inv)
    except Exception as e:
        # Fallback to simpler update if something fails
        logger.warning(f"Error in w_update: {e}, using fallback")
        return w_plus.copy()  # Return previous value as fallback

def h_update(w_renew, x, i, h_plus, alpha_h, rho):
    """
    Update H in the ADMM algorithm.

    Args:
        w_renew: Updated W matrix
        x: Data matrix X
        i: Identity matrix
        h_plus: Current H+ matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Updated H matrix
    """
    try:
        # Calculate w_renew.T.dot(w_renew) + rho * i
        wt_w = np.dot(w_renew.T, w_renew)
        left_term = wt_w + rho * i

        # Use safe inversion
        left_inv = safe_inv(left_term)

        # Calculate right term
        wt_x = np.dot(w_renew.T, x)
        right_term = wt_x + rho * h_plus - alpha_h

        # Final multiplication
        return np.dot(left_inv, right_term)
    except Exception as e:
        # Fallback to simpler update if something fails
        logger.warning(f"Error in h_update: {e}, using fallback")
        return h_plus.copy()  # Return previous value as fallback

def w_add(w_renew, d):
    """
    Apply normalization to W with robust error handling.

    Args:
        w_renew: Updated W matrix
        d: Diagonal normalization matrix

    Returns:
        Normalized W matrix
    """
    try:
        # Check for zero diagonal elements
        diag_elements = np.diag(d)
        if np.any(np.abs(diag_elements) < 1e-10):
            # Replace small values with 1.0 to avoid division by zero
            logger.warning("Near-zero values in normalization matrix, applying fix")
            for i in range(len(diag_elements)):
                if np.abs(diag_elements[i]) < 1e-10:
                    d[i, i] = 1.0

        # Compute inverse safely
        d_inv = safe_inv(d)
        return np.dot(w_renew, d_inv)
    except Exception as e:
        # Fallback if something fails
        logger.warning(f"Error in w_add normalization: {e}, using original matrix")
        return w_renew  # Return original matrix as fallback

def h_add(h_renew, d):
    """
    Apply normalization to H with robust error handling.

    Args:
        h_renew: Updated H matrix
        d: Diagonal normalization matrix

    Returns:
        Normalized H matrix
    """
    try:
        return np.dot(d, h_renew)
    except Exception as e:
        # Fallback if something fails
        logger.warning(f"Error in h_add normalization: {e}, using original matrix")
        return h_renew  # Return original matrix as fallback

def water_filling_w(beta_w, w, alpha_w, y, a, rho):
    """
    Water-filling function for W+ update with robust error handling.

    Args:
        beta_w: Beta parameter
        w: Current W matrix
        alpha_w: Lagrangian multiplier for W
        y: Source data matrix
        a: Weighting matrix
        rho: Penalty parameter

    Returns:
        Sum of maximum values minus 1
    """
    try:
        # Ensure rho is not zero to avoid division by zero
        safe_rho = max(rho, 1e-10)

        a_squared = np.square(a)
        w1 = a_squared * y + alpha_w + safe_rho * w - beta_w
        w2 = a_squared + safe_rho

        # Ensure division is safe
        safe_division = np.zeros_like(w1)
        valid_indices = w2 > 1e-10
        safe_division[valid_indices] = w1[valid_indices] / w2[valid_indices]

        return np.sum(np.maximum(0, safe_division)) - 1
    except Exception as e:
        # Fallback if something fails
        logger.warning(f"Error in water_filling_w: {e}")
        return 0.0  # Return neutral value

def water_filling_h(beta_h, h, alpha_h, rho):
    """
    Water-filling function for H+ update with robust error handling.

    Args:
        beta_h: Beta parameter
        h: Current H matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Sum of maximum values minus 1
    """
    try:
        # Ensure rho is not zero to avoid division by zero
        safe_rho = max(rho, 1e-10)

        # Calculate safely
        safe_term = np.zeros_like(h)
        safe_term = (alpha_h + safe_rho * h - beta_h) / safe_rho

        return np.sum(np.maximum(0, safe_term)) - 1
    except Exception as e:
        # Fallback if something fails
        logger.warning(f"Error in water_filling_h: {e}")
        return 0.0  # Return neutral value

def robust_root_scalar_w(w, alpha_w, y, a, rho):
    """
    Find the root of the water-filling function for W with robust error handling.

    Args:
        w: Current W matrix
        alpha_w: Lagrangian multiplier for W
        y: Source data matrix
        a: Weighting matrix
        rho: Penalty parameter

    Returns:
        Root value
    """
    try:
        # Compute reasonable brackets based on data
        values = alpha_w + rho * w
        max_val = np.max(values) + 10
        min_val = np.min(values) - 10

        # Ensure valid bracket (min < max)
        if min_val >= max_val:
            min_val = max_val - 20

        # Check if function values at bracket ends have opposite signs
        f_min = water_filling_w(min_val, w, alpha_w, y, a, rho)
        f_max = water_filling_w(max_val, w, alpha_w, y, a, rho)

        if f_min * f_max > 0:
            # No bracket, try expanded range
            expanded_min = min_val - 100
            expanded_max = max_val + 100
            f_expanded_min = water_filling_w(expanded_min, w, alpha_w, y, a, rho)
            f_expanded_max = water_filling_w(expanded_max, w, alpha_w, y, a, rho)

            if f_expanded_min * f_expanded_max <= 0:
                # Found bracket with expanded range
                min_val, max_val = expanded_min, expanded_max
            else:
                # Still no bracket, use bisection with artificial bracket
                logger.warning("Could not find root bracket, using heuristic value")
                # Return a heuristic value based on the data
                avg_val = np.mean(values)
                if f_min < 0:
                    return min_val  # Return lower bound
                else:
                    return max_val  # Return upper bound

        # Try to find root
        try:
            sol = root_scalar(water_filling_w, args=(w, alpha_w, y, a, rho),
                            method='brentq', bracket=[min_val, max_val])
            return sol.root
        except Exception as e:
            # If root finding fails, use bisection
            logger.warning(f"Root finding failed: {e}, using bisection")
            return bisection_root(water_filling_w, min_val, max_val,
                                args=(w, alpha_w, y, a, rho))
    except Exception as e:
        # Global error handler
        logger.warning(f"Error in root_scalar_w: {e}, returning average value")
        # Return a reasonable default value
        return np.mean(alpha_w + rho * w)

def bisection_root(func, a, b, args=(), max_iter=100, tol=1e-6):
    """
    Simple bisection method for root finding with robust error handling.

    Args:
        func: Function to find root of
        a: Lower bracket
        b: Upper bracket
        args: Additional arguments to func
        max_iter: Maximum iterations
        tol: Tolerance for convergence

    Returns:
        Estimated root
    """
    try:
        fa = func(a, *args)
        fb = func(b, *args)

        # Check if we have a bracket
        if fa * fb > 0:
            # No bracket - return something reasonable
            if np.abs(fa) < np.abs(fb):
                return a
            else:
                return b

        # Iterate up to max_iter times
        for i in range(max_iter):
            c = (a + b) / 2
            try:
                fc = func(c, *args)

                if abs(fc) < tol or (b - a) < tol:
                    return c

                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            except Exception as e:
                # Handle potential errors during function evaluation
                logger.warning(f"Error during bisection iteration: {e}")
                # Return the current midpoint as a best guess
                return c

        # Return best estimate after max iterations
        return (a + b) / 2
    except Exception as e:
        # Global error handler
        logger.warning(f"Bisection root finding failed: {e}")
        # Return the midpoint of the original interval
        return (a + b) / 2

def robust_root_scalar_h(h, alpha_h, rho):
    """
    Find the root of the water-filling function for H with robust error handling.

    Args:
        h: Current H matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Root value
    """
    try:
        # Compute reasonable brackets based on data
        values = alpha_h + rho * h
        max_val = np.max(values) + 10
        min_val = np.min(values) - 10

        # Ensure min < max
        if min_val >= max_val:
            min_val = max_val - 20

        # Check if function values at bracket ends have opposite signs
        f_min = water_filling_h(min_val, h, alpha_h, rho)
        f_max = water_filling_h(max_val, h, alpha_h, rho)

        if f_min * f_max > 0:
            # No bracket, try expanded range
            expanded_min = min_val - 100
            expanded_max = max_val + 100
            f_expanded_min = water_filling_h(expanded_min, h, alpha_h, rho)
            f_expanded_max = water_filling_h(expanded_max, h, alpha_h, rho)

            if f_expanded_min * f_expanded_max <= 0:
                # Found bracket with expanded range
                min_val, max_val = expanded_min, expanded_max
            else:
                # Still no bracket, use heuristic value
                logger.warning("Could not find root bracket for H, using heuristic value")
                avg_val = np.mean(values)
                if f_min < 0:
                    return min_val  # Return lower bound
                else:
                    return max_val  # Return upper bound

        # Try to find root
        try:
            sol = root_scalar(water_filling_h, args=(h, alpha_h, rho),
                            method='brentq', bracket=[min_val, max_val])
            return sol.root
        except Exception as e:
            # If root finding fails, use bisection
            logger.warning(f"H root finding failed: {e}, using bisection")
            return bisection_root(water_filling_h, min_val, max_val,
                                args=(h, alpha_h, rho))
    except Exception as e:
        # Global error handler
        logger.warning(f"Error in root_scalar_h: {e}, returning average value")
        # Return a reasonable default value
        return np.mean(alpha_h + rho * h)

def call_beta(args):
    """Helper function for parallel processing of beta calculation"""
    i, w_latest, alpha_w, y, a, rho = args
    return i, robust_root_scalar_w(w_latest[:, i], alpha_w[:, i], y[:, i], a[:, i], rho)

def batch_call_beta(batch):
    """Process a batch of columns for beta calculation"""
    results = []
    for args in batch:
        i, w_latest, alpha_w, y, a, rho = args
        results.append((i, robust_root_scalar_w(w_latest[:, i], alpha_w[:, i], y[:, i], a[:, i], rho)))
    return results

def w_plus_update(a, y, w_latest, alpha_w, rho, thread):
    """
    Update W+ in the ADMM algorithm with robust parallel processing.

    Args:
        a: Weighting matrix
        y: Source data matrix
        w_latest: Latest W matrix
        alpha_w: Lagrangian multiplier for W
        rho: Penalty parameter
        thread: Number of threads for parallel processing

    Returns:
        Updated W+ matrix
    """
    try:
        n_cols = w_latest.shape[1]

        # For small matrices or few threads, use sequential processing
        if n_cols <= 4 or thread <= 1:
            beta_results = []
            for i in range(n_cols):
                try:
                    beta = robust_root_scalar_w(w_latest[:, i], alpha_w[:, i], y[:, i], a[:, i], rho)
                    beta_results.append((i, beta))
                except Exception as e:
                    logger.warning(f"Error calculating beta for column {i}: {e}")
                    # Use a fallback value
                    avg_val = np.mean(alpha_w[:, i] + rho * w_latest[:, i])
                    beta_results.append((i, avg_val))
        else:
            # Parallel processing with batching
            try:
                # Create task list
                query_list = [(i, w_latest, alpha_w, y, a, rho) for i in range(n_cols)]

                # Determine batch size
                batch_size = max(1, n_cols // (thread * 2))
                batches = [query_list[i:i+batch_size] for i in range(0, n_cols, batch_size)]

                results = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=thread) as executor:
                    batch_results = list(executor.map(batch_call_beta, batches))

                    # Flatten the results
                    for batch_result in batch_results:
                        results.extend(batch_result)

                # Sort by column index
                results.sort(key=lambda x: x[0])
                beta_results = results
            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
                # Fall back to sequential processing
                beta_results = []
                for i in range(n_cols):
                    try:
                        beta = robust_root_scalar_w(w_latest[:, i], alpha_w[:, i], y[:, i], a[:, i], rho)
                        beta_results.append((i, beta))
                    except Exception as e:
                        # Use a fallback value for this column
                        avg_val = np.mean(alpha_w[:, i] + rho * w_latest[:, i])
                        beta_results.append((i, avg_val))

        # Extract beta values
        beta = np.zeros(n_cols)
        for i, val in beta_results:
            beta[i] = val

        # Calculate W+ safely
        a_squared = np.square(a)
        w1 = a_squared * y + alpha_w + rho * w_latest
        w1 = w1 - beta.reshape(1, -1)  # Reshape beta for broadcasting
        w2 = a_squared + rho

        # Safe division
        w_plus = np.zeros_like(w1)
        valid_indices = w2 > 1e-10
        w_plus[valid_indices] = w1[valid_indices] / w2[valid_indices]
        w_plus = np.maximum(0, w_plus)

        return w_plus
    except Exception as e:
        # Global error handler
        logger.warning(f"Error in w_plus_update: {e}, returning previous value")
        # Return the previous value as fallback
        return w_latest.copy()

def h_plus_update(h_latest, alpha_h, rho):
    """
    Update H+ in the ADMM algorithm with robust error handling.

    Args:
        h_latest: Latest H matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Updated H+ matrix
    """
    try:
        # Find root safely
        root = robust_root_scalar_h(h_latest, alpha_h, rho)

        # Ensure rho is not zero
        safe_rho = max(rho, 1e-10)

        # Safe division
        h_term = alpha_h + rho * h_latest - root
        h_plus = np.maximum(0, h_term / safe_rho)

        return h_plus
    except Exception as e:
        # Global error handler
        logger.warning(f"Error in h_plus_update: {e}, returning previous value")
        # Return the previous value as fallback
        return h_latest.copy()

def alpha_w_update(alpha_w, w_latest, w_plus, rho):
    """Update Lagrangian multiplier for W"""
    return alpha_w + rho * (w_latest - w_plus)

def alpha_h_update(alpha_h, h_latest, h_plus, rho):
    """Update Lagrangian multiplier for H"""
    return alpha_h + rho * (h_latest - h_plus)

def calculate_residuals(w, w_plus, h, h_plus):
    """
    Calculate primal and dual residuals safely.

    Args:
        w: Current W matrix
        w_plus: Current W+ matrix
        h: Current H matrix
        h_plus: Current H+ matrix

    Returns:
        Tuple of (primal_residual_norm, dual_residual_norm)
    """
    try:
        # Primal residual: r = ||w - w_plus||^2 + ||h - h_plus||^2
        primal_residual_norm = np.linalg.norm(w - w_plus)**2 + np.linalg.norm(h - h_plus)**2

        # Dual residual: s = ||w_plus - w_plus_prev||^2 + ||h_plus - h_plus_prev||^2
        # Note: We don't have previous values, so we use a simpler estimate
        dual_residual_norm = np.linalg.norm(w - w_plus)**2 + np.linalg.norm(h - h_plus)**2

        return primal_residual_norm, dual_residual_norm
    except Exception as e:
        # Handle any errors during calculation
        logger.warning(f"Error calculating residuals: {e}")
        return 1.0, 1.0  # Return balanced values as fallback

def adjust_rho(rho, w, w_plus, h, h_plus, primal_residual_norm, dual_residual_norm):
    """
    Adjust the penalty parameter rho based on primal and dual residuals.

    Args:
        rho: Current rho value
        w: Current W matrix
        w_plus: Current W+ matrix
        h: Current H matrix
        h_plus: Current H+ matrix
        primal_residual_norm: Norm of primal residual
        dual_residual_norm: Norm of dual residual

    Returns:
        Updated rho value
    """
    try:
        # Ensure both residuals are valid numbers
        if not np.isfinite(primal_residual_norm) or not np.isfinite(dual_residual_norm):
            return rho  # Keep current rho if residuals are invalid

        # Calculate primal and dual residuals
        if primal_residual_norm > 10 * dual_residual_norm:
            # Primal residual is too large, increase rho
            return min(rho * 2, 1e6)  # Upper bound to prevent numerical issues
        elif dual_residual_norm > 10 * primal_residual_norm:
            # Dual residual is too large, decrease rho
            return max(rho / 2, 1e-6)  # Lower bound to prevent numerical issues
        else:
            # Residuals are balanced, keep rho the same
            return rho
    except Exception as e:
        # Handle any errors during adjustment
        logger.warning(f"Error adjusting rho: {e}")
        return rho  # Keep current rho

def admm_step(w, x, y, a, i, w_plus, h_plus, alpha_w, alpha_h, rho, thread,
              adaptive_rho=True, use_active_set=False):
    """
    Perform one ADMM iteration step with robust error handling.

    Args:
        w: Current W matrix
        x: Data matrix X
        y: Source data matrix Y
        a: Weighting matrix
        i: Identity matrix
        w_plus: Current W+ matrix
        h_plus: Current H+ matrix
        alpha_w: Lagrangian multiplier for W
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter
        thread: Number of threads for parallel processing
        adaptive_rho: Whether to use adaptive rho adjustment
        use_active_set: Whether to use active-set method (ignored in this version)

    Returns:
        Tuple of (w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h)
    """
    try:
        # Update H
        h_renew = h_update(w, x, i, h_plus, alpha_h, rho)

        # Update W
        w_renew = w_update(w_plus, alpha_w, h_renew, x, i, rho)

        # Normalize W and H safely
        try:
            d = np.diag(np.sum(w_renew, axis=0))
            h_renew = h_add(h_renew, d)
            w_renew = w_add(w_renew, d)
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, using unnormalized matrices")
            # Continue with unnormalized matrices

        # Update auxiliary variables H+ and W+
        h_plus = h_plus_update(h_renew, alpha_h, rho)
        w_plus = w_plus_update(a, y, w_renew, alpha_w, rho, thread)

        # Update Lagrangian multipliers
        alpha_w = alpha_w_update(alpha_w, w_renew, w_plus, rho)
        alpha_h = alpha_h_update(alpha_h, h_renew, h_plus, rho)

        # Return the updated values (do NOT include rho to maintain consistent interface)
        return w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h
    except Exception as e:
        # Global error handler for the entire step
        logger.error(f"ADMM step failed: {e}")
        # Return the input values as fallback
        return w, w, w_plus, h_plus, alpha_w, alpha_h

def lagrangian(x, y, w, h, a, w_plus, h_plus, alpha_w, alpha_h, rho):
    """Calculate the Lagrangian function value safely"""
    try:
        # Data fidelity term
        x_wh = x - np.dot(w, h)
        l1 = 0.5 * np.sum(x_wh * x_wh)

        # Source fidelity term
        diff = w_plus - y
        l2 = 0.5 * np.sum(a * a * diff * diff)

        # Augmented Lagrangian terms for W
        w_diff = w - w_plus
        l3 = np.sum(alpha_w * w_diff)
        l4 = (rho / 2) * np.sum(w_diff * w_diff)

        # Augmented Lagrangian terms for H
        h_diff = h - h_plus
        l5 = np.sum(alpha_h * h_diff)
        l6 = (rho / 2) * np.sum(h_diff * h_diff)

        return l1 + l2 + l3 + l4 + l5 + l6
    except Exception as e:
        # Handle any errors during calculation
        logger.warning(f"Error calculating Lagrangian: {e}")
        return float('inf')  # Return a large value to indicate poor solution

def run_admm_with_optimizations(w, x, y, a, i, w_plus, h_plus, alpha_w, alpha_h,
                               initial_rho, thread, iteration, threshold):
    """
    Run the ADMM algorithm with robust error handling.

    Args:
        w: Initial W matrix
        x: Data matrix X
        y: Source data matrix Y
        a: Weighting matrix
        i: Identity matrix
        w_plus: Initial W+ matrix
        h_plus: Initial H+ matrix
        alpha_w: Initial Lagrangian multiplier for W
        alpha_h: Initial Lagrangian multiplier for H
        initial_rho: Initial penalty parameter
        thread: Number of threads for parallel processing
        iteration: Maximum number of iterations
        threshold: Convergence threshold

    Returns:
        Tuple of (w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h, loss_history)
    """
    # Initialize variables
    rho = max(initial_rho, 1e-6)  # Ensure positive rho
    loss = []
    w_renew, h_renew = w.copy(), h_plus.copy()

    try:
        # Initial step
        w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h = admm_step(
            w, x, y, a, i, w_plus, h_plus, alpha_w, alpha_h, rho, thread
        )

        # Calculate initial loss
        l_init = lagrangian(x, y, w_renew, h_renew, a, w_plus, h_plus, alpha_w, alpha_h, rho)
        loss.append(l_init)

        # Use adaptive threshold
        adaptive_threshold = threshold
        stagnation_count = 0

        # Main iteration loop
        for m in range(1, iteration):
            # Consider rho adaptation between steps
            if m > 1 and m % 5 == 0:  # Every 5 iterations
                # Calculate residuals
                primal_res, dual_res = calculate_residuals(w_renew, w_plus, h_renew, h_plus)
                # Adjust rho
                rho = adjust_rho(rho, w_renew, w_plus, h_renew, h_plus, primal_res, dual_res)

            # Perform one ADMM iteration
            w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h = admm_step(
                w_renew, x, y, a, i, w_plus, h_plus, alpha_w, alpha_h, rho, thread
            )

            # Calculate updated loss
            l_update = lagrangian(x, y, w_renew, h_renew, a, w_plus, h_plus, alpha_w, alpha_h, rho)
            loss.append(l_update)

            # Check for NaN or infinite loss
            if not np.isfinite(l_update):
                logger.warning(f"Non-finite loss at iteration {m}, stopping early")
                break

            # Calculate relative improvement
            rel_improvement = abs(l_update - loss[m - 1]) / max(abs(loss[m - 1]), 1e-10)

            # Early stopping with adaptive thresholding
            if rel_improvement < adaptive_threshold:
                stagnation_count += 1
                if stagnation_count >= 3:  # Stop if stagnating for 3 iterations
                    logger.info(f"Converged after {m+1} iterations (stagnation)")
                    break
            else:
                # Reset stagnation counter if we make progress
                stagnation_count = 0

                # Adaptively reduce threshold
                if m > 10 and m % 10 == 0:
                    adaptive_threshold = max(threshold * 0.1, threshold / (1 + m/100))
    except Exception as e:
        logger.error(f"ADMM optimization failed: {e}")
        # Return the best solution found so far

    return w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h, loss