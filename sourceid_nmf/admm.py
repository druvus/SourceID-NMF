"""
ADMM (Alternating Direction Method of Multipliers) implementation for NMF.

This module contains the ADMM algorithm used for solving the non-negative matrix factorization
problem in the SourceID-NMF method.
"""

import concurrent.futures
import logging
from typing import Tuple

import numpy as np
from numpy.linalg import inv
from scipy.optimize import root_scalar
from tqdm import tqdm

logger = logging.getLogger("sourceid_nmf.admm")


def w_update(w_plus: np.ndarray, alpha_w: np.ndarray, h: np.ndarray, x: np.ndarray, i: np.ndarray, rho: float) -> np.ndarray:
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
    return (x.dot(h.T) + rho * w_plus - alpha_w).dot(inv(h.dot(h.T) + rho * i))


def h_update(w_renew: np.ndarray, x: np.ndarray, i: np.ndarray, h_plus: np.ndarray, alpha_h: np.ndarray, rho: float) -> np.ndarray:
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
    return inv(w_renew.T.dot(w_renew) + rho * i).dot(w_renew.T.dot(x) + rho * h_plus - alpha_h)


def w_add(w_renew: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Apply normalization to W.

    Args:
        w_renew: Updated W matrix
        d: Diagonal matrix for normalization

    Returns:
        Normalized W matrix
    """
    return w_renew.dot(inv(d))


def h_add(h_renew: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Apply normalization to H.

    Args:
        h_renew: Updated H matrix
        d: Diagonal matrix for normalization

    Returns:
        Normalized H matrix
    """
    return d.dot(h_renew)


def water_filling_w(beta_w: float, w: np.ndarray, alpha_w: np.ndarray, y: np.ndarray, a: np.ndarray, rho: float) -> float:
    """
    Water-filling function for W+ update.

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
    w1 = np.power(a, 2) * y + alpha_w + rho * w - beta_w
    w2 = np.power(a, 2) + rho
    return np.sum(np.maximum(0, w1 / w2)) - 1


def water_filling_h(beta_h: float, h: np.ndarray, alpha_h: np.ndarray, rho: float) -> float:
    """
    Water-filling function for H+ update.

    Args:
        beta_h: Beta parameter
        h: Current H matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Sum of maximum values minus 1
    """
    return np.sum(np.maximum(0, (alpha_h + rho * h - beta_h) / rho)) - 1


def root_scalar_w(w: np.ndarray, alpha_w: np.ndarray, y: np.ndarray, a: np.ndarray, rho: float) -> float:
    """
    Find the root of the water-filling function for W.

    Args:
        w: Current W matrix
        alpha_w: Lagrangian multiplier for W
        y: Source data matrix
        a: Weighting matrix
        rho: Penalty parameter

    Returns:
        Root value
    """
    sol = root_scalar(water_filling_w, args=(w, alpha_w, y, a, rho), method='bisect', bracket=[-10000, 10000])
    return sol.root


def root_scalar_h(h: np.ndarray, alpha_h: np.ndarray, rho: float) -> float:
    """
    Find the root of the water-filling function for H.

    Args:
        h: Current H matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Root value
    """
    sol = root_scalar(water_filling_h, args=(h, alpha_h, rho), method='bisect', bracket=[-10000, 10000])
    return sol.root


def call_beta(args: tuple) -> float:
    """
    Helper function for parallel processing of beta calculation.

    Args:
        args: Tuple containing (index, w_latest, alpha_w, y, a, rho)

    Returns:
        Root value for the given column
    """
    i, w_latest, alpha_w, y, a, rho = args
    root = root_scalar_w(w_latest[:, i], alpha_w[:, i], y[:, i], a[:, i], rho)
    return root


def w_plus_update(a: np.ndarray, y: np.ndarray, w_latest: np.ndarray, alpha_w: np.ndarray, rho: float, thread: int) -> np.ndarray:
    """
    Update W+ in the ADMM algorithm with parallel processing.

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
    logger.debug(f"Updating W+ using {thread} threads")
    query_list = [(i, w_latest, alpha_w, y, a, rho) for i in range(w_latest.shape[1])]

    with concurrent.futures.ProcessPoolExecutor(max_workers=thread) as executor:
        # Use tqdm for progress tracking if in debug mode
        if logger.getEffectiveLevel() <= logging.DEBUG:
            beta = list(tqdm(executor.map(call_beta, query_list), total=len(query_list), desc="W+ update"))
        else:
            beta = list(executor.map(call_beta, query_list))

    # Calculate W+ using the obtained beta values
    w1 = np.power(a, 2) * y + alpha_w + rho * w_latest - np.array(beta)
    w2 = np.power(a, 2) + rho
    w_plus = np.maximum(0, w1 / w2)

    return w_plus


def h_plus_update(h_latest: np.ndarray, alpha_h: np.ndarray, rho: float) -> np.ndarray:
    """
    Update H+ in the ADMM algorithm.

    Args:
        h_latest: Latest H matrix
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Updated H+ matrix
    """
    # Find the root for H+ update
    root = root_scalar_h(h_latest, alpha_h, rho)

    # Calculate H+ using the root
    h_plus = np.maximum(0, (alpha_h + rho * h_latest - root) / rho)

    return h_plus


def alpha_w_update(alpha_w: np.ndarray, w_latest: np.ndarray, w_plus: np.ndarray, rho: float) -> np.ndarray:
    """
    Update Lagrangian multiplier for W.

    Args:
        alpha_w: Current Lagrangian multiplier for W
        w_latest: Latest W matrix
        w_plus: Updated W+ matrix
        rho: Penalty parameter

    Returns:
        Updated Lagrangian multiplier for W
    """
    return alpha_w + rho * (w_latest - w_plus)


def alpha_h_update(alpha_h: np.ndarray, h_latest: np.ndarray, h_plus: np.ndarray, rho: float) -> np.ndarray:
    """
    Update Lagrangian multiplier for H.

    Args:
        alpha_h: Current Lagrangian multiplier for H
        h_latest: Latest H matrix
        h_plus: Updated H+ matrix
        rho: Penalty parameter

    Returns:
        Updated Lagrangian multiplier for H
    """
    return alpha_h + rho * (h_latest - h_plus)


def admm_step(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    i: np.ndarray,
    w_plus: np.ndarray,
    h_plus: np.ndarray,
    alpha_w: np.ndarray,
    alpha_h: np.ndarray,
    rho: float,
    thread: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one ADMM iteration step.

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

    Returns:
        Tuple of updated matrices: (w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h)
    """
    # Update H
    h_renew = h_update(w, x, i, h_plus, alpha_h, rho)

    # Update W
    w_renew = w_update(w_plus, alpha_w, h_renew, x, i, rho)

    # Normalize W and H
    d = np.diag(np.sum(w_renew, axis=0))
    h_renew = h_add(h_renew, d)
    w_renew = w_add(w_renew, d)

    # Update auxiliary variables H+ and W+
    h_plus = h_plus_update(h_renew, alpha_h, rho)
    w_plus = w_plus_update(a, y, w_renew, alpha_w, rho, thread)

    # Update Lagrangian multipliers
    alpha_w = alpha_w_update(alpha_w, w_renew, w_plus, rho)
    alpha_h = alpha_h_update(alpha_h, h_renew, h_plus, rho)

    return w_renew, h_renew, w_plus, h_plus, alpha_w, alpha_h


def lagrangian(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    a: np.ndarray,
    w_plus: np.ndarray,
    h_plus: np.ndarray,
    alpha_w: np.ndarray,
    alpha_h: np.ndarray,
    rho: float
) -> float:
    """
    Calculate the Lagrangian function value.

    Args:
        x: Data matrix X
        y: Source data matrix Y
        w: Current W matrix
        h: Current H matrix
        a: Weighting matrix
        w_plus: Current W+ matrix
        h_plus: Current H+ matrix
        alpha_w: Lagrangian multiplier for W
        alpha_h: Lagrangian multiplier for H
        rho: Penalty parameter

    Returns:
        Lagrangian function value
    """
    # Data fidelity term
    l1 = 0.5 * np.linalg.norm(x - w.dot(h))

    # Source fidelity term
    l2 = 0.5 * np.linalg.norm(np.multiply(a, (w_plus - y)))

    # Augmented Lagrangian terms for W
    l3 = np.sum(np.diag(alpha_w.T.dot(w - w_plus)))
    l4 = (rho / 2) * np.linalg.norm(w - w_plus)

    # Augmented Lagrangian terms for H
    l5 = np.sum(np.diag(alpha_h.T.dot(h - h_plus)))
    l6 = (rho / 2) * np.linalg.norm(h - h_plus)

    return l1 + l2 + l3 + l4 + l5 + l6