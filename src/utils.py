#!/usr/bin/env python3
# ============================================================
# utils.py
# ============================================================
# Shared utility functions for simulations in the repository.
#
# Author: Alexander Yashin
# ============================================================

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def set_seed(seed: int = 0) -> np.random.Generator:
    """
    Initialize and return a NumPy random Generator with a fixed seed.

    Note:
    This function does NOT set any global RNG state; it returns an explicit
    Generator instance to be threaded through simulations for reproducibility.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    return np.random.default_rng(int(seed))


def loglog_fit(x: ArrayLike, y: ArrayLike) -> Tuple[float, float]:
    """
    Perform a log-log linear fit:
        log(y) = slope * log(x) + intercept

    Returns:
        (slope, intercept)

    Strictness:
    - Requires x>0 and y>0 elementwise (log defined).
    - Requires at least 2 finite data points.

    Note:
    This returns only point estimates (no standard errors). If you need
    uncertainty for the slope, fit via least squares and compute SE explicitly.
    """
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")
    if x_arr.size < 2:
        raise ValueError("Need at least 2 points for loglog_fit.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("x and y must be finite.")
    if np.any(x_arr <= 0.0) or np.any(y_arr <= 0.0):
        raise ValueError("loglog_fit requires positive x and y.")

    logx = np.log(x_arr)
    logy = np.log(y_arr)

    slope, intercept = np.polyfit(logx, logy, 1)
    return float(slope), float(intercept)


def confidence_interval(data: ArrayLike, level: float = 0.68) -> Tuple[float, float, float]:
    """
    Compute a symmetric normal-approximation confidence interval for the MEAN.

    Parameters:
        data : array-like
            Sample values.
        level : float
            Confidence level for the mean. Supported: 0.68 (~1σ), 0.95, 0.99.

    Returns:
        (mean, lower, upper) for the mean under normal approximation:
            mean ± z * s / sqrt(n)

    Notes:
    - This is NOT a distribution quantile band for the raw data.
    - For small n or heavy tails, a t-interval or bootstrap is more appropriate.
    """
    arr = np.asarray(list(data), dtype=float)

    if arr.size < 2:
        raise ValueError("Need at least 2 samples to compute an interval.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("data must be finite.")

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    n = int(arr.size)

    z_map = {
        0.68: 1.0,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_map.get(float(level), None)
    if z is None:
        raise ValueError(f"Unsupported confidence level: {level}. Supported: {sorted(z_map.keys())}")

    half_width = float(z * std / np.sqrt(n))
    return mean, mean - half_width, mean + half_width


def effective_sample_size(n: Union[int, float], autocorr_time: float = 1.0) -> float:
    """
    Estimate effective sample size for correlated data using an integrated
    autocorrelation time approximation.

    Parameters:
        n : int or float
            Total number of samples (nominal).
        autocorr_time : float
            Integrated autocorrelation time τ_int (in units of sample steps).

    Returns:
        n_eff ≈ n / (2 τ_int)

    Strictness / sanity:
    - Enforces autocorr_time > 0.
    - Clamps n_eff to (0, n] for physical plausibility.
    """
    n_float = float(n)
    if not np.isfinite(n_float) or n_float <= 0.0:
        raise ValueError("n must be a positive finite number.")
    if not np.isfinite(autocorr_time) or autocorr_time <= 0.0:
        raise ValueError("autocorr_time must be positive and finite.")

    n_eff = n_float / (2.0 * float(autocorr_time))
    # n_eff should not exceed n in this approximation
    n_eff = min(n_eff, n_float)
    return float(n_eff)
