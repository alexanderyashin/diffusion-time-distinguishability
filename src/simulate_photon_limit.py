#!/usr/bin/env python3
# ============================================================
# simulate_photon_limit.py
# ============================================================
# Numerical verification of the photon-limited temporal
# distinguishability scaling:
#     Δt_min ∝ Φ^{-1/3}
#
# IMPORTANT (role separation):
# - t0 is the elapsed physical diffusion time parameter we want to infer.
# - Δt is the acquisition/exposure window that determines photon count:
#       N_gamma ~ Poisson(Φ Δt).
# - Self-consistency arises because achieving smaller Δt requires photons,
#   but photons require exposure time.
#
# This script verifies the Φ^{-1/3} scaling in a way consistent with the
# manuscript's photon-limited derivation.
#
# Author: Alexander Yashin
# ============================================================

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Params:
    d: int = 2                  # spatial dimension
    D: float = 1.0e-12          # diffusion coefficient [m^2/s]
    sigma0: float = 200e-9      # PSF width / localization floor [m]
    t0: float = 5.0e-3          # elapsed time parameter to be inferred [s]
    kappa: float = 2.0          # Var(sigma^2_hat) = kappa * sigma^4 / N_gamma
    z: float = 1.0              # confidence multiplier (1.0 = 1σ)
    n_iter: int = 25            # fixed-point iterations
    n_trials: int = 2000        # Monte Carlo trials per Φ (keep CI-friendly)
    seed: int = 42              # RNG seed
    min_photons: int = 5        # minimal photon count for stable variance estimate
    max_photon_resample: int = 12  # resample attempts per iteration for too-low counts


def analytic_dt_min(phi: float, p: Params) -> float:
    """
    Closed-form self-consistent prediction (deterministic, no Poisson noise).

    Model:
      sigma_obs^2(t0) = sigma0^2 + 2 d D t0
      Var(t_hat) ≈ [kappa / (4 d^2 D^2)] * sigma_obs^4 / N_gamma
      N_gamma = phi * Δt
      Self-consistency: Δt = z * sqrt(Var(t_hat))

    => Δt^3 = z^2 * [kappa / (4 d^2 D^2)] * sigma_obs^4 / phi
    """
    if phi <= 0.0:
        raise ValueError("phi must be positive.")
    if p.D <= 0.0:
        raise ValueError("D must be positive.")
    if p.d <= 0:
        raise ValueError("d must be a positive integer.")
    if p.t0 <= 0.0:
        raise ValueError("t0 must be positive.")
    if p.sigma0 < 0.0:
        raise ValueError("sigma0 must be >= 0.")
    if p.kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    if p.z <= 0.0:
        raise ValueError("z must be positive.")

    sigma2 = p.sigma0**2 + 2.0 * p.d * p.D * p.t0
    pref = (p.z**2) * (p.kappa / (4.0 * (p.d**2) * (p.D**2))) * (sigma2**2)
    dt = (pref / phi) ** (1.0 / 3.0)
    return float(dt)


def fixed_point_mc(phi: float, p: Params, rng: np.random.Generator) -> np.ndarray:
    """
    Monte Carlo fixed-point iteration with Poisson photon counts.

    Returns:
      dt_estimates: array of length n_trials containing Δt_min estimates.

    Strictness note:
    - If N_gamma is too small, the variance proxy A/N_gamma becomes unstable.
      We therefore enforce a minimal photon regime by *increasing Δt and resampling*
      N_gamma from the correct Poisson(ΦΔt). This preserves the generative model.
    """
    if phi <= 0.0:
        raise ValueError("phi must be positive.")
    if p.min_photons < 1:
        raise ValueError("min_photons must be >= 1.")
    if p.max_photon_resample < 1:
        raise ValueError("max_photon_resample must be >= 1.")

    # Use analytic prediction as an initialization (stabilizes convergence).
    dt = np.full(p.n_trials, max(analytic_dt_min(phi, p), 1e-12), dtype=float)

    # sigma_obs depends on elapsed time parameter t0, not on Δt.
    sigma2 = p.sigma0**2 + 2.0 * p.d * p.D * p.t0

    # Var(t_hat) = A / N_gamma, where A = kappa * sigma_obs^4 / (4 d^2 D^2)
    A = p.kappa * (sigma2**2) / (4.0 * (p.d**2) * (p.D**2))

    for _ in range(p.n_iter):
        # Draw Poisson counts with current dt
        lam = np.maximum(phi * dt, 1e-300)
        N_gamma = rng.poisson(lam=lam)

        # If too few photons: increase dt and resample from Poisson(ΦΔt).
        # We do bounded retries to keep runtime predictable (CI-friendly).
        too_low = N_gamma < p.min_photons
        if np.any(too_low):
            dt_work = dt.copy()
            N_work = N_gamma.copy()

            for _k in range(p.max_photon_resample):
                if not np.any(too_low):
                    break
                dt_work[too_low] *= 2.0
                lam2 = np.maximum(phi * dt_work[too_low], 1e-300)
                N_new = rng.poisson(lam=lam2)
                N_work[too_low] = N_new
                # update mask
                too_low = N_work < p.min_photons

            # If still too low after retries, clamp to min_photons as a last resort
            # (this only affects extremely low-Φ tails and prevents division blowups).
            N_gamma = np.maximum(N_work, p.min_photons)
            dt = dt_work
        else:
            N_gamma = np.maximum(N_gamma, p.min_photons)

        var_t = A / N_gamma.astype(float)

        # Self-consistent update: Δt := z * sqrt(Var(t_hat))
        dt = p.z * np.sqrt(var_t)

    return dt


def fit_slope(log_x: np.ndarray, log_y: np.ndarray) -> Tuple[float, float]:
    """
    Fit log_y = a + b log_x; return (b, SE_b).
    """
    x = np.asarray(log_x, dtype=float)
    y = np.asarray(log_y, dtype=float)
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points for slope SE.")

    X = np.vstack([np.ones_like(x), x]).T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta[0], beta[1]

    y_hat = a + b * x
    resid = y - y_hat
    s2 = np.sum(resid**2) / (n - 2)
    Sxx = np.sum((x - np.mean(x))**2)
    se_b = np.sqrt(s2 / Sxx)
    return float(b), float(se_b)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify photon-limited scaling Δt_min ∝ Φ^{-1/3}.")
    ap.add_argument("--t0", type=float, default=5.0e-3, help="Elapsed diffusion time parameter t0 [s].")
    ap.add_argument("--D", type=float, default=1.0e-12, help="Diffusion coefficient D [m^2/s].")
    ap.add_argument("--d", type=int, default=2, help="Dimension d.")
    ap.add_argument("--sigma0", type=float, default=200e-9, help="PSF/localization floor sigma0 [m].")
    ap.add_argument("--kappa", type=float, default=2.0, help="Variance-estimator coefficient kappa (ideal Gaussian -> 2).")
    ap.add_argument("--z", type=float, default=1.0, help="Confidence multiplier z (1.0=1σ, 1.96=95%%).")
    ap.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials per Φ.")
    ap.add_argument("--iters", type=int, default=25, help="Fixed-point iterations.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--min-photons", type=int, default=5, help="Minimal photon count per update (stability).")
    ap.add_argument("--max-resample", type=int, default=12, help="Max Poisson resample attempts for too-low counts.")
    ap.add_argument("--out", type=str, default="photon_limit_results.txt", help="Output file.")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    p = Params(
        d=args.d,
        D=args.D,
        sigma0=args.sigma0,
        t0=args.t0,
        kappa=args.kappa,
        z=args.z,
        n_iter=args.iters,
        n_trials=args.trials,
        seed=args.seed,
        min_photons=args.min_photons,
        max_photon_resample=args.max_resample,
    )

    # Photon flux range [photons / second]
    phi_values = np.logspace(3, 7, 20)

    # Independent deterministic RNG substreams per Φ (clean reproducibility)
    ss = np.random.SeedSequence(p.seed)
    child_seeds = ss.spawn(len(phi_values))

    # Analytic baseline (deterministic)
    dt_analytic = np.array([analytic_dt_min(float(phi), p) for phi in phi_values], dtype=float)

    # Monte Carlo verification
    dt_mc_mean = []
    dt_mc_std = []

    for i, phi in enumerate(phi_values):
        rng_i = np.random.default_rng(child_seeds[i])
        dt_samples = fixed_point_mc(phi=float(phi), p=p, rng=rng_i)
        dt_mc_mean.append(float(np.mean(dt_samples)))
        dt_mc_std.append(float(np.std(dt_samples, ddof=1)))

    dt_mc_mean = np.array(dt_mc_mean, dtype=float)
    dt_mc_std = np.array(dt_mc_std, dtype=float)

    # Log-log fit for scaling using Monte Carlo means
    log_phi = np.log(phi_values)
    log_dt = np.log(dt_mc_mean)
    slope, slope_se = fit_slope(log_phi, log_dt)

    print("Photon-limited temporal resolution scaling (self-consistent)")
    print("----------------------------------------------------------")
    print("Model roles: elapsed parameter t0 fixed; acquisition window Δt controls photons Nγ ~ Poisson(ΦΔt)")
    print(f"Parameters: d={p.d}, D={p.D:.3e} m^2/s, sigma0={p.sigma0:.3e} m, t0={p.t0:.3e} s")
    print(f"Variance model: Var(sigma^2_hat)=kappa*sigma_obs^4/Nγ with kappa={p.kappa:g}, confidence z={p.z:g}")
    print(f"Monte Carlo: trials/Φ={p.n_trials}, iterations={p.n_iter}, seed={p.seed}")
    print(f"Stability: min_photons={p.min_photons}, max_resample={p.max_photon_resample}")
    print()
    print(f"Fitted scaling exponent (MC means): {slope:.4f} ± {slope_se:.4f} (SE)")
    print("Expected theoretical value: -1/3 ≈ -0.3333")
    print()

    print("Φ [photons/s]    Δt_min_MC [s]    std_MC [s]     Δt_min_analytic [s]")
    for phi, dtm, dts, dta in zip(phi_values, dt_mc_mean, dt_mc_std, dt_analytic):
        print(f"{phi:10.3e}    {dtm:12.4e}   {dts:10.3e}    {dta:12.4e}")

    # Save results
    np.savetxt(
        args.out,
        np.column_stack((phi_values, dt_mc_mean, dt_mc_std, dt_analytic)),
        header="Phi [photons/s]   Delta_t_min_MC_mean [s]   Delta_t_min_MC_std [s]   Delta_t_min_analytic [s]",
    )
    print()
    print(f"Saved: {args.out}")
