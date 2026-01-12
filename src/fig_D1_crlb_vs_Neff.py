#!/usr/bin/env python3
"""
Figure D1: Monte Carlo verification of the CRLB for time estimation
in normal diffusion: Var(t_hat) ~ 1 / N_eff.

Appendix D.2 setting:
- Propagator: P(x|t) = (4πDt)^(-d/2) exp(-|x|^2/(4Dt))
- Known diffusion coefficient D
- Negligible initial width (sigma0 = 0)

We generate N_eff independent d-dimensional Gaussian displacement vectors
from the diffusion propagator x ~ N(0, 2 D t I_d), construct the ML estimator
for t via the mean squared displacement, and compare the empirical variance
to the theoretical CRLB.

Theory (Appendix D):
    Var(t_hat) >= 2 t^2 / (d N_eff)

Output:
    figs/fig_D1_crlb_vs_Neff.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_FIG = FIG_DIR / "fig_D1_crlb_vs_Neff.pdf"


# ---------------------------------------------------------------------
# Model and estimators
# ---------------------------------------------------------------------

def sample_displacements(N: int, d: int, D: float, t_abs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw N independent displacement vectors from N(0, 2 D t I_d).
    Here sigma0 = 0 (no initial width / no localization floor).
    """
    sigma2 = 2.0 * D * t_abs  # per-coordinate variance
    return rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=(N, d))


def estimate_t_from_variance(X: np.ndarray, D: float) -> float:
    """
    ML estimator for time from mean squared displacement:
        t_hat = <|x|^2> / (2 d D)
    """
    _, d = X.shape
    r2_mean = float(np.mean(np.sum(X**2, axis=1)))
    return r2_mean / (2.0 * d * D)


def crlb_variance(t_abs: float, d: int, N_eff: int) -> float:
    """
    CRLB for time estimation variance (Appendix D.2):
        Var(t_hat) >= 2 t^2 / (d N_eff)
    """
    return 2.0 * t_abs**2 / (d * N_eff)


# ---------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------

def mc_variance_t_hat(
    N_eff: int,
    d: int,
    D: float,
    t_abs: float,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """
    Empirical variance of t_hat over Monte Carlo trials.
    N_eff is the number of independent samples (trajectories/observations) per trial.
    """
    t_hats = np.empty(n_trials, dtype=float)
    for k in range(n_trials):
        X = sample_displacements(N_eff, d, D, t_abs, rng)
        t_hats[k] = estimate_t_from_variance(X, D)
    return float(np.var(t_hats, ddof=1))


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit slope of y ~ x^m on log-log axes (returns m)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main() -> None:
    # -----------------------------------------------------------------
    # Fixed parameters
    # -----------------------------------------------------------------
    d = 2
    D = 1.0
    t_abs = 1.0

    # Deterministic RNG for reproducibility
    rng = np.random.default_rng(1234)

    # Effective sample sizes (log-spaced)
    N_eff_vals = np.unique(np.logspace(1, 4, 14).astype(int))

    # Monte Carlo repetitions per N_eff
    n_trials = 2000

    # -----------------------------------------------------------------
    # Run simulations
    # -----------------------------------------------------------------
    var_emp = np.array([mc_variance_t_hat(N, d, D, t_abs, n_trials, rng) for N in N_eff_vals], dtype=float)
    var_crlb = np.array([crlb_variance(t_abs, d, N) for N in N_eff_vals], dtype=float)

    # Fit slope on log-log (all points + asymptotic tail)
    slope_all = fit_loglog_slope(N_eff_vals, var_emp)

    tail_mask = N_eff_vals >= 100
    slope_tail = fit_loglog_slope(N_eff_vals[tail_mask], var_emp[tail_mask])

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    plt.figure(figsize=(6.2, 4.2))

    plt.loglog(
        N_eff_vals,
        var_emp,
        "o",
        label=fr"Monte Carlo: slope(all)={slope_all:.3f}, slope($N_{{\rm eff}}\geq 100$)={slope_tail:.3f}",
    )
    plt.loglog(
        N_eff_vals,
        var_crlb,
        "-",
        label=r"CRLB: $2t^2/(dN_{\mathrm{eff}})$",
    )

    plt.xlabel(r"Effective sample size $N_{\mathrm{eff}}$")
    plt.ylabel(r"$\mathrm{Var}(\hat t)$")
    plt.title("CRLB verification for time estimation (normal diffusion)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure saved to {OUTPUT_FIG}")
    print(f"[INFO] Fitted slope (all points):  {slope_all:.6f}  (expected: -1.0)")
    print(f"[INFO] Fitted slope (N_eff>=100):  {slope_tail:.6f}  (expected: -1.0)")
    print("[NOTE] Deviations at small N_eff are finite-sample effects.")


if __name__ == "__main__":
    main()
