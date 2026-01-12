#!/usr/bin/env python3
"""
Figure D3: Numerical equivalence between KL divergence and Fisher information
for time distinguishability in normal diffusion.

Goal
----
Verify the local expansion:
    D_KL(P_t || P_{t+Δt}) ≈ (1/2) I(t) (Δt)^2
for small Δt.

This establishes numerical equivalence between:
- hypothesis-testing distinguishability (KL),
- estimation-theoretic limits (CRLB / Fisher).

Model
-----
Isotropic diffusion in d dimensions:
    x ~ N(0, 2 D t I_d)
with per-coordinate variance var(t) = 2 D t.

Output
------
    figs/fig_D3_kl_vs_crlb.pdf
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

OUTPUT_FIG = FIG_DIR / "fig_D3_kl_vs_crlb.pdf"


# ---------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------

def gaussian_logpdf(x: np.ndarray, var: float) -> np.ndarray:
    """
    Log-pdf of N(0, var I_d) evaluated at samples x (shape: (N, d)),
    where 'var' is the per-coordinate variance.
    """
    d = x.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi * var) + np.sum(x**2, axis=1) / var)


def kl_gaussian_mc(x: np.ndarray, var_t: float, var_tp: float) -> float:
    """
    Monte Carlo estimate of KL divergence:
        D_KL(P_t || P_{t+Δt}) = E_{x~P_t}[ log p(x|t) - log p(x|t+Δt) ].
    """
    return float(np.mean(gaussian_logpdf(x, var_t) - gaussian_logpdf(x, var_tp)))


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit slope of y ~ x^m on log-log axes (returns m)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(123)

    # Physical parameters
    d = 2
    D = 1.0
    t0 = 1.0

    # per-coordinate variance at t0
    var_t = 2.0 * D * t0

    # Small Δt values (local regime)
    delta_t = np.logspace(-3, -1, 12)

    # Samples per KL estimate (can be lowered if runtime is an issue)
    N = 200_000

    kl_empirical = np.empty_like(delta_t, dtype=float)
    kl_fisher = np.empty_like(delta_t, dtype=float)

    # Fisher information for a single d-dim observation:
    # I1(t) = d / (2 t^2)  (independent of D for this parametrization)
    I1 = d / (2.0 * t0**2)

    for i, dt in enumerate(delta_t):
        var_tp = 2.0 * D * (t0 + dt)

        # Draw samples from P_t
        x = rng.normal(loc=0.0, scale=np.sqrt(var_t), size=(N, d))

        # Monte Carlo KL
        kl_empirical[i] = kl_gaussian_mc(x, var_t, var_tp)

        # Fisher quadratic approximation
        kl_fisher[i] = 0.5 * I1 * dt**2

    # Fit slopes on log-log (KL should scale ~ (Δt)^2)
    slope_mc = fit_loglog_slope(delta_t, kl_empirical)
    slope_fi = fit_loglog_slope(delta_t, kl_fisher)

    # Reference line with slope +2 anchored at smallest Δt
    dt0 = delta_t[0]
    y0 = kl_fisher[0]
    kl_ref = y0 * (delta_t / dt0) ** 2

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    plt.figure(figsize=(6.2, 4.2))

    plt.loglog(delta_t, kl_empirical, "o", label=fr"Monte Carlo KL (slope={slope_mc:.3f})")

    # IMPORTANT: Matplotlib mathtext does not accept \frac12, use \frac{1}{2}.
    plt.loglog(
        delta_t,
        kl_fisher,
        "-",
        label=fr"Fisher approx. $\frac{{1}}{{2}} I(t)(\Delta t)^2$ (slope={slope_fi:.3f})",
    )

    plt.loglog(delta_t, kl_ref, "--", linewidth=1.0, label=r"Reference slope $+2$")

    plt.xlabel(r"Time increment $\Delta t$")
    plt.ylabel(r"$D_{\mathrm{KL}}$")
    plt.title("KL divergence vs Fisher information (local regime)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure saved to {OUTPUT_FIG}")
    print(f"[INFO] Fitted slope (MC):     {slope_mc:.6f}  (expected: +2.0 in local regime)")
    print(f"[INFO] Fitted slope (Fisher): {slope_fi:.6f}  (expected: +2.0)")


if __name__ == "__main__":
    main()
