#!/usr/bin/env python3
"""
Figure D2: Photon-limited temporal resolution scaling Δt_min ∝ Φ^{-1/3}.

Appendix D.3 canonical model
---------------------------
Observed (per-coordinate aggregated) width:
    sigma_obs^2(t) = sigma0^2 + 2 d D t

Poisson photons: N_gamma = Φ Δt

Variance model for an (approximately) unbiased estimator of sigma^2:
    Var( (sigma^2)^ ) ≃ κ * sigma_obs^4 / N_gamma     with κ≈2 (ML)  [Ober 2004]

Time inference via sigma_obs^2 gives:
    Var(t^) = Var( (sigma^2)^ ) / (2 d D)^2

Self-consistency (Δt_min ≃ sqrt(Var(t^))) yields:
    Δt_min^3 = κ * sigma_obs^4 / ( (2 d D)^2 * Φ )
            = κ * sigma_obs^4 / ( 4 d^2 D^2 Φ )

In Appendix D.3, the scaling is written (up to prefactors) as:
    Δt_min ~ [ sigma_obs^4 / (2 d^2 D^2 Φ) ]^{1/3}
We keep κ explicit and implement the consistent cube-root law.

Expected on log-log axes: slope ≈ -1/3.

Output
------
    figs/fig_D2_photon_scaling.pdf
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

OUTPUT_FIG = FIG_DIR / "fig_D2_photon_scaling.pdf"


# ---------------------------------------------------------------------
# Theory: closed-form Δt_min from the cubic self-consistency
# ---------------------------------------------------------------------

def dt_min_theory(
    phi: np.ndarray,
    d: int,
    D: float,
    sigma0: float,
    t_abs: float,
    kappa: float = 2.0,
) -> np.ndarray:
    """
    Closed-form solution consistent with Appendix D.3:

        Δt_min = [ κ * sigma_obs^4 / (4 d^2 D^2 Φ) ]^{1/3},
    where
        sigma_obs^2 = sigma0^2 + 2 d D t.

    Notes
    -----
    - sigma0 is a *standard deviation*; sigma0^2 enters sigma_obs^2.
    - The scaling exponent is the key validation target; κ controls only a prefactor.
    """
    sigma_obs2 = sigma0**2 + 2.0 * d * D * t_abs
    return (kappa * (sigma_obs2**2) / (4.0 * (d**2) * (D**2) * phi)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------
# Optional Monte Carlo: fixed-point loop using Poisson photon counts
# ---------------------------------------------------------------------

def mc_dt_min_fixed_point(
    phi: float,
    d: int,
    D: float,
    sigma0: float,
    t_abs: float,
    kappa: float = 2.0,
    n_trials: int = 200,
    max_iter: int = 30,
    tol_rel: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo fixed-point loop implementing the same information model:

        1) start with trial Δt
        2) draw Nγ ~ Poisson(Φ Δt)
        3) Var( (sigma^2)^ ) ≃ κ sigma_obs^4 / Nγ
        4) Var(t^) = Var( (sigma^2)^ ) / (2 d D)^2
        5) update Δt <- sqrt( Var(t^) )
        6) iterate to fixed point

    Averaged across trials to smooth Poisson noise.

    This validates scaling + fixed-point behavior under the same model.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    sigma_obs2 = sigma0**2 + 2.0 * d * D * t_abs
    dt_init = float((kappa * (sigma_obs2**2) / (4.0 * (d**2) * (D**2) * phi)) ** (1.0 / 3.0))

    dts = []
    for _ in range(n_trials):
        dt = dt_init

        for __ in range(max_iter):
            lam = max(phi * dt, 1e-12)
            Ng = int(rng.poisson(lam=lam))

            if Ng <= 0:
                dt = min(dt * 1.5, 1e6)
                continue

            var_sigma2 = kappa * (sigma_obs2**2) / Ng  # sigma_obs^4 = (sigma_obs2)^2
            var_t = var_sigma2 / (2.0 * d * D) ** 2
            dt_new = float(np.sqrt(var_t))

            if abs(dt_new - dt) / max(dt, 1e-12) < tol_rel:
                dt = dt_new
                break
            dt = dt_new

        dts.append(dt)

    return float(np.mean(dts))


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit slope of y ~ x^m on log-log axes (returns m)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main() -> None:
    # -----------------------------------------------------------------
    # Parameters (representative values)
    # -----------------------------------------------------------------
    d = 2
    D = 1.0
    sigma0 = 1.0     # std; enters as sigma0^2 in sigma_obs^2
    t_abs = 1.0      # elapsed time at which we infer t
    kappa = 2.0

    # Control whether to run Monte Carlo overlay (can be slower)
    RUN_MC = True

    # Photon flux range
    phi = np.logspace(2, 6, 25)  # photons / time unit

    # Theory curve
    dt_th = dt_min_theory(phi, d=d, D=D, sigma0=sigma0, t_abs=t_abs, kappa=kappa)

    # Monte Carlo points (optional)
    dt_mc = None
    slope_mc = None
    if RUN_MC:
        rng = np.random.default_rng(7)
        dt_mc = np.array(
            [
                mc_dt_min_fixed_point(
                    float(p),
                    d=d,
                    D=D,
                    sigma0=sigma0,
                    t_abs=t_abs,
                    kappa=kappa,
                    n_trials=120,
                    max_iter=25,
                    tol_rel=2e-3,
                    rng=rng,
                )
                for p in phi
            ],
            dtype=float,
        )
        slope_mc = fit_loglog_slope(phi, dt_mc)

    # Slopes (theory + asymptotic tail)
    slope_th = fit_loglog_slope(phi, dt_th)
    tail_mask = phi >= 1e3
    slope_th_tail = fit_loglog_slope(phi[tail_mask], dt_th[tail_mask])

    # Reference line with slope -1/3 anchored at mid-phi
    mid = len(phi) // 2
    phi0 = phi[mid]
    dt0 = dt_th[mid]
    dt_ref = dt0 * (phi / phi0) ** (-1.0 / 3.0)

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    plt.figure(figsize=(6.2, 4.2))

    plt.loglog(
        phi,
        dt_th,
        "-",
        label=fr"Theory (Appendix D.3): slope(all)={slope_th:.3f}, slope($\Phi\geq10^3$)={slope_th_tail:.3f}",
    )
    if RUN_MC:
        plt.loglog(
            phi,
            dt_mc,
            "o",
            markersize=4,
            label=fr"MC fixed-point: slope(all)={slope_mc:.3f}",
        )
    plt.loglog(phi, dt_ref, "--", linewidth=1.0, label=r"Reference slope $-1/3$")

    plt.xlabel(r"Photon flux $\Phi$ (photons / time)")
    plt.ylabel(r"Minimal temporal resolution $\Delta t_{\min}$")
    plt.title(r"Photon-limited scaling: $\Delta t_{\min}\propto \Phi^{-1/3}$")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure saved to {OUTPUT_FIG}")
    print(f"[INFO] Fitted slope (theory, all):      {slope_th:.6f}  (expected: -0.333333...)")
    print(f"[INFO] Fitted slope (theory, Phi>=1e3): {slope_th_tail:.6f}  (expected: -0.333333...)")
    if RUN_MC:
        print(f"[INFO] Fitted slope (MC, all):          {slope_mc:.6f}  (expected: -0.333333...)")
    print("[NOTE] MC uses the same information model (Poisson photons + estimator-variance law),")
    print("       so it validates scaling and fixed-point behavior, not detector specifics.")


if __name__ == "__main__":
    main()
