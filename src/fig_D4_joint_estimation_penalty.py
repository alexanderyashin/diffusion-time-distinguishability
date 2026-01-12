#!/usr/bin/env python3
"""
Figure D4: Joint-estimation penalty for time inference in normal diffusion.

Strict statement of the issue
-----------------------------
For d-dimensional Brownian motion with propagator
    x ~ N(0, s I_d),   s(t) = 2 D t + b,   b := sigma0^2,
a *single-time* dataset enters the likelihood only through the scalar variance s.
Therefore, joint estimation of (t, D, b) from single-time data is NOT identifiable:
the observation Fisher information matrix (FIM) is rank-1 in (t, D, b).
No amount of increasing N_eff resolves this, because the mapping
    (t, D, b) -> s
is many-to-one at fixed t.

Consequences:
1) Single-time, no extra info: joint CRLB for t is infinite (non-identifiable).
2) Single-time + calibration priors on D and/or b: one can compute a *posterior*
   (Bayesian) CRLB via adding prior Fisher terms. This produces a finite variance,
   but may become prior-limited (plateau) and need not preserve 1/N_eff scaling.
3) Multi-time data (minimal identifiability): observing widths at several times
   (e.g., t, 2t, 3t) makes (t, D, b) identifiable from Fisher geometry alone.
   In that identifiable setting, nuisance coupling yields a *multiplicative penalty*
   that preserves the 1/N_eff scaling exponent.

What this figure shows (strict, two curves)
------------------------------------------
We plot the inflation factor Var_joint / Var_known versus the regime parameter
    r = (2 D t) / b
and show:
- Identifiable multi-time joint penalty (no priors; frequentist CRLB).
- Non-identifiable single-time + priors (posterior-CRLB; shown as a separate curve).

Output
------
    figs/fig_D4_joint_estimation_penalty.pdf
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

OUTPUT_FIG = FIG_DIR / "fig_D4_joint_estimation_penalty.pdf"


# ---------------------------------------------------------------------
# Helpers (numerical linear algebra)
# ---------------------------------------------------------------------

def safe_inv_sym(A: np.ndarray, cond_max: float = 1e12) -> np.ndarray:
    """
    Invert a symmetric positive (semi-)definite matrix with a stability guard.
    Uses inv if well-conditioned, otherwise uses pseudoinverse.

    This is appropriate here because Fisher matrices are symmetric PSD.
    """
    # Symmetrize to avoid tiny numerical asymmetries
    A = 0.5 * (A + A.T)
    c = np.linalg.cond(A)
    if not np.isfinite(c) or c > cond_max:
        return np.linalg.pinv(A)
    return np.linalg.inv(A)


# ---------------------------------------------------------------------
# Model + Fisher building blocks
# ---------------------------------------------------------------------

def s_variance(t: float, D: float, b: float) -> float:
    """Per-coordinate variance s(t) = 2 D t + b, with b = sigma0^2."""
    return 2.0 * D * t + b


def fim_single_time_observation(t: float, d: int, D: float, b: float, Neff: float) -> np.ndarray:
    """
    Observation Fisher matrix at a single time for θ = [t, D, b].

    For x ~ N(0, s I_d), s = 2Dt + b,
    Fisher for s (per sample) is: I_s = d / (2 s^2).
    With Neff independent samples: I_s_total = Neff * d / (2 s^2).

    Chain rule:
        F = I_s_total * (∂s/∂θ)(∂s/∂θ)^T
    with ∂s/∂[t, D, b] = [2D, 2t, 1].
    This is rank-1 by construction.
    """
    s = s_variance(t, D, b)
    I_s_total = Neff * d / (2.0 * s**2)
    g = np.array([2.0 * D, 2.0 * t, 1.0], dtype=float)
    return I_s_total * np.outer(g, g)


def fim_multi_time_observation(
    t: float,
    d: int,
    D: float,
    b: float,
    Neff: float,
    time_multipliers: np.ndarray,
) -> np.ndarray:
    """
    Observation Fisher matrix for θ = [t, D, b] from multiple time points:
        t_i = m_i * t,  with m_i given by time_multipliers.

    Each time contributes its own rank-1 term; the sum becomes full-rank
    for generic choices of {m_i} (e.g., [1,2,3]).
    """
    F = np.zeros((3, 3), dtype=float)
    for m in time_multipliers:
        ti = float(m * t)
        F += fim_single_time_observation(ti, d, D, b, Neff)
    return 0.5 * (F + F.T)


def var_t_crlb_known_nuisance(
    t: float,
    d: int,
    D: float,
    b: float,
    Neff: float,
    time_multipliers: np.ndarray,
) -> float:
    """
    CRLB for Var(t^) when D and b are known (nuisances fixed),
    using the same multi-time observation model.

    For each time point, F contributes to J_tt as:
        J_tt += I_s_total(t_i) * (∂s/∂t)^2,
    where ∂s/∂t at time t_i is 2D * (dt_i/dt) = 2D * m_i,
    because s depends on t_i = m_i t.

    This yields a strict frequentist benchmark for the identifiable multi-time case.
    """
    J_tt = 0.0
    for m in time_multipliers:
        ti = float(m * t)
        s = s_variance(ti, D, b)
        I_s_total = Neff * d / (2.0 * s**2)
        ds_dt = 2.0 * D * float(m)
        J_tt += I_s_total * (ds_dt**2)
    return 1.0 / J_tt


def var_t_joint_crlb_multi_time(
    t: float,
    d: int,
    D: float,
    b: float,
    Neff: float,
    time_multipliers: np.ndarray,
) -> float:
    """
    Joint (t, D, b) CRLB for Var(t^) in the identifiable multi-time setting:
        Var(t^) >= (F^{-1})_{tt}
    """
    F = fim_multi_time_observation(t, d, D, b, Neff, time_multipliers)
    C = safe_inv_sym(F)
    return float(C[0, 0])


# ---------------------------------------------------------------------
# Single-time + priors (posterior-CRLB)
# ---------------------------------------------------------------------

def var_t_joint_posterior_single_time(
    t: float,
    d: int,
    D: float,
    b: float,
    Neff: float,
    sigma_D_prior: float,
    sigma_b_prior: float,
) -> float:
    """
    Posterior (Bayesian) CRLB for Var(t^) in the non-identifiable single-time case,
    obtained by adding diagonal prior Fisher on (D, b):

        I_prior(D) = 1/sigma_D_prior^2
        I_prior(b) = 1/sigma_b_prior^2

    IMPORTANT:
    - This is not a pure frequentist CRLB from data alone.
    - Scaling with Neff may become prior-limited because identifiability does not
      improve with Neff at single time (data only measure s).
    """
    F_obs = fim_single_time_observation(t, d, D, b, Neff)
    F_prior = np.diag([0.0, 1.0 / (sigma_D_prior**2), 1.0 / (sigma_b_prior**2)])
    F_total = 0.5 * (F_obs + F_obs.T) + F_prior
    C = safe_inv_sym(F_total)
    return float(C[0, 0])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # -----------------------------
    # Parameters (single source)
    # -----------------------------
    d = 2
    D = 1.0
    sigma0 = 0.5
    b = sigma0**2

    # Effective sample size per time point
    Neff = 10_000.0

    # Multi-time identifiability: minimal choice
    time_multipliers = np.array([1.0, 2.0, 3.0], dtype=float)

    # Priors for the *single-time posterior* curve (explicit calibration)
    # Assumption: external calibration uncertainties
    sigma_D_prior = 0.05 * D          # 5% relative prior on D
    sigma_b_prior = 0.10 * b          # 10% relative prior on b = sigma0^2

    # Avoid t=0 (regime parameter diverges)
    t_grid = np.logspace(-3, 1, 300)

    # Regime parameter r = (2 D t) / b  (per-coordinate diffusion variance over floor)
    r = (2.0 * D * t_grid) / b

    # -----------------------------
    # Compute penalties
    # -----------------------------
    # Identifiable multi-time (frequentist CRLB)
    var_known_mt = np.array(
        [var_t_crlb_known_nuisance(float(t), d, D, b, Neff, time_multipliers) for t in t_grid],
        dtype=float,
    )
    var_joint_mt = np.array(
        [var_t_joint_crlb_multi_time(float(t), d, D, b, Neff, time_multipliers) for t in t_grid],
        dtype=float,
    )
    pen_multi_time = var_joint_mt / var_known_mt

    # Non-identifiable single-time + priors (posterior-CRLB)
    var_known_st = np.array(
        # known-nuisance single-time benchmark: Var >= s^2/(2 d D^2 Neff)
        [(s_variance(float(t), D, b) ** 2) / (2.0 * d * (D**2) * Neff) for t in t_grid],
        dtype=float,
    )
    var_post_st = np.array(
        [
            var_t_joint_posterior_single_time(
                float(t), d, D, b, Neff, sigma_D_prior=sigma_D_prior, sigma_b_prior=sigma_b_prior
            )
            for t in t_grid
        ],
        dtype=float,
    )
    pen_single_time_priors = var_post_st / var_known_st

    # Heuristic proxy (as a visual guide only)
    pen_proxy = 1.0 + 1.0 / np.maximum(r, 1e-300)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(6.2, 4.2))

    plt.loglog(
        r,
        pen_multi_time,
        "-",
        label=r"Identifiable multi-time joint CRLB (t,2t,3t): $\mathrm{Var}_{\rm joint}/\mathrm{Var}_{\rm known}$",
    )

    plt.loglog(
        r,
        pen_single_time_priors,
        "--",
        linewidth=1.2,
        label=(
            r"Single-time + calibration priors (posterior-CRLB), "
            r"$\sigma_D/D=5\%$, $\sigma_b/b=10\%$"
        ),
    )

    plt.loglog(
        r,
        pen_proxy,
        ":",
        linewidth=1.2,
        label=r"Proxy $1+1/r$ (heuristic guide only)",
    )

    plt.axvline(1.0, linestyle="--", linewidth=1.0, label=r"$2Dt = \sigma_0^2$")

    plt.xlabel(r"Regime parameter $r = (2Dt)/\sigma_0^2$")
    plt.ylabel(r"Inflation factor $\mathrm{Var}_{\mathrm{joint}} / \mathrm{Var}_{\mathrm{known}}$")
    plt.title("Joint estimation penalty for time inference (normal diffusion)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Console notes (strict)
    # -----------------------------
    print(f"[OK] Figure saved to {OUTPUT_FIG}")
    print("[INFO] Curve 1 (solid): identifiable multi-time frequentist CRLB (no priors). Scaling ~1/Neff preserved.")
    print("[INFO] Curve 2 (dashed): single-time is non-identifiable; dashed curve is posterior-CRLB with explicit priors.")
    print("[INFO] IMPORTANT: In single-time, increasing Neff does not resolve (t,D,b) degeneracy; priors can cause prior-limited behavior.")


if __name__ == "__main__":
    main()
