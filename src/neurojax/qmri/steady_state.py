"""Steady-state MRI signal equations — differentiable in JAX.

All functions are pure JAX and support:
  - jax.grad for sensitivity analysis and fitting
  - jax.vmap for batch voxelwise evaluation
  - jax.jit for GPU compilation

References:
  Deoni et al. (2003) MRM 49:515 — DESPOT1/2
  Zur et al. (1991) MRM 21:151 — bSSFP signal
  Ramani et al. (2002) MRI 20:721 — QMT two-pool
"""

import jax
import jax.numpy as jnp


# =====================================================================
# SPGR (Spoiled Gradient Echo) — DESPOT1 / VFA T1
# =====================================================================

def spgr_signal(M0: float, T1: float, flip_angle: float, TR: float) -> float:
    """SPGR steady-state signal.

    S = M0 * sin(α) * (1 - E1) / (1 - E1*cos(α))

    Args:
        M0: Equilibrium magnetisation (a.u.)
        T1: Longitudinal relaxation time (s)
        flip_angle: Flip angle (radians)
        TR: Repetition time (s)

    Returns:
        Signal magnitude
    """
    E1 = jnp.exp(-TR / T1)
    return M0 * jnp.sin(flip_angle) * (1 - E1) / (1 - E1 * jnp.cos(flip_angle))


def spgr_signal_multi(M0: float, T1: float, flip_angles: jnp.ndarray,
                       TR: float) -> jnp.ndarray:
    """SPGR signal at multiple flip angles (vectorised)."""
    return jax.vmap(lambda fa: spgr_signal(M0, T1, fa, TR))(flip_angles)


# =====================================================================
# bSSFP (balanced Steady-State Free Precession) — DESPOT2
# =====================================================================

def bssfp_signal(M0: float, T1: float, T2: float,
                  flip_angle: float, TR: float,
                  dphi: float = 0.0) -> float:
    """bSSFP steady-state signal magnitude.

    S = M0 * sin(α) * (1 - E1) * sqrt(E2) /
        (1 - (E1 - E2)*cos(α) - E1*E2)

    with off-resonance phase dphi (0 or π for phase cycling).

    Args:
        M0: Equilibrium magnetisation
        T1: T1 (s)
        T2: T2 (s)
        flip_angle: Flip angle (radians)
        TR: Repetition time (s)
        dphi: Off-resonance phase per TR (radians)
    """
    E1 = jnp.exp(-TR / T1)
    E2 = jnp.exp(-TR / T2)
    ca = jnp.cos(flip_angle)
    sa = jnp.sin(flip_angle)

    # Freeman-Hill formula with phase cycling
    denom = 1 - (E1 - E2) * ca - E1 * E2
    num = M0 * sa * (1 - E1) * jnp.sqrt(E2)
    return jnp.abs(num / denom)


def bssfp_signal_multi(M0: float, T1: float, T2: float,
                        flip_angles: jnp.ndarray, TR: float,
                        dphi: float = 0.0) -> jnp.ndarray:
    """bSSFP signal at multiple flip angles."""
    return jax.vmap(lambda fa: bssfp_signal(M0, T1, T2, fa, TR, dphi))(flip_angles)


# =====================================================================
# IR-SPGR (Inversion Recovery SPGR) — DESPOT1-HIFI B1 correction
# =====================================================================

def ir_spgr_signal(M0: float, T1: float, flip_angle: float,
                    TR: float, TI: float,
                    efficiency: float = 1.0) -> float:
    """Inversion-recovery SPGR signal for DESPOT1-HIFI.

    S = M0 * sin(α) * |1 - (1+eff)*exp(-TI/T1) + exp(-TR/T1)|
        / (1 - exp(-TR/T1)*cos(α))

    Used with SPGR VFA data to jointly estimate T1 and B1.
    """
    E1_TR = jnp.exp(-TR / T1)
    E1_TI = jnp.exp(-TI / T1)
    sa = jnp.sin(flip_angle)
    ca = jnp.cos(flip_angle)

    num = M0 * sa * jnp.abs(1 - (1 + efficiency) * E1_TI + E1_TR)
    denom = 1 - E1_TR * ca
    return num / denom


# =====================================================================
# Multi-echo GRE — T2* mapping
# =====================================================================

def multiecho_signal(S0: float, T2star: float, TE: float) -> float:
    """Mono-exponential T2* decay: S(TE) = S0 * exp(-TE/T2*)."""
    return S0 * jnp.exp(-TE / T2star)


def multiecho_signal_multi(S0: float, T2star: float,
                            TEs: jnp.ndarray) -> jnp.ndarray:
    """Multi-echo signal at multiple echo times."""
    return S0 * jnp.exp(-TEs / T2star)


# =====================================================================
# QMT two-pool model
# =====================================================================

def super_lorentzian_lineshape(delta_f: float, T2b: float = 11e-6,
                                n_points: int = 100) -> float:
    """Super-Lorentzian lineshape for the bound macromolecular pool.

    g(Δf) = ∫₀¹ sqrt(2/π) * T2b / |3u²-1| *
             exp(-2*(2π*Δf*T2b/(3u²-1))²) du

    Args:
        delta_f: Offset frequency (Hz)
        T2b: Bound pool T2 (s), typically ~10-12 µs
        n_points: Quadrature points
    """
    u = jnp.linspace(0.001, 0.999, n_points)
    denom = jnp.abs(3 * u**2 - 1)
    arg = 2 * jnp.pi * delta_f * T2b / denom
    integrand = jnp.sqrt(2 / jnp.pi) * T2b / denom * jnp.exp(-2 * arg**2)
    return jnp.trapezoid(integrand, u)


def qmt_signal_ramani(M0_f: float, f_b: float, k_bf: float,
                       T1_f: float, T2_f: float, T2_b: float,
                       sat_angle: float, sat_freq: float,
                       Trf: float, TR: float, FA: float,
                       R1_b: float = 2.5) -> float:
    """QMT two-pool Ramani model signal.

    Predicts the MT-weighted signal given free and bound pool parameters.

    Args:
        M0_f: Free pool equilibrium magnetisation
        f_b: Bound pool fraction (BPF)
        k_bf: Exchange rate bound→free (Hz)
        T1_f, T2_f: Free pool T1, T2 (s)
        T2_b: Bound pool T2 (s)
        sat_angle: MT saturation pulse angle (radians)
        sat_freq: MT offset frequency (Hz)
        Trf: MT pulse duration (s)
        TR: Repetition time (s)
        FA: Readout flip angle (radians)
        R1_b: Bound pool R1 (Hz), default 2.5

    Returns:
        MT-weighted signal
    """
    R1_f = 1.0 / T1_f
    f_f = 1 - f_b
    k_fb = k_bf * f_b / f_f  # detailed balance

    # CW-equivalent saturation rate for the bound pool
    # W = π * ω1_rms² * g(Δf)
    omega1_rms = sat_angle / Trf  # approximate RMS B1
    g = super_lorentzian_lineshape(sat_freq, T2_b)
    W = jnp.pi * omega1_rms**2 * g

    # Steady-state free pool signal under MT saturation
    # (Ramani 2002, simplified)
    R1_obs = R1_f + k_fb
    S_sat = M0_f * f_f * jnp.sin(FA) * (1 - jnp.exp(-TR * R1_obs)) / \
            (1 - jnp.cos(FA) * jnp.exp(-TR * R1_obs))

    # MT saturation factor
    mt_factor = 1 - f_b * W / (R1_b + k_bf + W)

    return S_sat * mt_factor


# =====================================================================
# MP2RAGE
# =====================================================================

def mp2rage_lookup(T1: float, TI1: float, TI2: float,
                    TR_GRE: float, TR_MP2RAGE: float,
                    FA1: float, FA2: float,
                    n_GRE: int = 176) -> float:
    """MP2RAGE uniform image intensity as a function of T1.

    The MP2RAGE signal removes B1 inhomogeneity:
    UNI = real(TI1 * conj(TI2)) / (|TI1|² + |TI2|²)

    This provides a monotonic T1 lookup for quantitative T1 estimation.
    """
    E1_TI1 = jnp.exp(-TI1 / T1)
    E1_TI2 = jnp.exp(-TI2 / T1)
    E1_TR = jnp.exp(-TR_GRE / T1)

    # Signal at each inversion time (simplified)
    ca1 = jnp.cos(FA1)
    ca2 = jnp.cos(FA2)

    # Approximate: ignore GRE train steady-state (full model needs n_GRE)
    S1 = jnp.sin(FA1) * (1 - 2 * E1_TI1)
    S2 = jnp.sin(FA2) * (1 - 2 * E1_TI2)

    # Uniform image
    uni = S1 * S2 / (S1**2 + S2**2 + 1e-10)
    return uni
