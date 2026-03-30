"""Quantitative BOLD (qBOLD): regional OEF from multi-echo GRE.

Separates observed R2* into reversible (R2') and irreversible (R2)
components. R2' arises from deoxygenated blood vessel susceptibility
and gives per-voxel oxygen extraction fraction (OEF).

Model (He & Yablonskiy 2007):
  S(TE) = S0 * exp(-TE * R2) * F(TE, R2', DBV)

Simplified mono-exponential approximation (valid for long TE):
  S(TE) ≈ S0 * exp(-TE * (R2 + R2'))
  R2* = R2 + R2'

Conversion (Yablonskiy & Haacke 1994):
  R2' = (4/3) * pi * gamma * delta_chi * Hct * (1-Y) * DBV * B0
  OEF = R2' / (k * DBV)

References:
  He & Yablonskiy (2007) MRM 57(1):115-126
  Yablonskiy & Haacke (1994) MRM 32(6):749-763
  Bulte et al. (2012) NeuroImage 60(1):582-591
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial


# Physical constants for R2' → OEF at 3T
GAMMA = 267.522e6       # rad/s/T (proton gyromagnetic ratio)
DELTA_CHI_0 = 0.264e-6  # SI (susceptibility difference per unit Hct, fully deoxy)
B0 = 3.0                # Tesla
HCT = 0.42              # assumed hematocrit


def qbold_signal(S0: float, R2: float, R2_prime: float,
                 TEs: jnp.ndarray) -> jnp.ndarray:
    """Mono-exponential qBOLD forward model.

    S(TE) = S0 * exp(-TE * (R2 + R2'))

    Args:
        S0: proton density / equilibrium signal
        R2: irreversible transverse relaxation rate (1/s, tissue)
        R2_prime: reversible relaxation rate (1/s, from deoxy blood)
        TEs: echo times (s)

    Returns:
        Signal at each echo time
    """
    return S0 * jnp.exp(-TEs * (R2 + R2_prime))


def _qbold_loss(params, data, TEs):
    """Squared residual for qBOLD fitting."""
    S0 = jnp.abs(params[0])
    R2 = jnp.clip(params[1], 1.0, 100.0)
    R2_prime = jnp.clip(params[2], 0.0, 50.0)
    pred = qbold_signal(S0, R2, R2_prime, TEs)
    return jnp.sum((pred - data) ** 2)


@partial(jax.jit, static_argnums=(2, 3))
def qbold_fit_voxel(data: jnp.ndarray, TEs: jnp.ndarray,
                     n_iters: int = 300, lr: float = 5e-3) -> dict:
    """Fit qBOLD model to single voxel multi-echo data.

    Two-stage: log-linear init for R2* = R2 + R2', then split via
    gradient descent with R2' constrained to [0, R2*].

    Args:
        data: signal at each echo (n_echoes,)
        TEs: echo times in seconds (n_echoes,)
        n_iters: optimisation steps
        lr: learning rate

    Returns:
        dict with S0, R2, R2_prime, R2star, rmse
    """
    # Stage 1: log-linear init for total R2*
    log_data = jnp.log(jnp.maximum(data, 1e-10))
    # Fit log(S) = log(S0) - TE * R2star
    A = jnp.column_stack([jnp.ones_like(TEs), -TEs])
    lstsq = jnp.linalg.lstsq(A, log_data, rcond=None)
    S0_init = jnp.exp(lstsq[0][0])
    R2star_init = jnp.clip(lstsq[0][1], 1.0, 200.0)

    # Stage 2: split R2* into R2 + R2' via gradient descent
    # Initialize: R2 = 0.7 * R2*, R2' = 0.3 * R2* (typical brain tissue)
    params = jnp.array([S0_init, R2star_init * 0.7, R2star_init * 0.3])

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_qbold_loss)(params, data, TEs)
        grads = jax.tree.map(lambda g: jnp.clip(g, -100, 100), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = jax.lax.scan(step, (params, opt_state), None,
                                        length=n_iters)

    S0 = jnp.abs(params[0])
    R2 = jnp.clip(params[1], 1.0, 100.0)
    R2_prime = jnp.clip(params[2], 0.0, 50.0)
    pred = qbold_signal(S0, R2, R2_prime, TEs)
    rmse = jnp.sqrt(jnp.mean((pred - data) ** 2))

    return {
        'S0': S0, 'R2': R2, 'R2_prime': R2_prime,
        'R2star': R2 + R2_prime, 'rmse': rmse,
    }


def r2prime_to_oef(R2_prime: jnp.ndarray, dbv: float = 0.03,
                    hct: float = HCT, b0: float = B0) -> jnp.ndarray:
    """Convert R2' to oxygen extraction fraction.

    R2' = (4/3) * pi * gamma * delta_chi_0 * Hct * (1-Y) * DBV * B0
    OEF = 1 - Y = R2' / ((4/3) * pi * gamma * delta_chi_0 * Hct * DBV * B0)

    Args:
        R2_prime: reversible relaxation rate (Hz)
        dbv: deoxygenated blood volume fraction (default 3%)
        hct: hematocrit (default 0.42)
        b0: field strength (Tesla)

    Returns:
        OEF (oxygen extraction fraction, 0-1)
    """
    k = (4.0 / 3.0) * jnp.pi * GAMMA * DELTA_CHI_0 * hct * dbv * b0
    oef = R2_prime / jnp.maximum(k, 1e-10)
    return jnp.clip(oef, 0.0, 1.0)


def compute_regional_cmro2(cbf: jnp.ndarray, oef: jnp.ndarray,
                            hb: float = 15.0, sao2: float = 0.98
                            ) -> jnp.ndarray:
    """Compute CMRO₂ from CBF and OEF via Fick's principle.

    CMRO₂ = CBF × OEF × CaO₂
    CaO₂ = Hb × 1.34 × SaO₂ / 100 / 0.0224  (µmol O₂ / ml blood)

    Args:
        cbf: cerebral blood flow (ml/100g/min)
        oef: oxygen extraction fraction (0-1)
        hb: hemoglobin (g/dl, default 15)
        sao2: arterial oxygen saturation (default 0.98)

    Returns:
        CMRO₂ (µmol O₂ / 100g / min)
    """
    cao2_ml = hb * 1.34 * sao2 / 100  # ml O₂ / ml blood
    cao2_umol = cao2_ml / 0.0224       # µmol O₂ / ml blood
    return cbf * oef * cao2_umol
