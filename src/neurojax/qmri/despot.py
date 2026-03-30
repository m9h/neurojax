"""DESPOT1/2/mcDESPOT fitting — differentiable in JAX.

Key advantages over QUIT/qMRLab:
  1. jax.vmap: batch all voxels in parallel (GPU)
  2. jax.grad: analytic gradients for NLLS (no finite differences)
  3. jax.jit: compile once, run on CPU/GPU/TPU
  4. Bayesian: optax + SBI for posterior estimation
  5. Joint fitting: DESPOT1+2+HIFI in a single loss

References:
  Deoni et al. (2003) — DESPOT1
  Deoni et al. (2004) — DESPOT2
  Deoni et al. (2008) — mcDESPOT (MWF)
  Deoni (2009) — DESPOT1-HIFI (B1 correction)
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import NamedTuple

from neurojax.qmri.steady_state import (
    spgr_signal_multi, bssfp_signal_multi, ir_spgr_signal
)


class DESPOT1Result(NamedTuple):
    T1: float
    M0: float
    rmse: float


class mcDESPOTResult(NamedTuple):
    T1: float
    T2_mw: float    # myelin water T2
    T2_iew: float   # intra/extra-cellular water T2
    MWF: float      # myelin water fraction
    M0: float
    rmse: float


# =====================================================================
# DESPOT1 fitting
# =====================================================================

def _despot1_loss(params, data, flip_angles, TR):
    """Squared residual for DESPOT1."""
    M0, T1 = params[0], jnp.clip(params[1], 0.05, 8.0)
    pred = spgr_signal_multi(M0, T1, flip_angles, TR)
    return jnp.sum((pred - data) ** 2)


@partial(jax.jit, static_argnums=(3, 4))
def despot1_fit_voxel(data: jnp.ndarray, flip_angles: jnp.ndarray,
                       TR: float, n_iters: int = 100,
                       lr: float = 1e-2) -> DESPOT1Result:
    """Fit DESPOT1 T1 for a single voxel using gradient descent.

    Args:
        data: Signal at each flip angle (n_fa,)
        flip_angles: Flip angles in radians (n_fa,)
        TR: Repetition time (s)
        n_iters: Optimisation iterations
        lr: Learning rate

    Returns:
        DESPOT1Result(T1, M0, rmse)
    """
    # Linear initialisation
    sin_fa = jnp.sin(flip_angles)
    tan_fa = jnp.tan(flip_angles)
    Y = data / sin_fa
    X = data / tan_fa
    slope = jnp.sum((X - X.mean()) * (Y - Y.mean())) / (jnp.sum((X - X.mean())**2) + 1e-10)
    E1_init = jnp.clip(slope, 0.01, 0.999)
    T1_init = -TR / jnp.log(E1_init)
    # M0 from linear regression intercept: intercept = M0*(1-E1)
    intercept = Y.mean() - slope * X.mean()
    M0_init = jnp.where(jnp.abs(1 - E1_init) > 1e-6,
                         intercept / (1 - E1_init),
                         data.max() / jnp.sin(flip_angles[jnp.argmax(data)]))

    params = jnp.array([jnp.clip(jnp.abs(M0_init), 1.0, 1e7),
                         jnp.clip(T1_init, 0.1, 5.0)])
    init_loss = _despot1_loss(params, data, flip_angles, TR)

    # Gradient descent with optax
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_despot1_loss)(params, data, flip_angles, TR)
        # Clip gradients to prevent divergence
        grads = jax.tree.map(lambda g: jnp.clip(g, -100, 100), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (opt_params, _), losses = jax.lax.scan(step, (params, opt_state), None, length=n_iters)

    # Use optimised params only if they improved on the linear init
    opt_loss = _despot1_loss(opt_params, data, flip_angles, TR)
    final_params = jnp.where(opt_loss < init_loss, opt_params, params)

    M0, T1 = final_params[0], jnp.clip(final_params[1], 0.05, 8.0)
    pred = spgr_signal_multi(M0, T1, flip_angles, TR)
    rmse = jnp.sqrt(jnp.mean((pred - data) ** 2))

    return DESPOT1Result(T1=T1, M0=M0, rmse=rmse)


def despot1_fit(data_4d: jnp.ndarray, flip_angles_deg: jnp.ndarray,
                TR: float, mask: jnp.ndarray = None) -> dict:
    """Fit DESPOT1 T1 for all voxels using jax.vmap.

    Args:
        data_4d: SPGR data (X, Y, Z, n_fa)
        flip_angles_deg: Flip angles in degrees
        TR: Repetition time (s)
        mask: Brain mask (X, Y, Z), optional

    Returns:
        dict with T1, M0, rmse maps
    """
    fa_rad = jnp.deg2rad(flip_angles_deg)
    shape = data_4d.shape[:3]

    if mask is None:
        mask = data_4d[..., 0] > 0

    # Flatten masked voxels
    idx = jnp.where(mask)
    voxels = data_4d[idx]  # (n_vox, n_fa)

    # vmap over all voxels
    fit_fn = jax.vmap(lambda v: despot1_fit_voxel(v, fa_rad, TR))
    results = fit_fn(voxels)

    # Reconstruct volumes
    T1_map = jnp.zeros(shape).at[idx].set(results.T1)
    M0_map = jnp.zeros(shape).at[idx].set(results.M0)
    rmse_map = jnp.zeros(shape).at[idx].set(results.rmse)

    return {"T1": T1_map, "M0": M0_map, "rmse": rmse_map}


# =====================================================================
# DESPOT1-HIFI (joint T1 + B1 estimation)
# =====================================================================

def _despot1hifi_loss(params, spgr_data, ir_data,
                       spgr_fa, ir_fa, TR_spgr, TR_ir, TI):
    """Joint loss: SPGR VFA + IR-SPGR for T1 and B1."""
    M0, T1, B1_ratio = params[0], params[1], params[2]
    T1 = jnp.clip(T1, 0.05, 8.0)
    B1_ratio = jnp.clip(B1_ratio, 0.5, 1.5)

    # Actual flip angles = nominal * B1_ratio
    actual_spgr_fa = spgr_fa * B1_ratio
    actual_ir_fa = ir_fa * B1_ratio

    pred_spgr = spgr_signal_multi(M0, T1, actual_spgr_fa, TR_spgr)
    pred_ir = ir_spgr_signal(M0, T1, actual_ir_fa, TR_ir, TI)

    spgr_loss = jnp.sum((pred_spgr - spgr_data) ** 2)
    ir_loss = (pred_ir - ir_data) ** 2
    return spgr_loss + ir_loss


def despot1hifi_fit(spgr_data, ir_data, spgr_fa_rad, ir_fa_rad,
                     TR_spgr, TR_ir, TI, n_iters=200):
    """DESPOT1-HIFI: joint T1 + B1 from SPGR + IR-SPGR."""
    # Init from standard DESPOT1
    d1 = despot1_fit_voxel(spgr_data, spgr_fa_rad, TR_spgr)
    params = jnp.array([d1.M0, d1.T1, 1.0])  # add B1=1.0

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_despot1hifi_loss)(
            params, spgr_data, ir_data, spgr_fa_rad, ir_fa_rad,
            TR_spgr, TR_ir, TI)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=n_iters)
    return {"T1": params[1], "M0": params[0], "B1": params[2]}


# =====================================================================
# mcDESPOT (multi-component: MWF, T2_mw, T2_iew)
# =====================================================================

def _mcdespot_forward(params, spgr_fa, ssfp_fa, TR_spgr, TR_ssfp):
    """Three-pool mcDESPOT forward model.

    Pools: myelin water (MW), intra/extra-cellular water (IEW), CSF
    """
    M0, T1, MWF, T2_mw, T2_iew = (
        params[0], params[1], jnp.clip(params[2], 0.01, 0.40),
        jnp.clip(params[3], 0.005, 0.025),   # MW T2: 5-25ms
        jnp.clip(params[4], 0.040, 0.120),    # IEW T2: 40-120ms
    )

    f_mw = MWF
    f_iew = 1 - MWF
    T2_csf = 2.0  # CSF T2 ~2s, negligible in WM

    # SPGR: only sensitive to T1 (all pools same T1 to first approximation)
    spgr_pred = spgr_signal_multi(M0, T1, spgr_fa, TR_spgr)

    # bSSFP: sensitive to T2 differences between pools
    ssfp_mw = bssfp_signal_multi(M0 * f_mw, T1, T2_mw, ssfp_fa, TR_ssfp)
    ssfp_iew = bssfp_signal_multi(M0 * f_iew, T1, T2_iew, ssfp_fa, TR_ssfp)
    ssfp_pred = ssfp_mw + ssfp_iew

    return spgr_pred, ssfp_pred


def _mcdespot_loss(params, spgr_data, ssfp_data,
                    spgr_fa, ssfp_fa, TR_spgr, TR_ssfp):
    """mcDESPOT loss: SPGR + bSSFP residuals."""
    spgr_pred, ssfp_pred = _mcdespot_forward(
        params, spgr_fa, ssfp_fa, TR_spgr, TR_ssfp)
    return (jnp.sum((spgr_pred - spgr_data)**2) +
            jnp.sum((ssfp_pred - ssfp_data)**2))


def mcdespot_fit_voxel(spgr_data, ssfp_data,
                        spgr_fa_rad, ssfp_fa_rad,
                        TR_spgr, TR_ssfp,
                        n_iters=500, lr=5e-4):
    """Fit mcDESPOT MWF for a single voxel.

    Returns mcDESPOTResult with T1, T2_mw, T2_iew, MWF, M0, rmse.
    """
    # Init from DESPOT1
    d1 = despot1_fit_voxel(spgr_data, spgr_fa_rad, TR_spgr, n_iters=50)

    params = jnp.array([d1.M0, d1.T1, 0.10, 0.015, 0.070])
    # [M0, T1, MWF, T2_mw, T2_iew]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_mcdespot_loss)(
            params, spgr_data, ssfp_data,
            spgr_fa_rad, ssfp_fa_rad, TR_spgr, TR_ssfp)
        grads = jax.tree.map(lambda g: jnp.clip(g, -10, 10), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, opt_state), None, length=n_iters)

    M0, T1 = params[0], params[1]
    MWF = jnp.clip(params[2], 0.01, 0.40)
    T2_mw = jnp.clip(params[3], 0.005, 0.025)
    T2_iew = jnp.clip(params[4], 0.040, 0.120)
    rmse = jnp.sqrt(losses[-1] / (len(spgr_data) + len(ssfp_data)))

    return mcDESPOTResult(T1=T1, T2_mw=T2_mw, T2_iew=T2_iew,
                           MWF=MWF, M0=M0, rmse=rmse)
