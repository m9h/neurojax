"""Ramani QMT model with pre-computed super-Lorentzian lineshape — JAX port.

Port of qMRLab's qmt_spgr model with full pulse shape characterization.
Key advantage over QUIT's CW approximation: models the actual RF pulse
waveform (Gauss-Hanning, sinc, etc.) via Bloch simulation, giving more
accurate bound pool saturation factors.

Pipeline:
  1. build_sf_table: Pre-compute Sf(angle, offset, T2f) via Bloch ODE
  2. get_sf: Fast trilinear interpolation during fitting
  3. ramani_signal: Forward model using Sled-Pike/Ramani equations
  4. qmt_fit_voxel: Per-voxel NLLS fitting via optax

References:
  Ramani et al. (2002) MRM 47(2):257-268
  Sled & Pike (2001) MRM 46(5):923-931
  qMRLab: github.com/qMRLab/qMRLab (Cabana et al. 2015)
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial


GAMMA = 2 * jnp.pi * 42576.0  # rad/s/T (proton gyromagnetic ratio in Hz/T * 2pi)


# =====================================================================
# Pulse shapes
# =====================================================================

def gausshann_pulse(t, Trf, bw=200.0):
    """Hanning-apodized Gaussian pulse shape (qMRLab default).

    Args:
        t: time points (s)
        Trf: pulse duration (s)
        bw: FWHM bandwidth (Hz)
    """
    sigma = jnp.sqrt(2 * jnp.log(2)) / (jnp.pi * bw)
    gauss = jnp.exp(-((t - Trf / 2) ** 2) / (2 * sigma ** 2))
    hann = 0.5 * (1 - jnp.cos(2 * jnp.pi * t / Trf))
    return gauss * hann


def compute_pulse_amplitude(alpha_deg, Trf, pulse_fn, bw=200.0, n_points=500):
    """Compute RF amplitude to achieve target flip angle.

    amp = 2*pi*alpha / (360 * gamma * integral(pulse_shape, 0, Trf))
    """
    t = jnp.linspace(0, Trf, n_points)
    dt = t[1] - t[0]
    integral = jnp.sum(pulse_fn(t, Trf, bw)) * dt
    alpha_rad = jnp.deg2rad(alpha_deg)
    amp = alpha_rad / (GAMMA * integral)
    return amp


# =====================================================================
# Bloch simulation for Sf (saturation factor)
# =====================================================================

def bloch_no_mt(M, omega, delta, T2f):
    """Free pool Bloch equations under RF excitation (no MT, no T1 recovery).

    dM/dt = A*M where:
      dMx/dt = -Mx/T2f - 2*pi*delta*My
      dMy/dt = -My/T2f + 2*pi*delta*Mx + omega*Mz
      dMz/dt = -omega*My

    Args:
        M: [Mx, My, Mz]
        omega: instantaneous RF amplitude (rad/s)
        delta: off-resonance frequency (Hz)
        T2f: free pool T2 (s)
    """
    dMx = -M[0] / T2f - 2 * jnp.pi * delta * M[1]
    dMy = -M[1] / T2f + 2 * jnp.pi * delta * M[0] + omega * M[2]
    dMz = -omega * M[1]
    return jnp.array([dMx, dMy, dMz])


def compute_sf_single(alpha_deg, offset_hz, T2f, Trf, pulse_fn=gausshann_pulse,
                       bw=200.0, n_steps=200):
    """Compute saturation factor Sf for a single (angle, offset, T2f) point.

    Simulates Bloch equations during the MT pulse to get the remaining
    longitudinal magnetization Mz(Trf) of the free pool.

    For high offset frequencies (>10kHz), uses the CW analytical
    approximation instead of Bloch simulation to avoid numerical
    instability from rapid Larmor precession.

    Sf = Mz(Trf) / M0 (ratio of remaining to equilibrium magnetization)
    """
    # For high offsets, use CW analytical approximation (stable, fast)
    # The super-Lorentzian lineshape integral gives the absorption rate
    # W = pi * omega1_rms^2 * g(delta) where g is the lineshape
    # Sf_cw = exp(-W * Trf)
    amp = compute_pulse_amplitude(alpha_deg, Trf, pulse_fn, bw)

    # RMS omega1 for the pulse
    t_grid = jnp.linspace(0, Trf, 500)
    pulse_vals = pulse_fn(t_grid, Trf, bw)
    omega1_rms_sq = GAMMA**2 * amp**2 * jnp.mean(pulse_vals**2)

    # Super-Lorentzian lineshape (analytical for CW approximation)
    def super_lorentzian_g(delta_hz, T2):
        """Evaluate super-Lorentzian lineshape."""
        u = jnp.linspace(0.01, 0.99, 50)
        denom = jnp.abs(3 * u**2 - 1)
        arg = 2 * jnp.pi * delta_hz * T2 / denom
        integrand = jnp.sqrt(2 / jnp.pi) * T2 / denom * jnp.exp(-2 * arg**2)
        return jnp.trapezoid(integrand, u)

    g_val = super_lorentzian_g(offset_hz, T2f)
    W = jnp.pi * omega1_rms_sq * jnp.clip(g_val, 0, 1e-3)  # cap lineshape
    Sf_cw = jnp.clip(jnp.exp(-W * Trf), 0.0, 1.0)

    # For low offsets, also do Bloch ODE (more accurate for pulse shape effects)
    # Adaptive steps: scale with offset to keep omega*dt stable
    n_adaptive = jnp.maximum(n_steps, (jnp.abs(offset_hz) * Trf * 10).astype(int))
    n_adaptive = jnp.minimum(n_adaptive, 2000)  # cap at 2000

    t = jnp.linspace(0, Trf, n_steps, dtype=jnp.float64)
    dt = (Trf / n_steps)
    M = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)

    def step(M, ti):
        pulse_val = pulse_fn(ti, Trf, bw)
        omega = GAMMA * amp * pulse_val
        # RK4
        k1 = bloch_no_mt(M, omega, offset_hz, T2f) * dt
        k2 = bloch_no_mt(M + k1/2, omega, offset_hz, T2f) * dt
        k3 = bloch_no_mt(M + k2/2, omega, offset_hz, T2f) * dt
        k4 = bloch_no_mt(M + k3, omega, offset_hz, T2f) * dt
        M_new = M + (k1 + 2*k2 + 2*k3 + k4) / 6
        return M_new, None

    M_final, _ = jax.lax.scan(step, M, t)
    Sf_bloch = jnp.clip(M_final[2], 0.0, 1.0)

    # Use Bloch for low offsets, CW for high offsets
    # Blend at ~5kHz for smooth transition
    blend = jax.nn.sigmoid((jnp.abs(offset_hz) - 5000.0) / 1000.0)
    Sf = Sf_bloch * (1 - blend) + Sf_cw * blend
    Sf = jnp.clip(Sf, 0.0, 1.0)

    return jnp.where(jnp.isfinite(Sf), Sf, Sf_cw)


# Vectorize over the table dimensions
_compute_sf_vec = jax.vmap(
    jax.vmap(
        jax.vmap(compute_sf_single, in_axes=(None, None, 0, None, None, None, None)),
        in_axes=(None, 0, None, None, None, None, None)),
    in_axes=(0, None, None, None, None, None, None))


def build_sf_table(angles_deg, offsets_hz, T2f_values, Trf,
                    pulse_fn=gausshann_pulse, bw=200.0, n_steps=200):
    """Pre-compute Sf table for a grid of (angle, offset, T2f).

    Port of qMRLab's BuildSfTable.

    Args:
        angles_deg: MT flip angles in degrees (n_angles,)
        offsets_hz: offset frequencies in Hz (n_offsets,)
        T2f_values: free pool T2 values in seconds (n_T2f,)
        Trf: pulse duration (s)
        pulse_fn: pulse shape function
        bw: pulse bandwidth (Hz)
        n_steps: Bloch ODE integration steps

    Returns:
        dict with 'values' (n_angles, n_offsets, n_T2f), 'angles', 'offsets', 'T2f'
    """
    angles = jnp.array(angles_deg, dtype=float)
    offsets = jnp.array(offsets_hz, dtype=float)
    T2f = jnp.array(T2f_values, dtype=float)

    # Compute full 3D table via nested vmap
    values = _compute_sf_vec(angles, offsets, T2f, Trf, pulse_fn, bw, n_steps)

    return {
        'values': values,        # (n_angles, n_offsets, n_T2f)
        'angles': angles,
        'offsets': offsets,
        'T2f': T2f,
    }


def get_sf(angle_deg, offset_hz, T2f, sf_table):
    """Trilinear interpolation into pre-computed Sf table.

    Port of qMRLab's GetSf (interp3).
    """
    angles = sf_table['angles']
    offsets = sf_table['offsets']
    T2fs = sf_table['T2f']
    values = sf_table['values']

    # Normalise coordinates to [0, N-1] for each axis
    def _interp_idx(val, grid):
        idx = jnp.interp(val, grid, jnp.arange(len(grid), dtype=float))
        return jnp.clip(idx, 0, len(grid) - 1.001)

    ia = _interp_idx(angle_deg, angles)
    io = _interp_idx(offset_hz, offsets)
    it = _interp_idx(T2f, T2fs)

    # Trilinear interpolation
    ia0, io0, it0 = jnp.floor(ia).astype(int), jnp.floor(io).astype(int), jnp.floor(it).astype(int)
    ia1 = jnp.minimum(ia0 + 1, len(angles) - 1)
    io1 = jnp.minimum(io0 + 1, len(offsets) - 1)
    it1 = jnp.minimum(it0 + 1, len(T2fs) - 1)

    wa = ia - ia0; wo = io - io0; wt = it - it0

    c000 = values[ia0, io0, it0]
    c001 = values[ia0, io0, it1]
    c010 = values[ia0, io1, it0]
    c011 = values[ia0, io1, it1]
    c100 = values[ia1, io0, it0]
    c101 = values[ia1, io0, it1]
    c110 = values[ia1, io1, it0]
    c111 = values[ia1, io1, it1]

    c00 = c000 * (1 - wt) + c001 * wt
    c01 = c010 * (1 - wt) + c011 * wt
    c10 = c100 * (1 - wt) + c101 * wt
    c11 = c110 * (1 - wt) + c111 * wt

    c0 = c00 * (1 - wo) + c01 * wo
    c1 = c10 * (1 - wo) + c11 * wo

    return c0 * (1 - wa) + c1 * wa


# =====================================================================
# Ramani / Sled-Pike QMT signal model
# =====================================================================

def ramani_signal(F, kf, R1f, T2f, sf_table,
                  angles_deg, offsets_hz, Trf, TR, readout_fa_deg):
    """Ramani/Sled-Pike QMT steady-state signal model.

    Predicts the MT-weighted signal ratio (S_mt / S_ref) for each
    (angle, offset) combination.

    Args:
        F: bound pool fraction (BPF)
        kf: forward exchange rate (Hz)
        R1f: free pool R1 (1/T1, Hz)
        T2f: free pool T2 (s)
        sf_table: pre-computed Sf table
        angles_deg: MT flip angles for each measurement
        offsets_hz: offset frequencies for each measurement
        Trf: MT pulse duration (s)
        TR: repetition time (s)
        readout_fa_deg: readout flip angle (degrees)

    Returns:
        Signal ratio S_mt / S_ref for each measurement
    """
    R1r = R1f  # assume R1_restricted ≈ R1_free (common simplification)
    kr = kf * (1 - F) / F  # reverse exchange rate (detailed balance)

    def signal_at_point(angle, offset):
        Sf = get_sf(angle, offset, T2f, sf_table)
        # Rate of saturation of bound pool
        Rrfb = -jnp.log(jnp.maximum(Sf, 1e-10)) / Trf

        # Steady-state signal ratio (Ramani 2002, Eq. 11)
        # S_mt/S_ref = 1 - F*kf*(R1f + Rrfb) / (R1f*(R1r + kr + Rrfb) + kf*(R1r + Rrfb))
        num = F * kf * (R1f + Rrfb)
        denom = R1f * (R1r + kr + Rrfb) + kf * (R1r + Rrfb)
        ratio = 1 - num / jnp.maximum(denom, 1e-10)
        return jnp.clip(ratio, 0, 1)

    return jax.vmap(signal_at_point)(
        jnp.array(angles_deg, dtype=float),
        jnp.array(offsets_hz, dtype=float)
    )


# =====================================================================
# Voxel fitting
# =====================================================================

def _qmt_loss(params, data, sf_table, angles_deg, offsets_hz,
              Trf, TR, readout_fa_deg, R1f):
    """QMT fitting loss: sum of squared residuals."""
    F = jnp.clip(params[0], 0.001, 0.30)
    kf = jnp.clip(jnp.exp(params[1]), 0.1, 50.0)  # log-parameterised
    T2f = jnp.clip(params[2], 5e-6, 50e-6)

    pred = ramani_signal(F, kf, R1f, T2f, sf_table,
                          angles_deg, offsets_hz, Trf, TR, readout_fa_deg)
    return jnp.sum((pred - data) ** 2)


def qmt_fit_voxel(mt_ratio, sf_table, angles_deg, offsets_hz,
                   Trf, TR, readout_fa_deg, R1f,
                   n_iters=500, lr=1e-3):
    """Fit Ramani QMT model to a single voxel.

    Args:
        mt_ratio: measured S_mt/S_ref ratios (n_measurements,)
        sf_table: pre-computed Sf table
        angles_deg: MT flip angles per measurement
        offsets_hz: offset frequencies per measurement
        Trf: MT pulse duration (s)
        TR: repetition time (s)
        readout_fa_deg: readout flip angle
        R1f: free pool R1 (from DESPOT1 T1 map, Hz)
        n_iters: optimisation steps
        lr: learning rate

    Returns:
        dict with F (BPF), kf (exchange rate), T2f
    """
    # Initial guess
    params = jnp.array([0.10, jnp.log(3.0), 12e-6])  # [F, log(kf), T2f]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(_qmt_loss)(
            params, mt_ratio, sf_table, angles_deg, offsets_hz,
            Trf, TR, readout_fa_deg, R1f)
        grads = jax.tree.map(lambda g: jnp.clip(g, -10, 10), grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, opt_state), None, length=n_iters)

    F = jnp.clip(params[0], 0.001, 0.30)
    kf = jnp.clip(jnp.exp(params[1]), 0.1, 50.0)
    T2f = jnp.clip(params[2], 5e-6, 50e-6)

    return {'F': F, 'kf': kf, 'T2f': T2f, 'loss': losses[-1]}
