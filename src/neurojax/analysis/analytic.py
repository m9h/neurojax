"""Analytic signal processing for M/EEG in JAX.

The analytic signal z(t) = x(t) + i*H[x(t)] (where H is the Hilbert
transform) provides instantaneous amplitude and phase --- more
fundamental representations than the raw real signal for neuroscience.

Working in complex space from the start is information-preserving and
benefits every processing stage:

  1. **Preprocessing**: Envelope-based artifact detection, phase-preserving
     filtering, complex-valued ASR thresholds
  2. **Source imaging**: Phase-based connectivity (PLV, imaginary coherence),
     envelope correlations, DICS cross-spectral beamforming
  3. **Dynamics**: Instantaneous frequency, phase-amplitude coupling,
     complex-valued HMM states
  4. **Connectivity**: Phase synchrony measures, coherence, phase lag index
  5. **Statistics**: Circular statistics on phase, Rayleigh test

Even a simple Hilbert transform in channel space gives you:
  - |z(t)| = instantaneous amplitude (envelope)
  - angle(z(t)) = instantaneous phase
  - d(phase)/dt = instantaneous frequency

All functions are JIT-compiled and differentiable via JAX.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core: Hilbert transform and analytic signal
# ---------------------------------------------------------------------------

@jax.jit
def hilbert(x: jnp.ndarray) -> jnp.ndarray:
    """Compute the analytic signal via the Hilbert transform.

    z(t) = x(t) + i * H[x(t)]

    where H is the Hilbert transform computed via the frequency domain:
    zero negative frequencies, double positive frequencies.

    Args:
        x: (..., n_times) real-valued signal(s)

    Returns:
        (..., n_times) complex analytic signal
    """
    N = x.shape[-1]
    X = jnp.fft.fft(x, axis=-1)

    # Build the frequency-domain filter
    h = jnp.zeros(N)
    if N % 2 == 0:
        h = h.at[0].set(1.0)
        h = h.at[1:N // 2].set(2.0)
        h = h.at[N // 2].set(1.0)
    else:
        h = h.at[0].set(1.0)
        h = h.at[1:(N + 1) // 2].set(2.0)

    return jnp.fft.ifft(X * h, axis=-1)


@jax.jit
def envelope(x: jnp.ndarray) -> jnp.ndarray:
    """Instantaneous amplitude (envelope) via Hilbert transform.

    Args:
        x: (..., n_times) real signal

    Returns:
        (..., n_times) amplitude envelope (real, non-negative)
    """
    return jnp.abs(hilbert(x))


@jax.jit
def instantaneous_phase(x: jnp.ndarray) -> jnp.ndarray:
    """Instantaneous phase via Hilbert transform.

    Args:
        x: (..., n_times) real signal

    Returns:
        (..., n_times) phase in radians [-pi, pi]
    """
    return jnp.angle(hilbert(x))


@jax.jit
def instantaneous_frequency(x: jnp.ndarray,
                             sfreq: float = 1.0) -> jnp.ndarray:
    """Instantaneous frequency from phase derivative.

    Args:
        x: (..., n_times) real signal
        sfreq: sampling frequency in Hz

    Returns:
        (..., n_times) instantaneous frequency in Hz
    """
    phase = instantaneous_phase(x)
    # Unwrap phase and take derivative
    dphase = jnp.diff(phase, axis=-1)
    # Wrap to [-pi, pi]
    dphase = (dphase + jnp.pi) % (2 * jnp.pi) - jnp.pi
    freq = dphase * sfreq / (2 * jnp.pi)
    # Pad to match input length
    return jnp.concatenate([freq, freq[..., -1:]], axis=-1)


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing — envelope-based artifact detection
# ---------------------------------------------------------------------------

@jax.jit
def envelope_zscore(x: jnp.ndarray) -> jnp.ndarray:
    """Z-score of the amplitude envelope for artifact detection.

    Samples where the envelope exceeds a threshold (e.g. 3-5 std)
    indicate transient artifacts. More robust than raw amplitude
    thresholding because the envelope is smooth.

    Args:
        x: (n_channels, n_times) sensor data

    Returns:
        (n_channels, n_times) z-scored envelope
    """
    env = envelope(x)
    mu = jnp.mean(env, axis=-1, keepdims=True)
    std = jnp.std(env, axis=-1, keepdims=True)
    return (env - mu) / jnp.maximum(std, 1e-10)


def detect_artifacts(x: jnp.ndarray,
                     threshold: float = 4.0) -> jnp.ndarray:
    """Detect artifact time points using envelope z-score.

    Args:
        x: (n_channels, n_times) sensor data
        threshold: z-score threshold for artifact detection

    Returns:
        (n_times,) boolean mask (True = artifact)
    """
    z = envelope_zscore(x)
    # Artifact if ANY channel exceeds threshold
    return jnp.any(jnp.abs(z) > threshold, axis=0)


# ---------------------------------------------------------------------------
# Stage 2: Source imaging — phase-based connectivity
# ---------------------------------------------------------------------------

@jax.jit
def phase_locking_value(x: jnp.ndarray,
                        y: jnp.ndarray) -> jnp.ndarray:
    """Phase Locking Value (PLV) between two signals.

    PLV = |<exp(i * (phase_x - phase_y))>_t|

    Measures consistency of phase difference across time.
    Range: 0 (no phase coupling) to 1 (perfect phase locking).

    Args:
        x, y: (..., n_times) real signals

    Returns:
        (...,) PLV values in [0, 1]
    """
    z_x = hilbert(x)
    z_y = hilbert(y)
    phase_diff = z_x * jnp.conj(z_y)
    phase_diff = phase_diff / jnp.maximum(jnp.abs(phase_diff), 1e-10)
    return jnp.abs(jnp.mean(phase_diff, axis=-1))


@jax.jit
def plv_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """Phase Locking Value matrix between all channel pairs.

    Args:
        x: (n_channels, n_times) real signal

    Returns:
        (n_channels, n_channels) PLV matrix
    """
    z = hilbert(x)
    n_ch = z.shape[0]
    # Normalise to unit complex
    z_norm = z / jnp.maximum(jnp.abs(z), 1e-10)
    # PLV_ij = |mean(z_i * conj(z_j))|
    plv = jnp.abs(z_norm @ jnp.conj(z_norm).T / z.shape[-1])
    return plv


@jax.jit
def imaginary_plv(x: jnp.ndarray) -> jnp.ndarray:
    """Imaginary part of PLV (robust to volume conduction).

    The imaginary component of coherence/PLV is zero for
    zero-lag (volume-conducted) signals, making it a robust
    measure of true neural interaction.

    Args:
        x: (n_channels, n_times) real signal

    Returns:
        (n_channels, n_channels) imaginary PLV matrix
    """
    z = hilbert(x)
    z_norm = z / jnp.maximum(jnp.abs(z), 1e-10)
    cplv = z_norm @ jnp.conj(z_norm).T / z.shape[-1]
    return jnp.abs(jnp.imag(cplv))


@jax.jit
def envelope_correlation(x: jnp.ndarray) -> jnp.ndarray:
    """Amplitude envelope correlation matrix.

    Correlates the Hilbert envelopes across channels — the standard
    measure of MEG resting-state functional connectivity after
    source reconstruction (Brookes et al. 2011, Hipp et al. 2012).

    Args:
        x: (n_channels, n_times) real signal

    Returns:
        (n_channels, n_channels) correlation matrix
    """
    env = envelope(x)
    # Demean
    env = env - jnp.mean(env, axis=-1, keepdims=True)
    # Normalise
    norms = jnp.sqrt(jnp.sum(env ** 2, axis=-1, keepdims=True))
    env = env / jnp.maximum(norms, 1e-10)
    return env @ env.T


# ---------------------------------------------------------------------------
# Stage 3: Dynamics — phase-amplitude coupling
# ---------------------------------------------------------------------------

@jax.jit
def phase_amplitude_coupling(x_phase: jnp.ndarray,
                              x_amp: jnp.ndarray) -> jnp.ndarray:
    """Modulation index for phase-amplitude coupling (PAC).

    Measures how the amplitude of a high-frequency signal is modulated
    by the phase of a low-frequency signal (Canolty et al. 2006).

    MI = |<A_high * exp(i * phase_low)>| / <A_high>

    Args:
        x_phase: (..., n_times) low-frequency signal (phase source)
        x_amp: (..., n_times) high-frequency signal (amplitude source)

    Returns:
        (...,) modulation index in [0, 1]
    """
    phase = instantaneous_phase(x_phase)
    amp = envelope(x_amp)
    # Mean vector length
    z = amp * jnp.exp(1j * phase)
    mi = jnp.abs(jnp.mean(z, axis=-1)) / jnp.maximum(jnp.mean(amp, axis=-1), 1e-10)
    return mi


# ---------------------------------------------------------------------------
# Stage 4: Narrowband analytic signal (bandpass + Hilbert)
# ---------------------------------------------------------------------------

@jax.jit
def narrowband_analytic(x: jnp.ndarray,
                        sfreq: float,
                        fmin: float,
                        fmax: float) -> jnp.ndarray:
    """Bandpass filter + Hilbert transform in one step.

    More efficient than separate filter + Hilbert because both are
    frequency-domain operations.

    Args:
        x: (..., n_times) real signal
        sfreq: sampling frequency
        fmin, fmax: passband edges in Hz

    Returns:
        (..., n_times) complex narrowband analytic signal
    """
    N = x.shape[-1]
    X = jnp.fft.fft(x, axis=-1)
    freqs = jnp.fft.fftfreq(N, d=1.0 / sfreq)

    # Bandpass: zero outside [fmin, fmax]
    mask = (jnp.abs(freqs) >= fmin) & (jnp.abs(freqs) <= fmax)

    # Analytic: zero negative frequencies
    h = jnp.zeros(N)
    if N % 2 == 0:
        h = h.at[0].set(1.0)
        h = h.at[1:N // 2].set(2.0)
        h = h.at[N // 2].set(1.0)
    else:
        h = h.at[0].set(1.0)
        h = h.at[1:(N + 1) // 2].set(2.0)

    return jnp.fft.ifft(X * mask * h, axis=-1)


# ---------------------------------------------------------------------------
# Circular statistics on phase
# ---------------------------------------------------------------------------

@jax.jit
def circular_mean(phases: jnp.ndarray,
                  axis: int = -1) -> jnp.ndarray:
    """Circular mean of phase angles.

    Args:
        phases: array of angles in radians
        axis: axis to average over

    Returns:
        mean angle in [-pi, pi]
    """
    return jnp.angle(jnp.mean(jnp.exp(1j * phases), axis=axis))


@jax.jit
def circular_variance(phases: jnp.ndarray,
                      axis: int = -1) -> jnp.ndarray:
    """Circular variance (1 - mean resultant length).

    Range: 0 (all phases aligned) to 1 (uniform distribution).

    Args:
        phases: array of angles in radians
        axis: axis to compute over

    Returns:
        circular variance in [0, 1]
    """
    R = jnp.abs(jnp.mean(jnp.exp(1j * phases), axis=axis))
    return 1.0 - R


@jax.jit
def rayleigh_z(phases: jnp.ndarray,
               axis: int = -1) -> jnp.ndarray:
    """Rayleigh test statistic for non-uniformity of phase.

    Z = n * R^2, where R is the mean resultant length.
    Reject uniformity (i.e., significant phase locking) when Z is large.

    Args:
        phases: array of angles in radians
        axis: axis to test over

    Returns:
        Rayleigh Z statistic
    """
    n = phases.shape[axis]
    R = jnp.abs(jnp.mean(jnp.exp(1j * phases), axis=axis))
    return n * R ** 2
