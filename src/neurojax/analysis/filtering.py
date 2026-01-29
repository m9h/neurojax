"""
Fast JAX Filtering for NeuroJAX.

Implements Highpass, Lowpass, Bandpass, and Notch filters using:
1. FFT-based filtering (Frequency Domain) - Very fast on GPU.
2. FIR Convolution (Time Domain).
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

@partial(jit, static_argnames=['sfreq', 'f_low', 'f_high'])
def filter_fft(data: jnp.ndarray, sfreq: float, f_low: float = None, f_high: float = None) -> jnp.ndarray:
    """
    Apply brick-wall filter (or smoothed) in Frequency Domain.
    Faster than convolution for long signals.
    
    Args:
        data: (Channels, Time)
        sfreq: Sampling rate.
        f_low: Low cutoff (Highpass). If None, no highpass.
        f_high: High cutoff (Lowpass). If None, no lowpass.
    """
    n_times = data.shape[-1]
    freqs = jnp.fft.rfftfreq(n_times, d=1/sfreq)
    
    # FFT
    spectra = jnp.fft.rfft(data, axis=-1)
    
    # Mask
    mask = jnp.ones_like(freqs)
    
    if f_low is not None:
        mask = jnp.where(freqs < f_low, 0.0, mask)
        
    if f_high is not None:
        mask = jnp.where(freqs > f_high, 0.0, mask)
        
    # Apply
    spectra_filtered = spectra * mask
    
    # IFFT
    return jnp.fft.irfft(spectra_filtered, n=n_times, axis=-1)

@partial(jit, static_argnames=['sfreq', 'freq', 'width'])
def notch_filter_fft(data: jnp.ndarray, sfreq: float, freq: float = 60.0, width: float = 1.0) -> jnp.ndarray:
    """
    Remove line noise (e.g. 60Hz) via FFT.
    """
    n_times = data.shape[-1]
    freqs = jnp.fft.rfftfreq(n_times, d=1/sfreq)
    
    # Soft notch or Hard notch
    # Hard brickwall notch
    f_min = freq - width/2
    f_max = freq + width/2
    
    mask = jnp.where((freqs >= f_min) & (freqs <= f_max), 0.0, 1.0)
    
    # Harmonics? (60, 120, 180...) - usually handled by caller or loop
    
    spectra = jnp.fft.rfft(data, axis=-1)
    return jnp.fft.irfft(spectra * mask, n=n_times, axis=-1)

def robust_reference(data: jnp.ndarray, max_iter: int = 3) -> jnp.ndarray:
    """
    Robust Average Reference (exclude bad channels from mean).
    Simple iterative approach:
    1. Compute mean
    2. Find channels far from mean
    3. Recompute mean without them
    4. Subtract.
    
    (Note: JAX scan/loop needed for iter, but 1-pass median might be faster)
    """
    # Median reference is robust but slow?
    # Mean is fast.
    
    # Iteration 1
    ref = jnp.mean(data, axis=0, keepdims=True)
    centered = data - ref
    
    # Estimate bads (high variance/amplitude)
    std = jnp.std(centered, axis=1) # (Channels,)
    threshold = 3.0 * jnp.median(std) # MAD-like
    
    mask = std < threshold
    
    # Recompute ref with good channels
    # careful with NaNs if all bad
    ref_robust = jnp.sum(data * mask[:, None], axis=0, keepdims=True) / jnp.sum(mask)
    
    return data - ref_robust
