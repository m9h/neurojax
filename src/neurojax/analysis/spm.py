"""
SPM-style Preprocessing Functions.
Based on standard practices in SPM/DCM for EEG/MEG.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=(0, 1))
def _dct_basis_kernel(n_time: int, n_components: int) -> jnp.ndarray:
    n = jnp.arange(n_time)
    k = jnp.arange(n_components)
    arg = jnp.outer(2*n + 1, k) * jnp.pi / (2 * n_time)
    basis = jnp.cos(arg)
    norm_0 = 1.0 / jnp.sqrt(n_time)
    norm_k = jnp.sqrt(2.0 / n_time)
    norms = jnp.where(k == 0, norm_0, norm_k)
    return basis * norms[None, :]

@jit
def _dct_residual(data: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    # weights: (C, K)
    # basis: (T, K)
    weights = jnp.dot(data, basis)
    trend = jnp.dot(weights, basis.T)
    return data - trend

def dct_filter(data: jnp.ndarray, sfreq: float, cutoff_freq: float) -> jnp.ndarray:
    """
    Apply DCT High-pass filter (Drift Removal) via regression.
    Runs in Python (determines k), calls JIT kernel.
    """
    n_chan, n_time = data.shape
    duration = n_time / sfreq
    
    # Calculate order k in Python
    k_max = int(2 * duration * cutoff_freq)
    k_max = max(1, k_max) 
    n_components = k_max + 1
    
    # Generate Basis
    # Note: For JIT kernel, n_time and n_components must be static if we want to trace 
    # but here we just call JIT function with concrete args usually.
    # If using within another JIT, 'data' shape determines n_time.
    # But n_components is derived.
    # We'll rely on JIT recompilation for different K sizes (minimal overhead).
    
    X0 = _dct_basis_kernel(n_time, n_components)
    
    return _dct_residual(data, X0)

def spm_svd(data: jnp.ndarray, n_modes: int = None, var_explained: float = None) -> jnp.ndarray:
    """
    Truncated SVD for spatial reduction (SPM-style).
    Dynamic shape output (not JIT-able directly).
    """
    # 1. SVD (JIT-able)
    U, S, Vt = jnp.linalg.svd(data, full_matrices=False)
    
    # 2. Determine k (Python or JAX)
    # Determine k values
    S_vals = S # triggers sync if S is on GPU, but required for dynamic shape output
    
    cum_var = jnp.cumsum(S_vals**2)
    total_var = cum_var[-1]
    
    k = len(S_vals)
    if var_explained is not None:
        # We need concrete values to determine Slice size in Python
        # Logic: find index.
        # This will block if S is traced.
        # But spm_svd returns variable size, so it CANT be traced/JITed as a whole.
        mask = cum_var / total_var >= var_explained
        # Use simple Numpy logic if JAX array matches
        idx = jnp.argmax(mask)
        k_var = int(idx) + 1
        
        if n_modes is None:
            k = k_var
        else:
            k = min(n_modes, k_var)
    elif n_modes is not None:
        k = n_modes
            
    # 3. Truncate
    # Slicing with concrete k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reduced Data
    # Y_reduced = diag(S) @ Vt = U.T @ Y
    data_reduced = jnp.dot(jnp.diag(S_k), Vt_k)
    
    return data_reduced, U_k, S_k
