"""
Rough Path Analysis for NeuroJAX.

This module uses Rough Path Theory (Signatures) to extract geometric features 
from high-frequency neural time series. Signatures are particularly effective 
at characterizing the "shape" of transient bursts (e.g. Beta/Gamma) robustly 
to noise and reparameterization.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import signax
from functools import partial

def augment_path(x: jnp.ndarray, add_time: bool = True) -> jnp.ndarray:
    """
    Augment the path to ensure uniqueness of the signature.
    
    Args:
        x (jnp.ndarray): Input path of shape (Time, Channels).
        add_time (bool): Whether to append a time channel.
        
    Returns:
        jnp.ndarray: Augmented path (Time, Channels + 1).
    """
    if not add_time:
        return x
        
    t = jnp.linspace(0, 1, x.shape[0])
    return jnp.column_stack([t, x])

@partial(jit, static_argnames=['depth'])
def compute_signature(path: jnp.ndarray, depth: int = 3) -> jnp.ndarray:
    """
    Compute the Signature of a path.
    
    Args:
        path: Input path (Time, Channels).
        depth: Truncation depth of the signature.
        
    Returns:
        jnp.ndarray: Signature terms (concatenated vector).
    """
    return signax.signature(path, depth)

@partial(jit, static_argnames=['depth'])
def compute_log_signature(path: jnp.ndarray, depth: int = 3) -> jnp.ndarray:
    """
    Compute the Log-Signature (compact representation).
    """
    return signax.logsignature(path, depth)

@partial(jit, static_argnames=['depth', 'window_size', 'stride'])
def sliding_signature(
    data: jnp.ndarray, 
    depth: int = 3, 
    window_size: int = 100, 
    stride: int = 50
) -> jnp.ndarray:
    """
    Compute signatures over a sliding window.
    
    Args:
        data: Input time series (Time, Channels).
        depth: Signature depth.
        window_size: Size of sliding window.
        stride: Step size.
        
    Returns:
        jnp.ndarray: Signatures (N_Windows, Sig_Dim).
    """
    # Create windows using JAX stride tricks or manual unfold
    # (Using simple reshaping/slicing for now, or vmap over indices)
    n_time = data.shape[0]
    n_windows = (n_time - window_size) // stride + 1
    
    starts = jnp.arange(n_windows) * stride
    
    # Extract windows: (n_windows, window_size, channels)
    # Using vmap to slice
    def get_window(start_idx):
        # Slice and Augment with time immediately?
        # Better to augment per window to reset time to [0,1]
        w = jax.lax.dynamic_slice(data, (start_idx, 0), (window_size, data.shape[1]))
        return augment_path(w, add_time=True)
        
    windows = vmap(get_window)(starts)
    
    # Compute signatures
    sigs = vmap(partial(compute_signature, depth=depth))(windows)
    
    return sigs
