"""
Superlet Transform in JAX.

Reference:
Moca et al., "Time-frequency super-resolution with superlets", Nat Commun 2021.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from neurojax.analysis.timefreq import morlet_transform

@partial(jit, static_argnames=['sfreq', 'freqs', 'base_cycles', 'order'])
def superlet_transform(
    data: jnp.ndarray,
    sfreq: float,
    freqs: tuple,
    base_cycles: float = 3.0,
    order: int = 3
):
    """
    Compute Superlet Transform (SLT).
    
    Combines Morlet wavelets with different cycle counts (c, 2c, 3c...) using
    geometric mean to achieve super-resolution.
    
    Args:
        data: (n_channels, n_times)
        freqs: Tuple of frequencies of interest.
        base_cycles: Number of cycles for the lowest order (c1).
        order: Superlet order (number of wavelets combined).
    
    Returns:
        tfr: (n_channels, n_freqs, n_times) Magnitude/Power.
    """
    
    # 1. Generate Cycle Counts
    # For multiplicative Superlet: c_i = c1 * i ? 
    # Moca 2021: "set of wavelets with cycles c_1, c_2, ... c_n"
    # where c_i = c_1 * i  (linearly increasing bandwidth constraint)
    # Yes, typically [3, 6, 9] for order 3.
    
    # We need to run morlet_transform for each cycle count.
    # Since morlet_transform takes n_cycles_min/max, we can just pass
    # a single cycle count for all freqs in one go if we modify it?
    # Or, we just loop/scan over orders. Order is small (3-5).
    
    # morlet_transform currently interpolates cycles between min/max.
    # We want fixed cycles for all freqs per order?
    # Actually, standard Morlet scales cycles. 
    # Superlets usually define cycle count per frequency or fixed?
    # "Adaptive Superlets": cycles vary with frequency.
    # Let's implement the standard: c_i = base_cycles * i
    # And we assume 'adaptive' behavior comes from the base wavelet scaling?
    # No, usually base Morlet has fixed cycles? 
    # NeuroJAX morlet_transform uses `linspace(min, max)`.
    # Let's assume we want constant cycles across freq per order for simplicity,
    # OR we scale the base_cycles linearly.
    
    # Simplest valid superlet:
    # order 1: cycles = base * 1
    # order 2: cycles = base * 2
    # ...
    
    accum_prod = None
    
    for i in range(1, order + 1):
        cycles_i = base_cycles * i
        
        # We use morlet_transform. But it expects min/max cycles.
        # We set min=max=cycles_i to enforce constant cycles across freq?
        # Or should we scale? Usually standard morlet uses constant cycles (e.g. 5 or 7).
        
        # Call morlet
        # Returns: (n_ch, n_freq, n_time) Complex
        wt = morlet_transform(
            data, 
            sfreq, 
            freqs, 
            n_cycles_min=cycles_i, 
            n_cycles_max=cycles_i,
            zero_mean=True
        )
        
        # Magnitude
        mag = jnp.abs(wt) + 1e-9 # Avoid zero
        
        if accum_prod is None:
            accum_prod = mag
        else:
            accum_prod = accum_prod * mag
            
    # Geometric Mean: (prod)^(1/order)
    geomean = jnp.power(accum_prod, 1.0 / order)
    
    return geomean
