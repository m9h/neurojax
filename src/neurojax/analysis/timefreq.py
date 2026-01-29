# © NeuroJAX developers
#
# License: BSD (3-clause)

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

@partial(jit, static_argnames=['sfreq', 'freqs', 'n_cycles_min', 'n_cycles_max', 'zero_mean'])
def morlet_transform(
    data: jnp.ndarray, 
    sfreq: float, 
    freqs: tuple, 
    n_cycles_min: float = 3.0,
    n_cycles_max: float = 7.0,
    zero_mean: bool = True
):
    """
    Compute Time-Frequency representation using Morlet Wavelets.
    
    Parameters
    ----------
    freqs : tuple of floats
        Must be a tuple to be static for JIT compilation of shapes.
    """
    freqs_arr = jnp.array(freqs)
    n_freqs = len(freqs)
    cycles = jnp.linspace(n_cycles_min, n_cycles_max, n_freqs)
    
    # Define Morlet for a single frequency
    def get_wavelet(f, n_cyc):
        sigma = n_cyc / (2.0 * jnp.pi * f)
        # 3-sigma range
        half_len = int(3.0 * sigma * sfreq)
        t = jnp.arange(-half_len, half_len + 1) / sfreq
        
        # Complex Morlet: exp(2j * pi * f * t) * exp(-t^2 / 2sigma^2)
        sine = jnp.exp(2j * jnp.pi * f * t)
        gauss = jnp.exp(- (t**2) / (2 * sigma**2))
        
        wavelet = sine * gauss
        wavelet = wavelet / jnp.linalg.norm(wavelet)
        
        if zero_mean:
             wavelet -= jnp.mean(wavelet)
             
        return wavelet, t.shape[0]

    # Calculate max length using static freqs[0] (assuming sorted min to max or checking min)
    min_f = min(freqs)
    sigma_min_f = n_cycles_min / (2.0 * jnp.pi * min_f)
    # Ensure odd length for centering
    max_len = int(3.0 * sigma_min_f * sfreq) * 2 + 1
    
    # Define time vector for the largest kernel
    t = jnp.arange(-(max_len // 2), (max_len // 2) + 1) / sfreq
    
    # Compute wavelets on the fixed grid
    # This works because Morlet decays to zero effectively
    
    def get_wavelet_fixed(f, n_cyc):
        sigma = n_cyc / (2.0 * jnp.pi * f)
        
        # Complex Morlet on fixed t grid
        sine = jnp.exp(2j * jnp.pi * f * t)
        gauss = jnp.exp(- (t**2) / (2 * sigma**2))
        
        wavelet = sine * gauss
        wavelet = wavelet / jnp.linalg.norm(wavelet)
        
        if zero_mean:
             wavelet -= jnp.mean(wavelet)
             
        return wavelet
        
    wavelets = vmap(get_wavelet_fixed)(freqs_arr, cycles) 
    # Shape (n_freqs, max_len)
    
    # Convolution
    # data: (n_channels, n_times) -> (batch, 1, n_times)
    # kernel: (n_freqs, 1, max_len) (OIW)
    # output: (batch, n_freqs, n_times)
    
    is_1d = data.ndim == 1
    if is_1d:
        data = data[None, :]
        
    n_channels, n_times = data.shape
    
    # JAX Conv
    # lhs: (N, C, L) -> (n_channels, 1, n_times)
    # rhs: (O, I, W) -> (n_freqs, 1, max_len)
    # out: (N, O, L') -> (n_channels, n_freqs, n_times)
    
    lhs = data[:, None, :].astype(jnp.complex64) # (n_ch, 1, n_times)
    rhs = wavelets[:, None, :] # (n_freqs, 1, max_len)
    
    # We want 'same' convolution
    out = jax.lax.conv_general_dilated(
        lhs, rhs, 
        window_strides=(1,), 
        padding='SAME', 
        dimension_numbers=('NCW', 'OIW', 'NCW')
    )
    
    if is_1d:
        return out[0]
    return out

from functools import partial
