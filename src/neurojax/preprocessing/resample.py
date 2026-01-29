"""JAX-native Polyphase Resampling."""
import jax
import jax.numpy as jnp
import scipy.signal
from functools import partial

@partial(jax.jit, static_argnames=['up', 'down', 'axis', 'window'])
def resample_poly(x, up, down, axis=-1, window=('kaiser', 5.0)):
    """
    Resample `x` along the given axis using polyphase filtering.
    """
    up = int(up)
    down = int(down)
    
    if up == down:
        return x
        
    # Design the FIR filter
    max_rate = max(up, down)
    f_c = 1. / max_rate  # Cutoff relative to the new Nyquist
    half_len = 10 * max_rate # Reasonable filter length
    
    num_taps = 2 * half_len + 1
    
    # Standard FIR design on host
    coefs = scipy.signal.firwin(num_taps, f_c, window=window) * up
    coefs = jnp.array(coefs, dtype=x.dtype)
    
    # Move axis to last for processing
    x_swapped = jnp.swapaxes(x, axis, -1)
    
    input_shape = x_swapped.shape
    n_samples = input_shape[-1]
    
    # Flatten all non-time dimensions into one batch dimension
    x_flat = x_swapped.reshape(-1, n_samples)
    
    # 1. Naive Upsample (Insert Zeros)
    # x_flat: (batch, time)
    # We want (batch, time*up)
    upsampled_len = n_samples * up
    x_up = jnp.zeros((x_flat.shape[0], n_samples, up), dtype=x.dtype)
    x_up = x_up.at[..., 0].set(x_flat)
    x_up = x_up.reshape(x_flat.shape[0], upsampled_len)
    
    # 2. Convolve
    # We use vmap to apply 1D convolution to each batch element
    # coefs is 1D
    
    def apply_conv(signal):
        # signal: (time,)
        # coefs: (taps,)
        return jax.scipy.signal.fftconvolve(signal, coefs, mode='same')
        
    x_filt = jax.vmap(apply_conv)(x_up)
    
    # 3. Downsample
    x_down = x_filt[..., ::down]
    
    # 4. Reshape and Swap back
    # New time length
    n_new = x_down.shape[-1]
    final_shape = input_shape[:-1] + (n_new,)
    
    x_final = x_down.reshape(final_shape)
    
    return jnp.swapaxes(x_final, axis, -1)

@partial(jax.jit, static_argnames=['original_sfreq', 'target_sfreq', 'axis'])
def resample_minimal(x, original_sfreq, target_sfreq, axis=-1):
    """
    Convenience wrapper to calculate factors.
    Uses GCD to find minimal integer factors.
    Original and Target sfreqs must be static.
    """
    # These must be concrete
    original = int(original_sfreq)
    target = int(target_sfreq)
    
    # Python stdlib math.gcd for integers (not jnp.gcd)
    import math
    common = math.gcd(original, target)
    
    up = target // common
    down = original // common
    
    return resample_poly(x, up, down, axis=axis)
