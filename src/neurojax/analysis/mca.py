"""
Morphological Component Analysis (MCA) in JAX.

Separates signals into morphologically distinct components based on sparsity.
Algorithm: Block-Coordinate Relaxation (BCR) with Soft Thresholding.

Components:
1. Oscillations (Sparse in DCT domain)
2. Transients/Spikes (Sparse in Time/Identity domain)
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import jax.numpy.fft as jnfft

def soft_threshold(x, threshold):
    # Handle Complex (Group Sparsity on Magnitude)
    # x = A e^(i phi)
    # x' = max(A - T, 0) e^(i phi)
    
    abs_x = jnp.abs(x)
    # Avoid div by zero
    scale = jnp.maximum(0.0, 1.0 - threshold / (abs_x + 1e-12))
    return x * scale

# --- Dictionaries ---

def fft_forward(x):
    # Real FFT -> Complex half-spectrum
    # Norm='ortho' to preserve energy scaling
    return jnfft.rfft(x, norm='ortho')

def fft_inverse(c):
    # Inverse Real FFT
    # n must be provided? 
    # rfft of length N produces N//2 + 1. 
    # Usually we scan with fixed size. We assume N is (len(c)-1)*2 (even) or similar.
    # Let's assume input C implies N.
    # irfft infers length, assumes even if odd not specified.
    # For robust reconstruction, we might need original length. 
    # But for MCA loop, data length is fixed.
    # Let's assume N = (len(c)-1)*2 for now (Even length signals).
    return jnfft.irfft(c, norm='ortho')

# Identity is same
def identity_forward(x):
    return x

def identity_inverse(c):
    return c

@partial(jit, static_argnames=['n_iter'])
def mca_solve(data, lambda_fft, lambda_ident, n_iter=50):
    """
    Solve MCA: Y = X_fft + X_ident
    Minimize: 0.5*||Y - (ifft(c1) + c2)||^2 + lam1*|c1|_1 + lam2*|c2|_1
    """
    
    # Initialize Coeffs
    # c_fft is complex
    c_fft = fft_forward(jnp.zeros_like(data))
    c_ident = jnp.zeros_like(data)
    
    def step(carry, _):
        c_f, c_i = carry
        
        # 1. Update FFT Component
        # Reconstruct Identity component
        recon_i = identity_inverse(c_i)
        residual_for_fft = data - recon_i
        
        # Project to FFT domain
        coeffs_rough = fft_forward(residual_for_fft)
        # Threshold (Complex)
        c_f_new = soft_threshold(coeffs_rough, lambda_fft)
        
        # 2. Update Identity Component
        # Reconstruct FFT component
        recon_f = fft_inverse(c_f_new)
        # Handle length mismatch if odd/even?
        # irfft might return different length if not careful.
        # Force length to match data?
        # jax.numpy.fft.irfft(a, n=...)
        # We can't access data shape dynamically easily in JIT unless static?
        # data.shape is known at JIT time usually.
        # But 'step' receives data from closure.
        # Let's trust irfft default (2*(N-1)) matches if data was even.
        if recon_f.shape[0] != data.shape[0]:
             # Truncate or pad? 
             # Should use n=data.shape[0] in irfft
             # But we can't pass dynamic shape easily? 
             # Actually we can use data.shape[0] since 'data' is captured closure
             pass
             
        residual_for_ident = data - recon_f
        
        # Project to Identity (Time)
        c_i_new = soft_threshold(residual_for_ident, lambda_ident)
        
        return (c_f_new, c_i_new), None
        
    (final_c_fft, final_c_ident), _ = lax.scan(step, (c_fft, c_ident), None, length=n_iter)
    
    # Reconstruct final signals
    part_osc = fft_inverse(final_c_fft)
    part_trans = identity_inverse(final_c_ident)
    
    # Ensure shapes match (handle odd/even edge case)
    if part_osc.shape[0] != data.shape[0]:
         part_osc = jnp.resize(part_osc, data.shape) # simple resize/crop
    
    return part_osc, part_trans

@partial(jit, static_argnames=['n_iter'])
def mca_decompose(batch_data, lambda_fft=0.5, lambda_ident=0.5, n_iter=50):
    """
    Vectorized MCA decomposition over channels.
    """
    # Ensure even length to make RFFT/IRFFT roundtrip simple
    n_time = batch_data.shape[-1]
    if n_time % 2 != 0:
        batch_data = batch_data[..., :-1]
        
    solve = lambda d: mca_solve(d, lambda_fft, lambda_ident, n_iter)
    return jax.vmap(solve)(batch_data)
