
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.analysis.mca import mca_decompose

def verify_mca():
    print("--- Verifying MCA Separation (Sine vs Spikes) ---")
    
    sfreq = 1000.0
    times = jnp.arange(0, 1.0, 1/sfreq)
    
    # 1. Ground Truth
    # Oscillatory: 30Hz Sine
    gt_osc = jnp.sin(2*jnp.pi*30*times)
    
    # Transient: Sparse Spikes
    key = jax.random.PRNGKey(0)
    spikes = jax.random.bernoulli(key, p=0.01, shape=times.shape).astype(jnp.float32) * 5.0
    # Make them alternating sign?
    gt_trans = spikes * jax.random.choice(key, jnp.array([-1., 1.]), shape=times.shape)
    
    # Mix
    data = gt_osc + gt_trans
    
    # 2. MCA Decompose
    # Thresholds need tuning. 
    # Osc: DCT is very sparse for Sine. Lambda can be low?
    # Trans: Identity is sparse for Spikes.
    # Let's try lambda = 0.5 * max? or fixed?
    # Sine amp=1. Spike amp=5.
    
    print("Running MCA...")
    # Add batch dim
    data_batch = data[None, :] 
    
    # Debug Norms
    print("Checking norms...")
    spectrum_spike = jnp.fft.rfft(gt_trans, norm='ortho')
    spectrum_sine = jnp.fft.rfft(gt_osc, norm='ortho')
    print(f"Max Spike Spectrum Mag: {jnp.max(jnp.abs(spectrum_spike)):.4f}")
    print(f"Max Sine Spectrum Mag: {jnp.max(jnp.abs(spectrum_sine)):.4f}")
    
    # Run MCA
    est_osc, est_trans = mca_decompose(
        data_batch, 
        lambda_fft=0.5,    # Should be > Spike Mag
        lambda_ident=1.0,  # Should be < Spike Amp (5.0) and > Noise
        n_iter=100
    )
    
    est_osc = est_osc[0]
    est_trans = est_trans[0]
    
    # 3. Metrics
    corr_osc = jnp.corrcoef(gt_osc, est_osc)[0, 1]
    corr_trans = jnp.corrcoef(gt_trans, est_trans)[0, 1]
    
    print(f"Correlation Oscillation (Sine): {corr_osc:.4f}")
    print(f"Correlation Transient (Spikes): {corr_trans:.4f}")
    
    # Check Residual Energy
    residual = data - (est_osc + est_trans)
    mse = jnp.mean(residual**2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    if corr_osc > 0.95 and corr_trans > 0.90:
        print("[SUCCESS] MCA Successfully Separated Sine from Spikes.")
    else:
        print("[WARNING] Separation poor. Tune thresholds.")
        print(f"Amplitudes: Osc={jnp.max(jnp.abs(est_osc)):.2f}, Trans={jnp.max(jnp.abs(est_trans)):.2f}")

if __name__ == "__main__":
    verify_mca()
