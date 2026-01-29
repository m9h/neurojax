
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.analysis.superlet import superlet_transform
from neurojax.analysis.timefreq import morlet_transform

def verify_superlets():
    print("--- Verifying Superlet Transform ---")
    sfreq = 1000.0
    times = jnp.arange(0, 1.0, 1/sfreq)
    
    # 1. Create Challenge Signal
    # Two close frequencies in time and freq
    # Burst 1: 30Hz at 0.3s-0.5s
    # Burst 2: 40Hz at 0.3s-0.5s (Overlap in time, distinct in freq)
    
    sig = jnp.zeros_like(times)
    
    # Gaussian Window
    win = jnp.exp(-(times - 0.4)**2 / (2 * 0.05**2))
    
    sig += win * jnp.cos(2*jnp.pi*30*times)
    sig += win * jnp.cos(2*jnp.pi*45*times) # 15Hz separation
    
    sig = sig[None, :] # (1, T)
    
    freqs = tuple(np.arange(10, 60, 1.0).tolist())
    
    # 2. Standard Morlet (Cycles=5)
    print("Computing Morlet (5 cycles)...")
    wt = morlet_transform(sig, sfreq, freqs, n_cycles_min=5.0, n_cycles_max=5.0)
    pow_mor = jnp.abs(wt[0])
    
    # 3. Superlet (Base=3, Order=10) -> cycles 3..30
    print("Computing Superlet (Order 10)...")
    pow_sup = superlet_transform(
        sig, sfreq, freqs, base_cycles=3.0, order=10
    )
    pow_sup = pow_sup[0] # (Freq, Time)
    
    # 4. Metric: Sharpness
    # Check if 30Hz and 45Hz are separable at 0.4s
    t_idx = int(0.4 * sfreq)
    
    slice_mor = pow_mor[:, t_idx]
    slice_sup = pow_sup[:, t_idx]
    
    freqs_arr = np.array(freqs)
    
    # Find peaks in spectrum at t=0.4
    print("\nSpectrum at t=0.4s:")
    
    # Normalize
    slice_mor /= jnp.max(slice_mor)
    slice_sup /= jnp.max(slice_sup)
    
    # Check dip between 30 and 45 (approx 37.5)
    idx_30 = np.argmin(np.abs(freqs_arr - 30))
    idx_45 = np.argmin(np.abs(freqs_arr - 45))
    idx_mid = np.argmin(np.abs(freqs_arr - 37.5))
    
    val_mid_mor = slice_mor[idx_mid]
    val_mid_sup = slice_sup[idx_mid]
    
    print(f"Morlet Valley (37.5Hz): {val_mid_mor:.2f}")
    print(f"Superlet Valley (37.5Hz): {val_mid_sup:.2f}")
    
    if val_mid_sup < val_mid_mor:
        print("[SUCCESS] Superlet dip is deeper (better separability).")
    else:
        print("[WARNING] Superlet did not improve separability.")

if __name__ == "__main__":
    verify_superlets()
