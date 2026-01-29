
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
from neurojax.io.cmi import CMILoader
from neurojax.analysis.ica import PICA
from neurojax.analysis.dimensionality import PPCA

SUBJECT_ID = "sub-NDARGU729WUR"

def demo_pica():
    print("=== PICA & SSVEP Component Selection Demo ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except:
        print("Data not found.")
        return
        
    # Preprocess
    raw.load_data()
    raw.filter(1, 90, verbose=False) 
    
    # 2. Extract ROI Data (Occipital) or Whole Brain?
    # ICA works best with more sensors. Use all 129.
    data = raw.get_data() # (N_chan, N_time)
    sfreq = raw.info['sfreq']
    
    # Downsample for speed? 500Hz is fine.
    # Convert to JAX
    X = jnp.array(data)
    
    print(f"Data Shape: {X.shape}")
    
    # 3. RYTHMIC Dimensionality Estimation
    print("\n--- RYTHMIC Dimensionality Estimation ---")
    # Using 'bic' or 'laplace' (approx)
    k_est = PPCA.estimate_dimensionality(X, method='bic')
    print(f"Estimated Dimensionality (BIC/Laplace): k = {k_est}")
    
    # 4. PICA Decomposition
    print(f"\n--- Running PICA (k={k_est}) ---")
    # PICA will PCA reduce to k, then FastICA
    pica = PICA(n_components=k_est)
    pica.fit(X)
    
    # Components S: (k, time)
    # Mixing A: (channels, k)
    S = pica.components_
    A = pica.mixing_
    
    print(f"Components Shape: {S.shape}")
    
    # 5. Component Selection (SSVEP at 30Hz)
    target_freq = 30.0
    print(f"\nSearching for SSVEP components at {target_freq} Hz...")
    
    # Compute Power Ratio for each component
    best_idx, ratio = pica.find_spectral_peak_component(sfreq, target_freq, f_width=2.0)
    best_ratio = ratio[best_idx]
    
    print(f"Best Component: IC {best_idx} (Power Ratio: {best_ratio:.4f})")
    
    # Print top 3 candidates
    top_3 = jnp.argsort(ratio)[::-1][:3]
    for idx in top_3:
        print(f" IC {idx}: Ratio {ratio[idx]:.4f}")
        
    # Check Topography of Best Component
    # Project mixing map to scalp?
    # We can just print max channel
    map_best = A[:, best_idx]
    ch_max_idx = jnp.argmax(jnp.abs(map_best))
    ch_max_name = raw.ch_names[ch_max_idx]
    print(f"Topography Peak: {ch_max_name}")
    
    if "O" in ch_max_name or "I" in ch_max_name or "z" in ch_max_name or "E75" in ch_max_name: # E75 is Oz
         print(" -> Matches Occipital VEP location!")
         
    # 6. Reconstruction
    # X_recon = A[:, idx] @ S[idx, :]
    # Reconstruct using ONLY the best component
    
    # Need to reshape S[best] to (1, T) and A[:, best] to (C, 1)
    S_sel = S[best_idx:best_idx+1, :]
    A_sel = A[:, best_idx:best_idx+1]
    
    X_clean = jnp.dot(A_sel, S_sel)
    
    # Add Mean back? PICA centered it.
    X_clean = X_clean + pica.mean_
    
    # 7. Validation: Correlation with Stimulus?
    # Or just check if 30Hz is clearer
    
    # Let's compare Spec of Raw vs Recon at Peak Channel
    print("\n--- Validation ---")
    from neurojax.analysis.filtering import filter_fft # Just import std/fft utils if needed
    
    # Raw Spec
    raw_ch = data[ch_max_idx]
    recon_ch = X_clean[ch_max_idx]
    
    # Simple PSD check at 30Hz
    spec_raw = jnp.abs(jnp.fft.rfft(raw_ch))
    spec_recon = jnp.abs(jnp.fft.rfft(recon_ch))
    freqs = jnp.fft.rfftfreq(len(raw_ch), d=1/sfreq)
    
    idx_30 = jnp.argmin(jnp.abs(freqs - 30.0))
    # Normalize by total power?
    p_raw_30 = spec_raw[idx_30] / jnp.sum(spec_raw)
    p_recon_30 = spec_recon[idx_30] / jnp.sum(spec_recon)
    
    print(f"30Hz Power Ratio (Raw): {p_raw_30:.6f}")
    print(f"30Hz Power Ratio (Recon): {p_recon_30:.6f}")
    print(f"Enhancement: {p_recon_30/p_raw_30:.2f}x")
    
    if p_recon_30 > p_raw_30 * 2.0:
        print("[SUCCESS] SSVEP component successfully isolated and enhanced.")

if __name__ == "__main__":
    demo_pica()
