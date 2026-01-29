
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
from neurojax.io.cmi import CMILoader
from neurojax.analysis.spm import dct_filter, spm_svd

SUBJECT_ID = "sub-NDARGU729WUR"

def demo_spm():
    print("=== SPM Routine Preprocessing Demo ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except:
        print("Data not found.")
        return
        
    print("Loading Data...")
    raw.load_data()
    # No MNE filter! We use DCT.
    
    # Get Data
    data = raw.get_data() # (C, T)
    sfreq = raw.info['sfreq']
    times = raw.times
    
    # Convert to JAX
    X = jnp.array(data)
    print(f"Data Shape: {X.shape}")
    
    # 2. DCT Filtering (High-pass / Drift Removal)
    # Cutoff 0.5 Hz
    cutoff = 0.5
    print(f"\n--- Applying DCT High-pass Filter (> {cutoff} Hz) ---")
    
    X_dct = dct_filter(X, sfreq, cutoff_freq=cutoff)
    
    # Visualize Drift Removal on one channel
    ch_idx = 0
    plt.figure(figsize=(12, 6))
    plt.plot(times[:1000], X[ch_idx, :1000] * 1e6, label="Raw", alpha=0.5)
    plt.plot(times[:1000], X_dct[ch_idx, :1000] * 1e6, label="DCT Filtered", linestyle='--')
    plt.title(f"DCT Drift Removal (Channel {raw.ch_names[ch_idx]})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.legend()
    plt.savefig("spm_dct.png")
    print("Saved 'spm_dct.png'")
    
    # 3. SPM SVD (Data Reduction)
    print("\n--- Applying Truncated SVD ---")
    # Explain 99% Variance returned K=1 (Artifact?). 
    # Let's force K=8 (Typical DCM reduction)
    n_keep = 8
    
    X_reduced, U_k, S_k = spm_svd(X_dct, n_modes=n_keep)
    
    n_modes = len(S_k)
    var_retained = jnp.sum(S_k**2) / jnp.sum(jnp.linalg.svd(X_dct, compute_uv=False)**2)
    
    print(f"Original Channels: {X.shape[0]}")
    print(f"Reduced Modes: {n_modes} (retaining {var_retained*100:.2f}% variance)")
    
    # Visualize Eigenvalues (Scree Plot)
    plt.figure(figsize=(8, 4))
    plt.semilogy(S_k**2, 'o-')
    plt.title("SVD Scree Plot (Eigenvalues)")
    plt.xlabel("Mode Index")
    plt.ylabel("Variance (Log)")
    plt.grid(True)
    plt.savefig("spm_scree.png")
    print("Saved 'spm_scree.png'")
    
    # Reconstruct Data from Modes to check fidelity
    # X_recon = U_k @ X_reduced
    X_recon = jnp.dot(U_k, X_reduced)
    
    mse = jnp.mean((X_dct - X_recon)**2)
    print(f"Reconstruction MSE: {mse:.2e}")
    
    # Correlation of channel 0
    corr = jnp.corrcoef(X_dct[ch_idx], X_recon[ch_idx])[0, 1]
    print(f"Reconstruction Fidelity (Corr): {corr:.4f}")
    
    if corr > 0.99:
        print("[SUCCESS] SVD Reduction maintains high fidelity.")

if __name__ == "__main__":
    demo_spm()
