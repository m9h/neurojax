
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert
from neurojax.io.cmi import CMILoader
from neurojax.source.champagne import champagne_solver, imaginary_coherence
from neurojax.source.beamformer import apply_lcmv, make_lcmv_filter

SUBJECT_ID = "sub-NDARGU729WUR"

def demo_champagne():
    print("=== CHAMPAGNE & Imaginary Coherence Demo ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except: return
        
    raw.load_data()
    raw.filter(1, 90, verbose=False)
    
    # Forward Model 
    raw.set_eeg_reference(projection=True)
    if raw.get_montage() is None:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        raw.set_montage(montage)
    sphere = mne.make_sphere_model(r0='auto', head_radius='auto', info=raw.info)
    src = mne.setup_volume_source_space(subject=None, pos=30.0, sphere=sphere, bem=sphere)
    fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False)
    
    # Data & Covariance (SCALED to uV)
    data = raw.get_data()
    X = jnp.array(data)
    scale = 1e6
    X_scaled = X * scale
    
    X_cent = X_scaled - jnp.mean(X_scaled, axis=1, keepdims=True)
    cov = jnp.dot(X_cent, X_cent.T) / X.shape[1]
    
    G = jnp.array(fwd['sol']['data']) 
    
    # 2. Run CHAMPAGNE
    print("Running CHAMPAGNE Solver (Empirical Bayes)...")
    # Regularize noise cov (5% of trace)
    noise_cov = jnp.eye(cov.shape[0]) * (jnp.trace(cov)/cov.shape[0]) * 0.05
    gamma, weights = champagne_solver(cov, G, noise_cov=noise_cov, max_iter=50)
    
    max_gam = jnp.max(gamma)
    n_active = jnp.sum(gamma > (max_gam * 0.01))
    print(f"CHAMPAGNE Sparse Sources (>1% max): {n_active} / {len(gamma)}")
    
    # 3. Extract V1 Timecourse
    # Find V1 index 
    src_pos = fwd['source_rr']
    target_pos = np.array([0, -0.08, 0])
    dists = np.linalg.norm(src_pos - target_pos, axis=1)
    v1_idx = np.argmin(dists)
    
    print(f"V1 Gamma Power: {gamma[v1_idx*3 : v1_idx*3+3]}")
    
    # Use LCMV weights for Seeding (Robust check requested)
    # If CHAMPAGNE pruned V1, we can't use it for coherence seed.
    print("Using LCMV weights for V1 Seed (Robustness against Pruning)...")
    w_lcmv_all = make_lcmv_filter(cov, G, reg=0.05)
    w_v1_lcmv = w_lcmv_all[v1_idx*3 : v1_idx*3+3]
    
    # Project Scaled Data
    s_vec_scaled = apply_lcmv(X_scaled, w_v1_lcmv)
    
    # SVD for dominant orientation
    u, s, vt = jnp.linalg.svd(s_vec_scaled, full_matrices=False)
    s_v1 = vt[0] * s[0]
    
    print("Extracted V1 Timecourse via LCMV (for Coherence Seeding).")
    
    # 4. Imaginary Coherence Analysis
    print("Computing Analytic Signals...")
    s_v1_ana = hilbert(s_v1)
    X_ana = hilbert(X_scaled, axis=1)
    
    s_v1_jax = jnp.array(s_v1_ana)
    X_jax = jnp.array(X_ana)
    
    # Imaginary Coherence: Im( E[xy*] / sqrt(E[xx]E[yy]) )
    cross = X_jax * jnp.conj(s_v1_jax[None, :])
    csd = jnp.mean(cross, axis=1)
    p_x = jnp.mean(jnp.abs(X_jax)**2, axis=1)
    p_s = jnp.mean(jnp.abs(s_v1_jax)**2)
    
    # Avoid div/0
    coh = csd / jnp.sqrt(p_x * p_s + 1e-12)
    
    icoh_map = jnp.imag(coh)
    coh_map = jnp.abs(coh)
    
    # Max iCoh
    idx_max = jnp.argmax(jnp.abs(icoh_map))
    max_icoh = icoh_map[idx_max]
    max_ch = raw.ch_names[idx_max]
    print(f"Max Imaginary Coherence with V1: {max_icoh:.4f} at {max_ch}")
    print(f"Max Magnitude Coherence: {jnp.max(coh_map):.4f}")
    
    # 5. SSVEP SNR of V1 Source
    freqs = np.fft.rfftfreq(len(s_v1), d=1/raw.info['sfreq'])
    spec = np.abs(np.fft.rfft(s_v1))
    idx_30 = np.argmin(np.abs(freqs - 30.0))
    ratio = spec[idx_30] / np.sum(spec)
    
    print(f"V1 Source 30Hz Ratio: {ratio:.5f}")
    
    # Plot iCoh
    plt.figure(figsize=(10, 5))
    plt.plot(icoh_map, label='Imaginary Coherence')
    plt.plot(coh_map, label='Magnitude Coherence', alpha=0.5)
    plt.legend()
    plt.title("V1 Connectivity (Imaginary Coherence)")
    plt.xlabel("Channel Index")
    plt.savefig("champagne_icoh.png")
    print("Saved 'champagne_icoh.png'")

if __name__ == "__main__":
    demo_champagne()
