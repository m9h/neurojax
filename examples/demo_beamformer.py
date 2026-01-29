
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
from neurojax.io.cmi import CMILoader
from neurojax.source.beamformer import make_lcmv_filter, apply_lcmv
from neurojax.analysis.ica import PICA
from scipy.signal import correlate

SUBJECT_ID = "sub-NDARGU729WUR"

def demo_beamformer():
    print("=== Beamforming: V1 Source Reconstruction ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except:
        print("Data not found.")
        return
        
    raw.load_data()
    raw.filter(1, 90, verbose=False) # Highpass needed for covariance stability
    
    # 2. Setup Forward Model (Sphere)
    print("Setting up Spherical Forward Model...")
    raw.set_eeg_reference(projection=True)
    
    # Montage
    montage = raw.get_montage()
    if montage is None:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        raw.set_montage(montage)
        
    # Sphere Model
    sphere = mne.make_sphere_model(r0='auto', head_radius='auto', info=raw.info)
    
    # Source Space (Discrete grid)
    # Use loose spacing to be fast
    src = mne.setup_volume_source_space(subject=None, pos=30.0, sphere=sphere, bem=sphere)
    
    # Forward Solution
    fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere, eeg=True, meg=False, verbose=False)
    
    print(f"Source Space: {fwd['nsource']} dipoles.")
    
    # 3. Covariance (Data)
    data = raw.get_data()
    # Normalize? 
    # MNE Covariance usually handles noise cov.
    # We'll compute simple data covariance matrix C = XX.T / N
    X = jnp.array(data)
    X_cent = X - jnp.mean(X, axis=1, keepdims=True)
    n_samples = X.shape[1]
    cov = jnp.dot(X_cent, X_cent.T) / n_samples
    
    # 4. Filter Weights (LCMV)
    # Gain Matrix
    G = fwd['sol']['data'] # (n_chan, n_src * 3) usually (vector)
    # Convert to NumPy for JAX
    G_jax = jnp.array(G)
    
    # We have 3 orientations per source.
    # We can perform LCMV for each source (3 columns) or scan.
    # make_lcmv_filter takes (C, n_chan) and (n_chan, n_sources).
    # If we pass all G, we get weights for all dipoles*orientations.
    # Let's try it (if n_src is small, 30mm spacing -> ~100 points -> 300 cols).
    
    print("Computing LCMV Weights...")
    weights = make_lcmv_filter(cov, G_jax, reg=0.05)
    
    # Weights: (n_sourcestotal, n_chan)
    print(f"Weights Shape: {weights.shape}")
    
    # 5. Apply to V1
    # Where is V1? Back of the head.
    # Sphere model coords: Z is Up, Y is Front, X is Right.
    # V1 is approx (0, -r, 0). (Back).
    
    src_pos = fwd['source_rr'] # (N, 3)
    # transform to head frame?
    # Usually in sphere model, center is 0. Head radius ~0.09m.
    # Back is Y = -0.09.
    
    # Find dipole closest to [0, -0.08, 0]
    target_pos = np.array([0, -0.08, 0])
    dists = np.linalg.norm(src_pos - target_pos, axis=1)
    v1_idx = np.argmin(dists)
    
    print(f"Targeting V1: Index {v1_idx}, Pos {src_pos[v1_idx]}")
    
    # Get weights for this dipole (3 orientations)
    # Indices in G are v1_idx*3 : v1_idx*3 + 3
    w_v1 = weights[v1_idx*3 : v1_idx*3+3]
    
    # Project Data
    # source_tc = W @ X
    s_v1_vec = apply_lcmv(X, w_v1) # (3, T)
    
    # Extract dominant orientation signal (Scalar) via SVD
    # s_vec = U S Vt. 
    # Dominant timecourse is Vt[0] scaled by S[0]
    # SVD of (3, T)
    # n_src_dim = 3
    # We want timecourse: v[0] * s[0]
    
    # JAX SVD on the 3xT block
    u_b, s_b, vt_b = jnp.linalg.svd(s_v1_vec, full_matrices=False)
    
    # Dominant scaler signal
    # Note: Phase ambiguity (sign flip).
    s_v1 = vt_b[0, :] * s_b[0] # (T,)
    
    # Beamformer Power (Vector magnitude) - OLD
    # s_v1 = jnp.linalg.norm(s_v1_vec, axis=0) # (T,) -> Rectified! 60Hz!
    
    # 6. Compare with ICA (Previous Phase)
    # Quickly run PICA just to get the top component for comparison
    print("Running PICA for comparison...")
    pica = PICA(n_components=30).fit(X)
    S_ica = pica.components_
    # Find 30Hz peak
    best_idx, _ = pica.find_spectral_peak_component(raw.info['sfreq'], 30.0)
    s_ica = S_ica[best_idx]
    
    # 7. Spectrogram Comparison
    # Normalize
    s_v1_norm = (s_v1 - jnp.mean(s_v1)) / jnp.std(s_v1)
    s_ica_norm = (s_ica - jnp.mean(s_ica)) / jnp.std(s_ica)
    s_raw = data[data.shape[0]//2] # arbitrary mid channel? No, use Oz (E75)
    # Find Oz
    oz_idx = raw.ch_names.index('E75') if 'E75' in raw.ch_names else 0
    s_raw_norm = (X[oz_idx] - jnp.mean(X[oz_idx])) / jnp.std(X[oz_idx])
    
    # Check 30Hz power
    def get_30hz_power(sig):
        spec = jnp.abs(jnp.fft.rfft(sig))
        freqs = jnp.fft.rfftfreq(len(sig), d=1/raw.info['sfreq'])
        idx = jnp.argmin(jnp.abs(freqs - 30.0))
        return spec[idx] / jnp.sum(spec)
        
    p_v1 = get_30hz_power(s_v1_norm)
    p_ica = get_30hz_power(s_ica_norm)
    p_raw = get_30hz_power(s_raw_norm)
    
    print(f"\n30Hz Power Ratios:")
    print(f"Raw (Oz): {p_raw:.5f}")
    print(f"ICA Best: {p_ica:.5f}")
    print(f"Beamformer (V1): {p_v1:.5f}")
    
    if p_v1 > p_raw:
        print("[SUCCESS] Beamformer improved SSVEP SNR.")
        
    # Plot PSDs
    plt.figure(figsize=(10, 5))
    freqs = np.fft.rfftfreq(len(s_v1_norm), d=1/raw.info['sfreq'])
    plt.semilogy(freqs, np.abs(np.fft.rfft(s_v1_norm)), label='Beamformer (V1)')
    plt.semilogy(freqs, np.abs(np.fft.rfft(s_ica_norm)), label='ICA Best', alpha=0.7)
    plt.semilogy(freqs, np.abs(np.fft.rfft(s_raw_norm)), label='Raw Oz', alpha=0.5)
    plt.xlim(0, 60)
    plt.title("PSD Comparison: V1 Beamformer vs ICA vs Raw")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (Log)")
    plt.legend()
    plt.savefig("beamformer_comparison.png")
    print("Saved 'beamformer_comparison.png'")
    
    # Save N-back notes
    print("\n[NOTE] N-back dataset was not found for this subject.")
    print("Comparison across tasks will require external dataset integration.")

if __name__ == "__main__":
    demo_beamformer()
