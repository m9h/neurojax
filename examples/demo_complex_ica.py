
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
import jax
from scipy.signal import hilbert
from neurojax.io.cmi import CMILoader
from neurojax.analysis.complex_ica import ComplexICA

SUBJECT_ID = "sub-NDARGU729WUR"

def demo_complex_ica():
    print("=== Complex ICA Demo: Analytic Signals ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except:
        print("Data not found.")
        return
        
    raw.load_data()
    raw.filter(1, 90, verbose=False)
    data = raw.get_data() # (C, T)
    sfreq = raw.info['sfreq']
    
    # 2. Compute Analytic Signal (Hilbert)
    print("Computing Analytic Signal (Hilbert Transform)...")
    # Use scipy hilbert (fast enough for cpu)
    data_analytic = hilbert(data, axis=1)
    X_complex = jnp.array(data_analytic)
    print(f"Complex Data Shape: {X_complex.shape}, Dtype: {X_complex.dtype}")
    
    # 3. Complex ICA
    n_components = 30 # Reduce dimension slightly to speed up / focus
    print(f"Running Complex FastICA (k={n_components})...")
    
    cica = ComplexICA(n_components=n_components)
    cica.fit(X_complex)
    
    S = cica.components_ # (k, T) Complex
    
    print("ICA Converged.")
    
    # 4. Analyze Components for SSVEP (30Hz)
    # Check PSD of |S|? Or PSD of S (Complex PSD)?
    # Complex PSD: FFT(S). 
    # Since S is analytic, it only has positive frequencies.
    
    # Compute Power Spectral Density
    freqs = jnp.fft.rfftfreq(S.shape[1], d=1/sfreq)
    # For analytic signal, FFT is full spectrum?
    # FFT of analytic signal: 2*FFT(real) for f>0, 0 for f<0.
    # So we can just take abs(FFT(S))
    
    specs = jnp.abs(jnp.fft.fft(S, axis=1)) 
    # freqs for full FFT
    all_freqs = jnp.fft.fftfreq(S.shape[1], d=1/sfreq)
    
    # Mask for 30Hz (+ve)
    # Need to be careful with fftshift/ordering
    # standard fft: 0, pos, neg.
    
    target_freq = 30.0
    idx_30 = jnp.argmin(jnp.abs(all_freqs - target_freq))
    
    power_30 = specs[:, idx_30]
    total_power = jnp.sum(specs**2, axis=1) # Energy
    
    ratio = (power_30**2) / (total_power + 1e-12) # Just rough peakiness
    
    # Rank by 30Hz Ratio
    ranked_indices = jnp.argsort(ratio)[::-1]
    best_ic = ranked_indices[0]
    best_ratio = ratio[best_ic]
    
    print(f"Best Complex IC: {best_ic} (Ratio: {best_ratio:.5f})")
    
    # Top 3
    print("Top 3 Candidates:")
    for i in range(3):
        idx = ranked_indices[i]
        print(f" IC {idx}: Ratio {ratio[idx]:.5f}")
        
    # Visualize Best Component (Magnitude and Phase)
    s_best = S[best_ic]
    amp = jnp.abs(s_best)
    phase = jnp.angle(s_best)
    
    times = raw.times[:1000] # First second
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, amp[:1000], label="Envelope (Amplitude)")
    plt.title(f"Complex IC {best_ic}: Amplitude Dynamics")
    plt.ylabel("Magnitude")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(times, phase[:1000], label="Phase", color='orange', alpha=0.5)
    plt.title("Phase Dynamics")
    plt.ylabel("Phase (rad)")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig("complex_ica_ssvep.png")
    print("Saved 'complex_ica_ssvep.png'")
    
    # Check Phase Consistency?
    # Ideally, Phase should be 30Hz slope.
    # dPhase/dt = frequency.
    # Unwrapped phase
    unwrapped = jnp.unwrap(phase)
    slope = (unwrapped[-1] - unwrapped[0]) / (raw.times[-1] - raw.times[0])
    est_freq = slope / (2*jnp.pi)
    
    print(f"Estimated Frequency from Phase Slope: {est_freq:.2f} Hz")
    
    if abs(est_freq - 30.0) < 2.0:
        print("[SUCCESS] Complex ICA locked onto 30Hz phase dynamics.")

if __name__ == "__main__":
    demo_complex_ica()
