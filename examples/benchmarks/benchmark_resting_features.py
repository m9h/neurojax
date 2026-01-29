
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.io.cmi import CMILoader
from neurojax.analysis.filtering import filter_fft, notch_filter_fft, robust_reference
from neurojax.analysis.spectral import SpecParam
import mne
from jax import vmap

SUBJECT_ID = "sub-NDARGU729WUR"

def benchmark_features():
    print(f"\n--- Extracting Resting Features for {SUBJECT_ID} ---")
    loader = CMILoader(SUBJECT_ID)
    raw = loader.load_task("RestingState")
    
    # 1. Preprocess (JAX)
    print("Preprocessing (Highpass 1Hz, Notch 60Hz, Ref)...")
    data = raw.get_data().astype(np.float32)
    sfreq = raw.info['sfreq']
    
    # Send to JAX
    # To save memory/time, maybe process only 60s
    data_jax = jnp.array(data[:, :int(60*sfreq)])
    
    data_clean = robust_reference(notch_filter_fft(filter_fft(data_jax, sfreq, f_low=1.0), sfreq))
    data_clean.block_until_ready()
    
    # 2. Compute PSD (Welch) using MNE on cleaned data
    # (JAX FFT Welch is possible but MNE is robust)
    print("Computing PSD (Welch)...")
    # Convert back to numpy for MNE PSD
    data_np = np.array(data_clean)
    
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_np, sfreq, fmin=1, fmax=100, n_fft=int(2*sfreq), n_overlap=int(sfreq), verbose=False
    )
    # psds: (n_channels, n_freqs)
    log_psds = np.log10(psds)
    
    # 3. Fit SpecParam (FOOOF)
    print("Fitting SpecParam (JAX vmap) to all channels...")
    t0 = time.time()
    
    # Setup Data
    freqs_jax = jnp.array(freqs)
    psds_jax = jnp.array(log_psds)
    
    # Vectorized fit
    # SpecParam.fit(freqs, spectrum, n_peaks=3...)
    # We need to partial fix freqs
    from functools import partial
    
    def fit_single(spectrum):
        return SpecParam.fit(freqs_jax, spectrum, n_peaks=3, steps=1000, lr=0.1)
        
    models = vmap(fit_single)(psds_jax)
    # models is a PyTree of SpecParams with extra batch dim
    
    # How to block/wait for eqx module?
    # Ensure computation is done by accessing a parameter
    _ = models.aperiodic_params.block_until_ready()
    
    t_fit = time.time() - t0
    print(f"SpecParam Fit Time (129 channels): {t_fit:.2f}s")
    
    # 4. Extract Features
    # Aperiodic Exponent
    exps = models.aperiodic_params[:, 2] # (N_ch,)
    print(f"Mean Aperiodic Exponent: {jnp.mean(exps):.2f} +/- {jnp.std(exps):.2f}")
    
    # Peak Alpha (Find peak in 8-13Hz)
    # peaks: (N_ch, n_peaks, 3) [center, amp, width]
    peaks = models.peak_params
    
    # Naive extraction: find peak with center in [8, 13] and max amplitude
    centers = peaks[:, :, 0]
    amps = peaks[:, :, 1]
    
    # Mask for alpha
    alpha_mask = (centers >= 8.0) & (centers <= 13.0)
    
    # Amp of alpha peaks
    alpha_amps = jnp.where(alpha_mask, amps, 0.0)
    
    # Best alpha per channel
    best_idx = jnp.argmax(alpha_amps, axis=1) # (N_ch,)
    
    # Gather
    # Advanced indexing in JAX
    idx_grid = jnp.arange(len(centers))
    paf = centers[idx_grid, best_idx]
    pap = alpha_amps[idx_grid, best_idx] # Peak Alpha Power
    
    valid_alpha = pap > 0.05 # Threshold
    
    print(f"Channels with Alpha Peak: {jnp.sum(valid_alpha)} / {len(centers)}")
    if jnp.sum(valid_alpha) > 0:
        mean_paf = jnp.mean(paf[valid_alpha])
        print(f"Mean Peak Alpha Frequency: {mean_paf:.2f} Hz")
    
    # Simple Plot
    # Plot PSD of channel with massive alpha
    best_ch = jnp.argmax(pap)
    
    # Reconstruct model for best channel manually or use get_model
    # Get params for best_ch
    model_best = jax.tree_util.tree_map(lambda x: x[best_ch], models)
    pred = model_best.get_model(freqs_jax)
    
    print(f"Best Alpha Channel: {raw.ch_names[best_ch]} (PAF: {paf[best_ch]:.2f} Hz)")

if __name__ == "__main__":
    benchmark_features()
