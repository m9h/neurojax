"""
Demo: Complex ICA on 40 Hz ASSR Data (ASZED)

This script demonstrates:
1. Loading ASZED data (or generating synthetic mock data if download fails/skipped).
2. filtering to the gamma band (40 Hz).
3. Computing the Analytic Signal (Hilbert Transform).
4. Separating independent complex sources using Complex FastICA.
5. Visualizing the magnitude and phase of the separated components.
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert

from neurojax.io.aszed import load_aszed_subject, download_aszed
from neurojax.analysis.ica import complex_fastica, whiten_complex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_assr(n_channels=16, duration=10, sfreq=500):
    """Generates synthetic 40Hz ASSR data mixed with noise."""
    t = np.linspace(0, duration, int(duration*sfreq))
    
    # Source 1: 40Hz Driver (ASSR) - Phase locked
    s1 = np.sin(2 * np.pi * 40 * t) 
    
    # Source 2: 10Hz Alpha (Background) - Wandering phase
    s2 = np.sin(2 * np.pi * 10 * t + np.random.normal(0, 0.5, size=t.shape))
    
    # Source 3: Noise
    s3 = np.random.randn(*t.shape)
    
    S = np.vstack([s1, s2, s3])
    
    # Mixing matrix (n_channels, 3)
    A = np.random.randn(n_channels, 3)
    
    X = A @ S
    
    # Create MNE Raw
    info = mne.create_info(ch_names=[f"C{i}" for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(X, info)
    
    # Set standard montage for topomaps
    # Need to map C0..C15 to standard names if possible, or just use standard 10-20 subset
    standard_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
        'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8'
    ]
    if n_channels == 16:
        mapping = {f"C{i}": name for i, name in enumerate(standard_names)}
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
    return raw

def main():
    # 1. Load Data
    try:
        # Try finding a real subject (skip download if big)
        logger.info("Attempting to load real ASZED data...")
        # FORCE MOCK for demo speed
        raise RuntimeError("Skipping download for demo speed")
        # download_aszed(cache_dir=Path.home() / ".cache" / "neurojax" / "aszed", force_update=False)
        # raw = load_aszed_subject("1", group="SCZ", download=False)
    except Exception as e:
        logger.warning(f"Could not load real data ({e}). Generating MOCK ASSR data.")
        raw = generate_mock_assr()

    # 2. Preprocessing
    # Filter for Gamma (38-42 Hz) center at 40
    raw_filt = raw.copy().filter(38, 42, fir_design='firwin')
    
    # Get data (n_channels, n_times)
    data = raw_filt.get_data()
    
    # 3. Analytic Signal (Complex)
    logger.info("Computing analytic signal...")
    analytic = hilbert(data, axis=1) # Scipy hilbert works on last axis
    
    # Convert to JAX apply centering
    X = jnp.array(analytic)
    X -= jnp.mean(X, axis=1, keepdims=True)
    
    # 4. Whiten
    logger.info("Whitening complex data...")
    X_white, _ = whiten_complex(X)
    
    # 5. Complex ICA
    n_components = 3
    logger.info(f"Running Complex FastICA (k={n_components})...")
    S, W = complex_fastica(X_white, n_components=n_components)
    
    # 6. Visualize
    # Recover physical mixing matrix A_hat = inv(W_white @ W) doesn't work directly due to whiten
    # Better: Activation patterns over scalp
    # Pattern = Cov(X_orig, S)
    # X (n_ch, T), S (n_comp, T)
    # A_pattern = X @ S.conj().T / T
    
    patterns = jnp.dot(X, S.conj().T) / X.shape[1]
    
    fig, axes = plt.subplots(2, n_components, figsize=(15, 6))
    
    times = np.arange(100) / raw.info['sfreq']
    
    for i in range(n_components):
        # Time course (first 100 samples)
        ax_t = axes[0, i]
        ax_t.plot(times, S[i, :100].real, label='Real')
        ax_t.plot(times, S[i, :100].imag, label='Imag')
        ax_t.plot(times, jnp.abs(S[i, :100]), 'k--', label='Env')
        ax_t.set_title(f"Comp {i} (40Hz Env)")
        if i == 0: ax_t.legend()

        # Topomap (Magnitude of spatial pattern)
        # Note: MNE plot_topomap requires numpy
        ax_topo = axes[1, i]
        mag_pattern = np.abs(patterns[:, i])
        mne.viz.plot_topomap(mag_pattern, raw.info, axes=ax_topo, show=False)
        ax_topo.set_title(f"Spatial Pattern {i}")
        
    plt.tight_layout()
    out_file = "demo_aszed_ica.png"
    plt.savefig(out_file)
    logger.info(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    main()
