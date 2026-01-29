"""
ASR Demonstration Script
========================
This script demonstrates the Artifact Subspace Reconstruction (ASR) algorithm
implemented in `neurojax.preprocessing.asr`.

It generates synthetic EEG data with:
1. "Clean" background activity (sine waves + noise).
2. "Artifact" bursts injected into a specific linear subspace (simulating muscle/movement).

Steps:
1. Generate data.
2. Calibrate ASR on a clean segment.
3. Apply ASR to the whole dataset.
4. Visualize the removal of artifacts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.preprocessing.asr import calibrate_asr, apply_asr

# Set random seed
key = jax.random.PRNGKey(42)

def generate_synthetic_data(n_channels=8, n_samples=2000, srate=250):
    """Generate synthetic EEG-like data with artifacts."""
    t = jnp.linspace(0, n_samples / srate, n_samples)
    
    # 1. Background Signal (Clean)
    # Random mixtures of sines
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Source signals
    freqs = jnp.array([10.0, 2.0, 50.0]) # Alpha, Delta, Line/Gamma
    sources = jnp.array([
        jnp.sin(2 * jnp.pi * f * t) for f in freqs
    ])
    
    # Mixing matrix for sources -> channels
    mixing = jax.random.normal(k1, (n_channels, 3))
    clean_signal = jnp.dot(mixing, sources)
    
    # Add sensor noise
    noise = 0.1 * jax.random.normal(k2, (n_channels, n_samples))
    clean_signal = clean_signal + noise
    
    # 2. Artifact Injection
    # Create a high-amplitude artifact in a specific random direction
    artifact_mixing = jax.random.normal(k3, (n_channels, 1))
    
    # Artifact time course: Burst in the middle
    # Gaussian envelope
    center_idx = n_samples // 2
    width = n_samples // 10
    envelope = jnp.exp(-0.5 * ((jnp.arange(n_samples) - center_idx) / width) ** 2)
    
    # Artifact signal: High frequency noise modulated by envelope
    # Artifact signal: High frequency noise modulated by envelope
    k3, k4 = jax.random.split(k3)
    artifact_source = 20.0 * envelope * jax.random.normal(k4, (n_samples,))
    
    artifact_component = jnp.dot(artifact_mixing, artifact_source[None, :])
    
    raw_data = clean_signal + artifact_component
    
    return t, raw_data, clean_signal, artifact_component

def main():
    print("Generating synthetic data...")
    t, raw, clean_ref, art_ref = generate_synthetic_data()
    
    n_channels, n_samples = raw.shape
    print(f"Data Shape: {raw.shape}")
    
    # Split into Calibration (Clean start) and Test (Artifact middle)
    # In this synthetic example, the first 20% is clean-ish (artifact is near middle)
    calib_samples = n_samples // 5
    calibration_data = raw[:, :calib_samples]
    
    print("Calibrating ASR...")
    asr_state = calibrate_asr(calibration_data, cutoff=3.0) # Strict cutoff for demo
    
    print("Applying ASR...")
    cleaned_data = apply_asr(
        raw, 
        asr_state, 
        window_size=100, 
        step_size=50
    )
    
    # Compute metrics
    # Artifact Reduction Ratio (in terms of RMS in artifact region)
    idx_start = n_samples // 2 - 100
    idx_end = n_samples // 2 + 100
    
    rms_raw = jnp.sqrt(jnp.mean(raw[:, idx_start:idx_end]**2))
    rms_clean = jnp.sqrt(jnp.mean(cleaned_data[:, idx_start:idx_end]**2))
    rms_ref = jnp.sqrt(jnp.mean(clean_ref[:, idx_start:idx_end]**2))
    
    print(f"RMS (Artifact Region) - Raw: {rms_raw:.2f}")
    print(f"RMS (Artifact Region) - Cleaned: {rms_clean:.2f}")
    print(f"RMS (Artifact Region) - True Clean: {rms_ref:.2f}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Plot Channel 0 (arbitrary)
    ch = 0
    axes[0].plot(t, raw[ch], label='Raw Input', color='tab:red', alpha=0.7)
    axes[0].plot(t, clean_ref[ch], label='Ground Truth (Clean)', color='k', linestyle='--', alpha=0.5)
    axes[0].set_title(f"Channel {ch} - Time Domain")
    axes[0].legend()
    
    axes[1].plot(t, cleaned_data[ch], label='ASR Cleaned', color='tab:blue')
    axes[1].plot(t, clean_ref[ch], label='Ground Truth (Clean)', color='k', linestyle='--', alpha=0.5)
    axes[1].set_title(f"Channel {ch} - ASR Output")
    axes[1].legend()

    # Plot difference (Residual Artifact)
    axes[2].plot(t, raw[ch] - cleaned_data[ch], label='Removed Content', color='tab:orange')
    axes[2].set_title(f"Channel {ch} - Removed Artifact Subspace")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('asr_demonstration.png')
    print("Saved plot to asr_demonstration.png")

if __name__ == "__main__":
    main()
