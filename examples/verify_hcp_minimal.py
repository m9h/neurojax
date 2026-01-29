"""Verification script for HCP Minimal Pipeline."""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.pipeline.hcp_minimal import hcp_minimal_preproc

def generate_sensor_coords(n_channels=32):
    """Generate random points on a unit sphere."""
    # Fibonacci sphere for even distribution
    i = np.arange(0, n_channels, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n_channels)
    golden_ratio = (1 + 5**0.5)/2
    theta = 2 * np.pi * i / golden_ratio
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack([x, y, z], axis=1)

def run_verification():
    print("Initialize verification...")
    # Parameters
    sfreq = 1000.0
    duration = 5.0
    n_channels = 32
    target_fs = 200.0
    times = np.linspace(0, duration, int(sfreq*duration))
    
    # 1. Generate Synthetic Data
    # Signal: 10Hz sine wave
    # Noise: High freq noise
    # Artifact: Low freq drift (0.1Hz)
    
    signal = np.sin(2 * np.pi * 10 * times)
    noise = 0.2 * np.random.randn(*times.shape)
    drift = 2.0 * np.sin(2 * np.pi * 0.1 * times) # Strong drift
    
    data = np.tile(signal + noise + drift, (n_channels, 1))
    
    # Add some channel var
    data += 0.1 * np.random.randn(*data.shape)
    
    # Setup Geometry
    coords = generate_sensor_coords(n_channels)
    
    # Simulate Bad Channel
    bad_ch = 5
    # Store "Ground Truth" for this channel (without badness)
    ground_truth = data[bad_ch].copy()
    
    # Make it bad (e.g., flat with huge offset)
    data[bad_ch] = 500.0 # Flatline/Rail
    
    print("Running HCP Minimal Pipeline...")
    print(f"Bad Channel: {bad_ch}")
    
    # Convert to JAX
    data_jax = jnp.array(data)
    coords_jax = jnp.array(coords)
    
    # Run Pipeline
    cleaned_jax = hcp_minimal_preproc(
        data_jax,
        sfreq=sfreq,
        layout_coords=coords_jax,
        bad_channels=[bad_ch],
        target_fs=target_fs,
        highpass_freq=0.5
    )
    
    cleaned = np.array(cleaned_jax)
    
    # Verification Checks
    
    # 1. Check Sampling Rate
    expected_samples = int(duration * target_fs)
    print(f"Original shape: {data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    if abs(cleaned.shape[1] - expected_samples) > 2:
        print(f"FAILED: Resampling length mismatch. Got {cleaned.shape[1]}, expected {expected_samples}")
    else:
        print("PASSED: Resampling length correct.")
        
    # 2. Check Drift Removal
    # Average of good channels
    # Original mean (with drift)
    # Cleaned mean should be close to 0
    clean_mean = np.mean(cleaned)
    print(f"Cleaned Global Mean: {clean_mean:.4f}")
    if abs(clean_mean) < 0.1: # Loose check
        print("PASSED: Drift removed (mean zero-ish).")
    else:
        print("WARNING: Drift might not be fully removed.")
        
    # 3. Check Interpolation
    # We compare the interpolated bad channel with the resampled ground truth
    # We need to manually resample the ground truth to compare
    from scipy.signal import resample
    gt_resampled = resample(ground_truth, cleaned.shape[1])
    # Remove drift from GT manually to be fair? 
    # Or just check correlation of the 10Hz signal component.
    
    bad_ch_cleaned = cleaned[bad_ch]
    
    # Correlation
    corr = np.corrcoef(bad_ch_cleaned, gt_resampled)[0, 1]
    print(f"Interpolation Correlation with GT: {corr:.4f}")
    
    if corr > 0.8:
        print("PASSED: Interpolation recovered signal structure.")
    else:
        print("FAILED: Interpolation poor.")
        
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Original Data (Subset)
    plt.subplot(3, 1, 1)
    plt.plot(times, data[0], label="Ch0 (Good, Raw)")
    plt.plot(times, data[bad_ch], label=f"Ch{bad_ch} (Bad, Raw)")
    plt.title("Raw Data (with Drift + Bad Ch)")
    plt.legend()
    
    # Cleaned Data
    new_times = np.linspace(0, duration, cleaned.shape[1])
    plt.subplot(3, 1, 2)
    plt.plot(new_times, cleaned[0], label="Ch0 (Cleaned)")
    plt.plot(new_times, cleaned[bad_ch], label=f"Ch{bad_ch} (Interpolated)")
    plt.title("Cleaned Data (HP + Resample + Interp)")
    plt.legend()
    
    # Interpolation Check
    plt.subplot(3, 1, 3)
    plt.plot(new_times, gt_resampled, 'k--', label="Ground Truth (Resampled)")
    plt.plot(new_times, cleaned[bad_ch], 'r', label="Interpolated")
    plt.title(f"Interpolation Verification (Corr={corr:.3f})")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('verification_plot.png')
    print("Saved plot to verification_plot.png")

if __name__ == "__main__":
    run_verification()
