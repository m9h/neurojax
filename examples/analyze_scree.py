
import os
import sys
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import jax
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from neurojax.io.uci_loader import load_uci_eeg
from neurojax.analysis.tensor import unfold

def compute_svd_spectrum(tensor, mode):
    """
    Computes singular values of the mode-n unfolding of the tensor.
    """
    # Unfold along mode (0=Trials, 1=Space, 2=Time)
    mat = unfold(tensor, mode) # (Dim_n, Product_of_others)
    
    # SVD
    # We only need singular values (s)
    # Using jnp.linalg.svd with compute_uv=False is faster but sometimes limited on CPU? 
    # Let's use standard full=False.
    # Note: On CPU with large matrix, this might be slow. 
    # But shape is (64, N*T) -> 64 is small. (256, N*C) -> 256 is okay.
    s = jnp.linalg.svd(mat, compute_uv=False)
    
    return s

def plot_scree(s, title, cutoff, ax):
    """
    Plots scree plot and marks variance at cutoff.
    """
    # Variance explained by each component is s^2 / sum(s^2)
    var = s**2
    total_var = jnp.sum(var)
    explained_ratio = var / total_var
    cumulative_var = jnp.cumsum(explained_ratio)
    
    # Plot Explained Variance Ratio (Scree)
    ax.plot(np.arange(1, len(s)+1), explained_ratio, 'b.-', label='Var Ratio')
    ax.set_title(title)
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_xlabel("Component Index")
    ax.grid(True, alpha=0.3)
    
    # Mark Cutoff
    if cutoff <= len(s):
        cum_val = cumulative_var[cutoff-1]
        ax.axvline(cutoff, color='r', linestyle='--', label=f'Cutoff={cutoff}')
        ax.plot(cutoff, explained_ratio[cutoff-1], 'ro')
        
        # Add text
        y_text = np.max(explained_ratio) * 0.8
        ax.text(cutoff + 1, y_text, f"CumVar @ {cutoff}: {cum_val:.1%}\n(Var: {explained_ratio[cutoff-1]:.3f})")
        
    ax.legend()
    
    return cumulative_var[cutoff-1] if cutoff <= len(s) else 0.0


def center_tensor(tensor, mode):
    """
    Centers the tensor by subtracting the mean along the specified mode.
    """
    mean = jnp.mean(tensor, axis=mode, keepdims=True)
    return tensor - mean

def main():
    dataset_path = "downloads/smni_eeg_data.tar.gz"
    
    print("1. Loading Data...")
    X, info = load_uci_eeg(dataset_path, max_subjects=10)
    print(f"Data Shape: {X.shape}") # (Trials, Channels, Time)
    
    # Baseline correction (Center over Time) - standard EEG step
    # Remove mean of each channel/trial (DC offset)
    X = X - jnp.mean(X, axis=2, keepdims=True)
    
    # 2. Compute Spectra (Raw vs Centered over Trials)
    # Wang 2000 might have analyzed the variance of the *deviations* from the mean?
    # Or just the raw data including the ERP.
    # Let's check both.
    
    # Case A: "Raw" (Baseline corrected but containing ERP)
    print("2. Computing Raw Spectrum...")
    s_space_raw = compute_svd_spectrum(X, 1)
    s_time_raw = compute_svd_spectrum(X, 2)
    
    # Case B: "Centered" (Remove Grand Mean / Avg ERP across trials)
    # This analyzes the 'Single Trial Noise/Variability' beyond the shared ERP.
    # Or maybe they centered across the Mode they unfolded?
    # Standard PCA on Mode-n unfolding removes the mean of the columns/rows of that matrix.
    # For Mode 2 (Time), unfolding is (Time, Trials*Channels). 
    # Centering would be removing mean of each Time point (across all trials/channels)?
    # Or removing mean of each Trial/Channel trace (which we did with baseline)?
    
    # Let's try removing the Mean ERP (Average over Trials)
    X_centered = center_tensor(X, 0) # Remove Mean over Trials
    
    print("3. Computing Centered Spectrum (Noise/Variability)...")
    s_space_centered = compute_svd_spectrum(X_centered, 1)
    s_time_centered = compute_svd_spectrum(X_centered, 2)
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate SNR (Approximate)
    # Total Variance
    var_total = jnp.var(X)
    # Variance of the Mean ERP (Signal)
    mean_erp = jnp.mean(X, axis=0)
    var_signal = jnp.var(mean_erp)
    # Variance of Noise (Residual)
    var_noise = jnp.var(X - mean_erp)
    
    snr = var_signal / var_noise
    print(f"\nEstimated SNR (Signal Variance / Noise Variance): {snr:.4f}")
    
    # Plot Spatial Comparison
    ax = axes[0, 0]
    ax.plot(s_space_raw**2 / np.sum(s_space_raw**2), 'b-', label='Raw (Signal+Noise)')
    ax.plot(s_space_centered**2 / np.sum(s_space_centered**2), 'r--', label='Centered (Noise Only)')
    ax.set_title("Spatial Spectrum (Linear)")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend()
    ax.grid(True)
    
    # Plot Temporal Comparison
    ax = axes[0, 1]
    ax.plot(s_time_raw**2 / np.sum(s_time_raw**2), 'b-', label='Raw')
    ax.plot(s_time_centered**2 / np.sum(s_time_centered**2), 'r--', label='Centered')
    ax.set_title("Temporal Spectrum (Linear)")
    ax.legend()
    ax.grid(True)
    
    # Log Scale Comparison (Spatial)
    ax = axes[1, 0]
    ax.semilogy(s_space_raw**2 / np.sum(s_space_raw**2), 'b-')
    ax.semilogy(s_space_centered**2 / np.sum(s_space_centered**2), 'r--')
    ax.set_title("Spatial Spectrum (Log Scale)")
    ax.grid(True, which="both", ls="-")
    
    # Log Scale Comparison (Temporal)
    ax = axes[1, 1]
    ax.semilogy(s_time_raw**2 / np.sum(s_time_raw**2), 'b-')
    ax.semilogy(s_time_centered**2 / np.sum(s_time_centered**2), 'r--')
    ax.set_title("Temporal Spectrum (Log Scale)")
    ax.grid(True, which="both", ls="-")
    
    print("-" * 40)
    print("Raw Data (Base-corrected):")
    # We re-calculate to print
    var_s_r = jnp.cumsum(s_space_raw**2 / jnp.sum(s_space_raw**2))[15]
    var_t_r = jnp.cumsum(s_time_raw**2 / jnp.sum(s_time_raw**2))[42]
    print(f"Spatial (16): {var_s_r:.1%}")
    print(f"Temporal (43): {var_t_r:.1%}")
    
    print("\nCentered Data (Mean ERP Removed):")
    # Recalculate cumsum for centered
    var_s_c = jnp.cumsum(s_space_centered**2 / jnp.sum(s_space_centered**2))[15]
    var_t_c = jnp.cumsum(s_time_centered**2 / jnp.sum(s_time_centered**2))[42]
    print(f"Spatial (16): {var_s_c:.1%}")
    print(f"Temporal (43): {var_t_c:.1%}")
    print("-" * 40)
    
    plt.tight_layout()
    plt.savefig("scree_plots_analysis.png", dpi=150)
    print("Saved scree_plots_analysis.png")

if __name__ == "__main__":
    main()
