"""
Comprehensive NeuroJAX demonstration using the MNE Sample Dataset.
Demonstrates:
1. Data Loading (MNE Sample)
2. Preprocessing (neurojax.preprocessing.andersen_2018)
3. SPM 25 Analysis (neurojax.analysis.spm25)
   - JAX-native GLM
   - Differentiable Random Field Theory (RFT)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import mne
from mne.datasets import sample

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neurojax.preprocessing.andersen_2018 import temporal_filtering, epoch_data
from neurojax.analysis.spm25 import GeneralLinearModel, RandomFieldTheory

def main():
    print("NeuroJAX MNE Sample Data Demo")
    print("=============================")
    
    # 1. Load Data
    print("\n1. Loading MNE Sample Dataset...")
    data_path = sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
    event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
    
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    events = mne.read_events(event_fname)
    
    # Restrict to MEG (Grad + Mag) for simplicity
    raw.pick(['meg', 'eog'])
    
    # 2. Preprocessing
    print("\n2. Preprocessing (Andersen 2018 pipeline component)...")
    # Note: Sample data is already filtered 0-40Hz, but we'll apply our filter to show usage
    raw_filt = temporal_filtering(raw, l_freq=1.0, h_freq=40.0)
    
    event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}
    tmin, tmax = -0.2, 0.5
    
    # Epoching
    epochs = epoch_data(raw_filt, events, event_id, tmin=tmin, tmax=tmax)
    print(f"  Extracted {len(epochs)} epochs.")
    
    # 3. SPM 25 Analysis
    print("\n3. SPM 25 Analysis (JAX GLM & RFT)...")
    
    # Get data at N100 peak (approx 100ms)
    peak_time = 0.100
    # Average over a small window around peak
    t_idx = epochs.time_as_index(peak_time)[0]
    window = 5 # +/- samples
    
    # Extract single-trial data: (n_trials, n_sensors)
    data_left = epochs['Auditory/Left'].get_data()[:, :, t_idx-window:t_idx+window].mean(axis=2)
    data_right = epochs['Auditory/Right'].get_data()[:, :, t_idx-window:t_idx+window].mean(axis=2)
    
    n_left = data_left.shape[0]
    n_right = data_right.shape[0]
    
    # Stack Y
    Y = np.concatenate([data_left, data_right], axis=0) # (Total_trials, n_sensors)
    Y = jnp.array(Y)
    
    # Design Matrix X: [Left, Right]
    X = jnp.zeros((n_left + n_right, 2))
    X = X.at[:n_left, 0].set(1.0)
    X = X.at[n_left:, 1].set(1.0)
    
    # Fit GLM
    print("  Fitting GLM (JAX-accelerated)...")
    glm = GeneralLinearModel()
    beta = glm.fit(Y, X)
    
    # Contrast: Left - Right
    print("  Computing Contrast: Auditory Left - Right")
    contrast = jnp.array([1.0, -1.0])
    t_map = glm.compute_stats(Y, X, beta, contrast, stat_type='T')
    
    # RFT Correction
    print("  Applying RFT Correction...")
    rft = RandomFieldTheory()
    # Estimate resels (simplified for demo)
    resels = 200.0 # Standard MNE helmet often ~200-400 resels at 40Hz
    p_corrected = rft.correct_p_values(t_map, resels=resels, D=2)
    
    max_t = jnp.max(t_map)
    min_p = jnp.min(p_corrected)
    print(f"  Max T-statistic: {max_t:.2f}")
    print(f"  Min FWE-corrected p-value: {min_p:.4e}")
    
    # 4. Visualization
    print("\n4. Visualizing Results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot T-Map
    # Use MNE's built-in plotting but inject our data
    # Create an Evoked object to use plot_topomap
    evoked_dummy = epochs.average()
    
    im, _ = mne.viz.plot_topomap(np.array(t_map), evoked_dummy.info, axes=axes[0], show=False, names=None)
    axes[0].set_title(f"SPM 25 T-Map (Left - Right)\n@ {peak_time*1000:.0f} ms")
    plt.colorbar(im, ax=axes[0])
    
    # Plot Significance (Thresholded)
    # Mask non-significant sensors (p > 0.05)
    mask = np.array(p_corrected) < 0.05
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=4)
    
    im2, _ = mne.viz.plot_topomap(np.array(t_map), evoked_dummy.info, axes=axes[1], show=False, mask=mask, mask_params=mask_params)
    axes[1].set_title(f"Significant Clusters (FWE p < 0.05)\nCorrected by JAX RFT")
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle("NeuroJAX MNE Sample Demonstration", fontsize=16)
    out_file = "demo_mne_sample_results.png"
    plt.savefig(out_file)
    print(f"\nResult plot saved to {out_file}")

if __name__ == "__main__":
    main()
