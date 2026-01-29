
import os
import sys
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from neurojax.io.uci_loader import load_uci_eeg
from neurojax.analysis.tensor import hosvd, tucker_to_tensor, unfold

def plot_components(factors, core, info):
    """
    Visualize the components from HOSVD.
    Mode 1: Channels (Space)
    Mode 2: Time
    Mode 0: Trials (Subjects/Conditions)
    """
    
    # Factors order matches tensor dims: (Trials, Channels, Time)
    # factor[0]: Trials (Weights per trial)
    # factor[1]: Channels (Spatial Topography)
    # factor[2]: Time (Temporal Waveforms)
    
    U_trials = factors[0]
    U_space = factors[1]
    U_time = factors[2]
    
    n_comps_to_plot = 3
    
    # Analyze components to find P300-like features (Peak 280-600ms)
    # Time vector in ms
    t_ms = np.arange(U_time.shape[0]) / info['sfreq'] * 1000
    
    # Identify P300 candidates
    # We look for max amplitude in 280-600ms window in the temporal mode
    mask_p300 = (t_ms >= 280) & (t_ms <= 600)
    
    # Heuristic: Score = Mean Amplitude in window * Peak Prominence
    scores = []
    for i in range(U_time.shape[1]):
        # Check polarity (SVD sign is indeterminate, we often align to positive peak)
        # We want the max deviation to be positive in the window if possible
        # Or we check correlation with a template window
        comp = U_time[:, i]
        
        # Simple metric: Peak value in window
        peak_val = np.max(np.abs(comp[mask_p300]))
        peak_idx = np.argmax(np.abs(comp[mask_p300]))
        
        # Check if it's a parietal spatial map?
        # We will just sort by temporal energy in P300 window relative to total energy
        energy_in_window = np.sum(comp[mask_p300]**2)
        total_energy = np.sum(comp**2)
        ratio = energy_in_window / (total_energy + 1e-9)
        scores.append(ratio)
        
    # Sort indices by P300-likeness
    sorted_indices = np.argsort(scores)[::-1]
    
    print("Component Sorting (P300 Likelihood):", sorted_indices)

    # Use GridSpec for better control
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, n_comps_to_plot, hspace=0.4, wspace=0.3)
    
    # MNE Setup
    try:
        import mne
        full_ch_names = info['channel_names']
        to_drop = ['X', 'Y', 'nd']
        keep_indices = [i for i, ch in enumerate(full_ch_names) if ch not in to_drop]
        ch_names_kept = [full_ch_names[i] for i in keep_indices]
        
        def fix_ch_name(name):
            if name.endswith('Z'): name = name[:-1] + 'z'
            if name.startswith('FP'): name = 'Fp' + name[2:]
            return name
        mne_ch_names = [fix_ch_name(ch) for ch in ch_names_kept]
        
        U_space_filtered = U_space[keep_indices, :]
        mne_info = mne.create_info(mne_ch_names, sfreq=info['sfreq'], ch_types='eeg')
        
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            mne_info.set_montage(montage, on_missing='ignore')
        except Exception as e:
            print(f"Montage warning: {e}")

        # Plot Sorted Components
        for plot_idx, comp_idx in enumerate(sorted_indices[:n_comps_to_plot]):
            # 1. Spatial
            ax = fig.add_subplot(gs[0, plot_idx])
            comp_space = U_space_filtered[:, comp_idx]
            
            # Auto-invert if P300 is negative (SVD sign ambiguity)
            # We assume P300 is positive parietal. 
            # We'll rely on the visual inspection for now or check Parietal sign
            # Let's verify typical polarity
            
            mne.viz.plot_topomap(comp_space, mne_info, axes=ax, show=False, contours=0)
            ax.set_title(f"Component {comp_idx+1}\nSpatial Topography")
            
            # 2. Temporal
            ax2 = fig.add_subplot(gs[1, plot_idx])
            comp_time = U_time[:, comp_idx]
            ax2.plot(t_ms, comp_time, linewidth=2)
            
            # Highlight P300 window
            ax2.axvspan(300, 500, color='yellow', alpha=0.1, label='P300 Window')
            if plot_idx == 0: ax2.legend()
            
            ax2.set_title(f"Temporal Dynamics")
            ax2.set_xlabel("Time (ms)")
            ax2.grid(True, alpha=0.3)

            # 3. Trial Loadings
            ax3 = fig.add_subplot(gs[2, plot_idx])
            conditions = [t['condition'] for t in info['trial_info']]
            cond_unique = sorted(list(set(conditions)))
            colors = ['r', 'b', 'g', 'k']
            
            for j, cond in enumerate(cond_unique):
                idx = [k for k, c in enumerate(conditions) if c == cond]
                ax3.scatter(np.array(idx), U_trials[idx, comp_idx], label=cond, s=15, alpha=0.6, c=colors[j])
            
            ax3.set_title(f"Trial Loadings")
            if plot_idx == 0: ax3.legend()

    except ImportError:
        print("MNE not installed.")
    
    plt.suptitle("Multiway SVD Components (Sorted by P300 Window Energy)", fontsize=16)
    plt.savefig("multiway_svd_paper_matched.png", dpi=150)
    print("Saved multiway_svd_paper_matched.png")

def main():
    dataset_path = "downloads/smni_eeg_data.tar.gz"
    
    print("1. Loading Data...")
    # Load a subset for demo speed if needed, or full
    # There are 122 subjects. It might take a minute.
    # Let's load e.g. 10 subjects to be quick but representative
    X, info = load_uci_eeg(dataset_path, max_subjects=10)
    print(f"Loaded Tensor Shape: {X.shape}")
    
    # X shape: (Trials, Channels, Time)
    
    print("2. Computing HOSVD...")
    # We keep full rank or truncate?
    # Paper suggests retrieving specific components.
    # Let's do a truncated HOSVD to extracting top 5 components per mode
    ranks = [5, 5, 5] 
    core, factors = hosvd(X, ranks=ranks)
    
    print(f"Core Shape: {core.shape}")
    print(f"Factor Shapes: {[f.shape for f in factors]}")
    
    print("3. Visualizing...")
    plot_components(factors, core, info)
    
    # Calculate explained variance?
    # Reconstruction error
    X_rec = tucker_to_tensor(core, factors)
    err = jnp.linalg.norm(X - X_rec) / jnp.linalg.norm(X)
    print(f"Reconstruction Error (Relative): {err:.4f}")

if __name__ == "__main__":
    main()
