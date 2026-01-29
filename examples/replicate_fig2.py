
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import numpy as np
import jax
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from neurojax.io.uci_loader import load_uci_eeg
from neurojax.analysis.tensor import hosvd # mode_dot

def parse_groups(info):
    groups = []
    for t in info['trial_info']:
        subj = t['subject'].split('/')[-1]
        if subj.startswith('a_'): groups.append('Alcoholic')
        elif subj.startswith('c_'): groups.append('Control')
        else: groups.append('Unknown')
    return np.array(groups)

def main():
    dataset_path = "downloads/smni_eeg_data.tar.gz"
    
    print("1. Loading Data...")
    X, info = load_uci_eeg(dataset_path, max_subjects=20) 
    groups = parse_groups(info)
    
    # 2. HOSVD to get Shared Temporal Basis
    # Rank 5 approximation
    print("2. HOSVD Decomposition...")
    core, factors = hosvd(X, ranks=[10, 10, 10])
    
    U_time = factors[2] # (Time, Rank)
    
    # 3. Identify P3 Component (Max Energy in 280-500ms)
    t_ms = np.arange(X.shape[2]) / info['sfreq'] * 1000
    mask_p3 = (t_ms >= 280) & (t_ms <= 500)
    
    p3_scores = []
    for i in range(U_time.shape[1]):
        comp = U_time[:, i]
        energy_window = np.sum(comp[mask_p3]**2)
        total_energy = np.sum(comp**2) + 1e-9
        p3_scores.append(energy_window / total_energy)
        
    p3_idx = np.argmax(p3_scores)
    p3_wave = U_time[:, p3_idx]
    
    # Ensure positive polarity in window
    if np.max(p3_wave[mask_p3]) < np.abs(np.min(p3_wave[mask_p3])):
        p3_wave = -p3_wave
        print(f"Inverted polarity of Component {p3_idx}")
        
    print(f"Identified P3 Component: Index {p3_idx}")
    
    # 4. Compute Group-Specific Spatial Maps
    # We project the Group-Averaged Data (Space x Time) onto the P3 Time Course
    # Map = Data . P3_Wave
    # Data is (N, C, T)
    
    # Alcoholic Map
    idx_alc = (groups == 'Alcoholic')
    X_alc = X[idx_alc] # (N_a, C, T)
    # Mean over trials
    Mean_alc = np.mean(X_alc, axis=0) # (C, T)
    # Project: (C, T) . (T,) -> (C,)
    Map_alc = np.dot(Mean_alc, p3_wave)
    
    # Control Map
    idx_ctl = (groups == 'Control')
    X_ctl = X[idx_ctl]
    Mean_ctl = np.mean(X_ctl, axis=0)
    Map_ctl = np.dot(Mean_ctl, p3_wave)
    
    # 5. Visualize (Figure 2 layout)
    # (a) Temporal P3
    # (b) Map Alc
    # (c) Map Ctl
    
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)
    
    # (a) Temporal
    ax_t = fig.add_subplot(gs[0])
    ax_t.plot(t_ms, p3_wave, 'k', linewidth=2)
    ax_t.set_title("(a) P3 Temporal Component")
    ax_t.set_xlabel("Time (ms)")
    ax_t.axvspan(280, 500, color='yellow', alpha=0.1)
    ax_t.grid(True, alpha=0.3)
    
    # MNE Setup
    import mne
    full_ch = info['channel_names']
    keep_idx = [i for i,x in enumerate(full_ch) if x not in ['X','Y','nd']]
    ch_kept = [full_ch[i] for i in keep_idx]
    def fix(n): return n.replace('Z','z').replace('FP','Fp')
    mne_ch = [fix(n) for n in ch_kept]
    mne_info = mne.create_info(mne_ch, info['sfreq'], 'eeg')
    try:
        mne_info.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='ignore')
    except: pass
    
    # Filter maps
    Map_alc = Map_alc[keep_idx]
    Map_ctl = Map_ctl[keep_idx]
    
    # Normalize scales for comparison?
    vlim = max(np.max(np.abs(Map_alc)), np.max(np.abs(Map_ctl)))
    
    # (b) Map Alc
    ax_alc = fig.add_subplot(gs[1])
    mne.viz.plot_topomap(Map_alc, mne_info, axes=ax_alc, show=False, contours=4, vlim=(-vlim, vlim))
    ax_alc.set_title("(b) Alcoholic P3 Source")
    
    # (c) Map Ctl
    ax_ctl = fig.add_subplot(gs[2])
    mne.viz.plot_topomap(Map_ctl, mne_info, axes=ax_ctl, show=False, contours=4, vlim=(-vlim, vlim))
    ax_ctl.set_title("(c) Control P3 Source")
    
    plt.suptitle("Replication of Figure 2 (Wang et al. 2000)\nP3 Component and Group-Specific Topographies", fontsize=14)
    plt.savefig("wang2000_fig2_replication.png", dpi=150)
    print("Saved wang2000_fig2_replication.png")

if __name__ == "__main__":
    main()
