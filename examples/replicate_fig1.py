
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
from neurojax.analysis.tensor import hosvd, mode_dot

def parse_group_condition(info):
    """
    Parses 'subject' and 'condition' strings to identify Group (Alcoholic/Control)
    and detailed condition (Matched/Non-Matched).
    """
    groups = []
    conditions = []
    
    for t in info['trial_info']:
        # Subject path example: smni_eeg_data/a_1_co2a0000364
        # a_... -> Alcoholic
        # c_... -> Control
        subj_str = t['subject'].split('/')[-1]
        if subj_str.startswith('a_'):
            groups.append('Alcoholic')
        elif subj_str.startswith('c_'):
            groups.append('Control')
        else:
            groups.append('Unknown')
            
        # Condition is already parsed as 'matched'/'non-matched' in loader
        # But let's trust it.
        conditions.append(t['condition'])
        
    return np.array(groups), np.array(conditions)

def main():
    dataset_path = "downloads/smni_eeg_data.tar.gz"
    
    # Needs enough subjects to have both groups
    print("1. Loading Data (20 subjects)...")
    X, info = load_uci_eeg(dataset_path, max_subjects=20) 
    # X: (Trials, Channels, Time)
    
    groups, conditions = parse_group_condition(info)
    print(f"Groups found: {np.unique(groups)}")
    print(f"Conditions found: {np.unique(conditions)}")
    
    # 2. HOSVD with Rank 16 Space
    # We want 16 spatial components.
    # Dimensions: Trials (Core), Space (16), Time (Core/Full?)
    # Figure 1 shows spatial maps and time courses.
    # The time courses in the paper are likely 'reconstructed' or 'projected' components.
    # If we do HOSVD(Rank=[R1, 16, R3]), we get 16 spatial columns in Factor[1].
    # We can project data onto these 16 spatial maps to get S(trials, time) for each map.
    # Y_i = X x_2 U_space(:, i).T
    # This gives (Trials, 1, Time) -> (Trials, Time) for component i.
    
    print("2. Computing Spatial Decomposition (Rank 16)...")
    # We can leave other ranks full or compressed. 
    # Let's compress Time to e.g. 20 to denoise, or keep full.
    # Wang 2000 mentions "16 spatial and 16 temporal components".
    ranks = [min(X.shape[0], 20), 16, min(X.shape[2], 20)] 
    core, factors = hosvd(X, ranks=ranks)
    
    U_space = factors[1] # (64, 16)
    
    print("3. Projecting and Averaging...")
    # Project data onto spatial components to get time courses
    # We do this instead of using U_time directly because we want the group averages
    # which might differ, whereas U_time is a common basis.
    # By projecting X onto U_space, we see how the raw data 'activates' these maps.
    
    # X: (N, C, T)
    # U_space: (C, 16)
    # Proj: X . U_space -> (N, T, 16) -- tensordot axes (1, 0)
    
    # jnp.tensordot(X, U_space, axes=(1, 0)) -> (N, T, 16)
    X_proj = jnp.tensordot(X, U_space, axes=(1, 0))
    # Shape (Trials, Time, 16)
    
    # Move axis to (16, Trials, Time) for easier iteration
    X_proj = jnp.moveaxis(X_proj, 2, 0)
    
    # Calculate Averages
    # We want to plot: Alcoholic (Matched), Control (Matched). Or just Group diff?
    # Paper Fig 1 shows "Alcoholic vs Control".
    # Let's aggregate by Group.
    
    t_axis = np.arange(X.shape[2]) / info['sfreq'] * 1000 # ms
    
    # Setup Figure
    fig = plt.figure(figsize=(20, 20))
    # 4x4 grid
    outer_grid = gridspec.GridSpec(4, 4, wspace=0.3, hspace=0.4)
    
    import mne
    # Setup MNE Info (simplified from demo)
    full_ch = info['channel_names']
    keep_idx = [i for i,x in enumerate(full_ch) if x not in ['X','Y','nd']]
    ch_kept = [full_ch[i] for i in keep_idx]
    def fix(n): return n.replace('Z','z').replace('FP','Fp')
    mne_ch = [fix(n) for n in ch_kept]
    
    U_space_filt = U_space[keep_idx, :]
    mne_info = mne.create_info(mne_ch, info['sfreq'], 'eeg')
    try:
        mne_info.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='ignore')
    except: pass

    # Plot 16 components
    for i in range(16):
        # Create a nested grid for this cell: Top=Map, Bottom=Time
        cell = outer_grid[i]
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, height_ratios=[1, 1], hspace=0.05)
        
        # 1. Topomap
        ax_map = fig.add_subplot(inner_grid[0])
        comp_map = U_space_filt[:, i]
        mne.viz.plot_topomap(comp_map, mne_info, axes=ax_map, show=False, contours=0)
        ax_map.set_title(f"Comp {i+1}", fontsize=10)
        
        # 2. Time Course (Group Average)
        ax_time = fig.add_subplot(inner_grid[1])
        proj_data = X_proj[i] # (Trials, Time)
        
        # Alcoholics
        mask_alc = (groups == 'Alcoholic')
        if np.any(mask_alc):
            avg_alc = np.mean(proj_data[mask_alc], axis=0)
            ax_time.plot(t_axis, avg_alc, 'r', label='Alc', linewidth=1)
            
        # Controls
        mask_ctl = (groups == 'Control')
        if np.any(mask_ctl):
            avg_ctl = np.mean(proj_data[mask_ctl], axis=0)
            ax_time.plot(t_axis, avg_ctl, 'k', label='Ctl', linewidth=1) # Black for control as in paper?
            
        ax_time.axis('off') # Clean look like typical component plots
        # Add a baseline line
        ax_time.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        
        if i == 0:
            ax_time.legend(fontsize='x-small')

    plt.suptitle("Replication of Figure 1 (Wang et al. 2000)\n16 Spatial Components with Group-Averaged Dynamics", fontsize=16)
    plt.savefig("wang2000_fig1_replication.png", dpi=150)
    print("Saved wang2000_fig1_replication.png")

if __name__ == "__main__":
    main()
