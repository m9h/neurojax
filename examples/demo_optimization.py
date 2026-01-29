
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
import jax
from neurojax.io.cmi import CMILoader
from neurojax.analysis.ica import PICA
from neurojax.spatial.splines import SphericalSpline

SUBJECT_ID = "sub-NDARGU729WUR"

def make_design_matrix(raw, events, event_id, freq=30.0):
    """
    Construct a GLM Design Matrix for SSVEP.
    Regressors:
    1. 30Hz Sine (Modulated by Trial)
    2. 30Hz Cosine (Modulated by Trial)
    3. Boxcar (Trial ON/OFF)
    """
    sfreq = raw.info['sfreq']
    n_time = len(raw.times)
    times = raw.times
    
    # Trial Mask
    mask = np.zeros(n_time)
    
    # Duration of trial? Assuming 1s or until next event?
    # Epoching usually uses tmin=-0.2, tmax=0.6.
    # Let's assume stimulus is ON for 0.5s?
    stim_dur = 0.5 
    
    # Events
    # event_id map: let's use all 'contrastTrial_start' or similar
    # From demo_norcia: `np.str_('contrastTrial_start'): 4`
    # or `contrastChangeB1_start`: 3
    
    t_idxs = events[:, 0]
    e_ids = events[:, 2]
    
    # Find trial starts
    # ID 4 or 6 or 8?
    target_ids = []
    if 'contrastTrial_start' in event_id: target_ids.append(event_id['contrastTrial_start'])
    if 'left_target' in event_id: target_ids.append(event_id['left_target'])
    if 'right_target' in event_id: target_ids.append(event_id['right_target'])
    
    for i, t_idx in enumerate(t_idxs):
        if e_ids[i] in target_ids:
            # Mark stimulus duration
            t_end = int(t_idx + stim_dur * sfreq)
            if t_end > n_time: t_end = n_time
            mask[t_idx:t_end] = 1.0
            
    # Regressors
    time_vec = np.arange(n_time) / sfreq
    boxcar = mask
    sine = np.sin(2 * np.pi * freq * time_vec) * mask
    cosine = np.cos(2 * np.pi * freq * time_vec) * mask
    
    return np.stack([boxcar, sine, cosine], axis=0) # (3, n_time)

def demo_optimization():
    print("=== Optimization: PARE & Design Matrix ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except:
        print("Data not found.")
        return
        
    raw.load_data()
    # Highpass 1Hz to enable ICA/Design limits
    raw.filter(1, 90, verbose=False)
    
    data = raw.get_data() # (C, T)
    sfreq = raw.info['sfreq']
    
    # 2. PARE Correction (Spatial Deblurring / True Average Ref)
    print("\n--- Applying PARE Correction ---")
    
    # Get Positions (Unit Sphere)
    montage = raw.get_montage()
    if montage is None:
        print("No montage found. Loading GSN-129...")
        montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        raw.set_montage(montage)
        
    # Convert to (N, 3)
    # Positions in 'm'?
    pos_dict = raw.get_montage().get_positions()['ch_pos']
    pos_list = []
    for ch in raw.ch_names:
        if ch in pos_dict:
            pos_list.append(pos_dict[ch])
        else:
            pos_list.append([0,0,1]) # dummy
    
    pos = np.array(pos_list)
    # Project to Unit Sphere
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    pos_unit = pos / (norms + 1e-12)
    
    # Initialize Spline
    # PARE requires JAX arrays
    spline = SphericalSpline(jnp.array(pos_unit))
    
    # Correct Data (Apply to each timepoint? Or vmap over time?)
    # pare_correction takes (N_chan,) -> returns (N_chan,)
    # We vmap over Time axis (1)
    # data_jax: (C, T) -> we want to map over T, so input (C,)
    
    # data_jax: (C, T) -> we want to map over T, so input (C,) for each timepoint
    # Spline fitting is independent per timepoint.
    
    # pare_correction returns (X_corrected, c0)
    # X_corrected: (C,). c0: scalar.
    # If we use vmap with in_axes=1 (scan over T columns),
    # output X_corrected will be stacked along axis 0 by default -> (T, C) unless out_axes specified.
    # We want X_corrected to be (C, T) to match input. So out_axes=1 for first output.
    # c0 is scalar. stacking T scalars -> (T,). out_axes=0 is the only option for 1D result.
    # trying out_axes=(1, 0)
    
    correction_fn = jax.vmap(spline.pare_correction, in_axes=1, out_axes=(1, 0))
    
    X = jnp.array(data)
    # Check DC before
    dc_before = jnp.mean(X, axis=1) # across time? No, mean spatial DC?
    
    X_pare, c0 = correction_fn(X)
    
    print("PARE applied.")
    print(f"Mean spatial DC removed (c0): {jnp.mean(jnp.abs(c0)):.2e}")
    
    # 3. PICA on PARE Data
    print("\n--- PICA on PARE Data ---")
    pica = PICA(n_components=118) # Use same K as before for fair comparison
    pica.fit(X_pare)
    
    S = pica.components_ # (K, T)
    A = pica.mixing_
    
    # 4. Design Matrix Matching
    print("\n--- Design Matrix Ranking ---")
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"Events: {event_id}")
    
    design = make_design_matrix(raw, events, event_id, freq=30.0)
    # design shape (3, T)
    
    # Correlate Components with Design
    # We want max correlation with Sine/Cosine (SSVEP)
    # Combine Sine/Cos into specific power?
    # Simple correlation with Sine term
    
    design_sine = jnp.array(design[1, :])
    
    corrs = jnp.abs(jnp.corrcoef(S, design_sine)[:-1, -1])
    
    # Rank Components by Correlation
    ranked_indices = jnp.argsort(corrs)[::-1]
    
    # 5. Check Variance (Original Weights)
    # In PICA, S has unit variance (Z-scored roughly by FastICA or PCA whiten).
    # The "Variance" is in the Mixing Matrix A (if unnormalized) or Singular Values.
    # My PICA implementation:
    # X_white = sqrt(N) * Vt
    # S = W @ X_white
    # A = U @ diag(S_val/sqrt(N)) @ W.T
    # The column norms of A represent the contribution (amplitude) of each source.
    
    A_norms = jnp.linalg.norm(A, axis=0) # (K,)
    
    # Sort global ICs by Variance
    variance_rank = jnp.argsort(A_norms)[::-1]
    
    # Find where the "Best Design Match" is in Variance Ranking
    best_design_ic = ranked_indices[0] # The one that looks most like SSVEP
    best_design_corr = corrs[best_design_ic]
    
    # Find its Rank in Variance
    # variance_rank is array of indices [heavy_ic, ..., light_ic]
    # np.where(variance_rank == best_design_ic)
    rank_of_ssvep = float(jnp.where(variance_rank == best_design_ic)[0][0])
    
    print(f"Best Design-Matching IC: {best_design_ic}")
    print(f"Correlation with Design: {best_design_corr:.4f}")
    print(f"Variance Rank of this IC: #{int(rank_of_ssvep)} (out of {S.shape[0]})")
    
    # Compare with Uncorrected? 
    # Previously it was IC 34. (We assume indices might shuffle, but roughly correspond if consistently seeded).
    # But if "Rank" was lower (higher index) before?
    # Actually without calculating variance rank in `demo_pica_ssvep.py`, we don't know exactly.
    # But IC 34 implies it was the 34th component extracted (if FastICA extracts in order? No, FastICA is random/parallel).
    # PCA sorts by variance. 
    # FastICA mixes them.
    # But usually one sorts ICs by explained variance A_norm after extraction.
    # My PICA implementation didn't sort S/A after solving.
    # So "IC 34" is just random index 34.
    
    # However, since my PICA uses PCA whitening first, the subspace is the top K PCA components.
    # The mixing matrix A tells us how much of X is explained by S.
    
    if rank_of_ssvep < 20:
        print("[SUCCESS] SSVEP promoted to Top 20 Variance.")
    else:
        print("[INFO] SSVEP Variance Rank is still lower.")

if __name__ == "__main__":
    demo_optimization()
