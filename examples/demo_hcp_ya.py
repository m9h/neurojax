"""HCP Minimal Pipeline Demo on Real HCP YA MEG Data (PICA + GGMM)."""
import os
import scipy.io
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from neurojax.pipeline.hcp_minimal import hcp_minimal_preproc
from neurojax.analysis.decomposition import probabilistic_ica
from neurojax.analysis.stats import threshold_ggmm

# Helper for Topoplotting (using MNE)
import mne

def load_hcp_mat(fname):
    """Load HCP FieldTrip MAT file and extract data/coords."""
    print(f"Loading {fname}...")
    mat = scipy.io.loadmat(fname, squeeze_me=True, struct_as_record=False)
    data_struct = mat['data']
    
    # Extract params
    sfreq = float(data_struct.fsample)
    ch_names_data = data_struct.label # (248,)
    if isinstance(ch_names_data[0], np.str_):
         ch_names_list = list(ch_names_data)
    else:
        ch_names_list = [str(x).strip() for x in ch_names_data]

    # Extract Trials
    runs = data_struct.trial
    print(f"Found {len(runs)} trials. Concatenating...")
    run_list = [r for r in runs]
    concatenated_data = np.concatenate(run_list, axis=1)
    
    # Extract Geometry
    grad = data_struct.grad
    grad_labels = grad.label
    grad_pos = grad.chanpos
    
    grad_map = {name: i for i, name in enumerate(grad_labels)}
    grad_pos_subset = []
    
    for name in ch_names_list:
        if name in grad_map:
            idx = grad_map[name]
            grad_pos_subset.append(grad_pos[idx])
        else:
            grad_pos_subset.append([0, 0, 1]) 
            
    sensor_coords = np.array(grad_pos_subset)
    norms = np.linalg.norm(sensor_coords, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 
    sensor_coords_unit = sensor_coords / norms
    
    return concatenated_data, sfreq, sensor_coords_unit, ch_names_list

def run_demo():
    fname = 'examples/hcp_data/104012_MEG_10-Motort_tmegpreproc_TFLA.mat'
    if not os.path.exists(fname):
        print(f"Error: {fname} not found.")
        return

    # 1. Load Data
    raw_data, sfreq, coords, ch_names = load_hcp_mat(fname)
    print(f"Data Loaded: {raw_data.shape} @ {sfreq}Hz")
    
    # Crop for PICA speed (take 1 minute of data, roughly 30k samples)
    # PICA on 250 chs x 30k samples is fast.
    n_samples = int(60 * sfreq)
    raw_crop = raw_data[:, :n_samples]
    
    # 2. HCP Minimal Preproc
    print("Running Preprocessing...")
    target_fs = 200.0
    
    raw_jax = jnp.array(raw_crop)
    coords_jax = jnp.array(coords)
    # No bad channels needed for PICA demo, assume cleaned or robust
    bad_channels_jax = jnp.array([]) 
    
    cleaned_jax = hcp_minimal_preproc(
        raw_jax,
        sfreq=sfreq,
        layout_coords=coords_jax,
        bad_channels=bad_channels_jax,
        target_fs=target_fs,
        highpass_freq=0.5
    )
    cleaned = np.array(cleaned_jax) # (n_chs, n_times_new)
    
    # 3. PICA
    print("Running PICA (Dimensionality Est. + FastICA)...")
    # Transpose for PICA? 
    # Usually ICA is X = A S. 
    # X is (features/sensors, samples). S is (components, samples). A is (sensors, components).
    # neurojax `probabilistic_ica` expects (n_features, n_samples).
    
    # We estimate dimension 
    # Let's ask for 15 components to be safe/interesting
    n_comps = 15
    S_z, Mixing, evals = probabilistic_ica(cleaned_jax, n_components=n_comps)
    
    S_z_np = np.array(S_z)
    Mixing_np = np.array(Mixing)
    
    print(f"PICA Converged. Found {n_comps} components.")
    
    # 4. GGMM Thresholding
    print("Thresholding Components (GGMM proxy)...")
    
    # We want to identify the "Motor" component.
    # Since we don't have triggers easily, we look for periodic activity or alpha/beta bursts.
    # Or just show the first 3 components.
    
    # Visualize top 3 components
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Prepare MNE Info for Topomap
    # We need to construct a robust info object from coords
    # This is tricky without a trans file. 
    # Use generic layout?
    # Simpler: Use MNE's `plot_topomap` with `pos` argument.
    # pos needs to be 2D. Project 3D coords to 2D.
    # Azimuthal Equidistant Projection
    
    def project_3d_2d(coords_unit):
        # coords: (n, 3)
        x, y, z = coords_unit.T
        # primitive projection
        theta = np.arctan2(y, x)
        phi = np.arccos(z)
        r = phi / (np.pi/2) 
        x2 = r * np.cos(theta)
        y2 = r * np.sin(theta)
        return np.stack([x2, y2], axis=1)
        
    pos_2d = project_3d_2d(coords)
    
    times_new = np.arange(S_z_np.shape[1]) / target_fs
    
    for i in range(3):
        # Time Course
        ax_time = axes[i, 0]
        ax_time.plot(times_new, S_z_np[i], 'k', linewidth=0.5)
        ax_time.set_title(f"IC {i} Time Course")
        ax_time.set_ylabel("Z-score")
        
        # Power Spectrum
        ax_psd = axes[i, 1]
        from scipy.signal import welch
        f, p = welch(S_z_np[i], fs=target_fs, nperseg=512)
        ax_psd.semilogy(f, p)
        ax_psd.set_xlim(0, 50)
        ax_psd.set_title(f"IC {i} PSD")
        
        # Spatial Map
        ax_map = axes[i, 2]
        
        # Get map from Mixing matrix column
        spatial_map = Mixing_np[:, i]
        
        # Apply Thresholding
        z_robust, _, _ = threshold_ggmm(jnp.array(spatial_map))
        z_robust = np.array(z_robust)
        
        # Plot Topomap using MNE
        # We use mne.viz.plot_topomap
        im, _ = mne.viz.plot_topomap(z_robust, pos_2d, axes=ax_map, show=False, contours=0)
        ax_map.set_title(f"IC {i} Spatial (Z > 3.0)")
        plt.colorbar(im, ax=ax_map)
        
    plt.tight_layout()
    plt.savefig('hcp_pica_results.png')
    print("Saved hcp_pica_results.png")

if __name__ == "__main__":
    run_demo()
