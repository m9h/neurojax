"""
Wakeman-Henson Multi-subject Analysis script using SPM 25 module.
"""
import sys
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neurojax.io.wakeman_henson import WakemanHensonLoader
from neurojax.preprocessing.andersen_2018 import run_pipeline
from neurojax.analysis.spm25 import GeneralLinearModel, RandomFieldTheory

def main():
    # 1. Load Data
    loader = WakemanHensonLoader()
    # Note: Requires 'uv run' or pre-downloaded data
    try:
        raw = loader.load_subject(subject="01", run=1)
    except FileNotFoundError:
        print("Data not found. Attempting download (this may take a while)...")
        # loader.download(subject="01") # Uncomment to enable download
        print("Please ensure dataset ds000117 is downloaded to ~/mne_data/ds000117")
        return

    events = loader.load_events(raw)
    
    # Event IDs from dataset
    event_id = {
        'Face/Famous': 5,
        'Face/Unfamiliar': 13,
        'Scrambled': 17
    }
    
    # 2. Preprocess (Andersen 2018)
    print("Running preprocessing...")
    epochs = run_pipeline(raw, events, event_id)
    
    # Average to get ERPs (Evoked)
    evoked_famous = epochs['Face/Famous'].average()
    evoked_scrambled = epochs['Scrambled'].average()
    
    # 3. SPM 25 Analysis (GLM)
    print("Running SPM 25 GLM...")
    
    # Prepare Design Matrix X
    # Simple design: Famous vs Scrambled
    # We will compute GLM on *single trial* data for a specific time point or sensor map
    # For demo, let's take a sensor map at peak latency (e.g. 170ms - N170)
    
    peak_time = 0.170
    data_famous = epochs['Face/Famous'].get_data(tmin=peak_time, tmax=peak_time+0.01).mean(axis=2) # (n_trials, n_sensors)
    data_scrambled = epochs['Scrambled'].get_data(tmin=peak_time, tmax=peak_time+0.01).mean(axis=2) # (n_trials, n_sensors)
    
    Y = np.concatenate([data_famous, data_scrambled], axis=0) # (Total_trials, n_sensors)
    Y = jnp.array(Y)
    
    n_famous = data_famous.shape[0]
    n_scrambled = data_scrambled.shape[0]
    
    # Design Matrix: [Famous, Scrambled] (Cell means model)
    X = jnp.zeros((n_famous + n_scrambled, 2))
    X = X.at[:n_famous, 0].set(1.0)
    X = X.at[n_famous:, 1].set(1.0)
    
    # Fit GLM
    glm = GeneralLinearModel()
    beta = glm.fit(Y, X)
    
    # Contrast: Famous - Scrambled [1, -1]
    contrast = jnp.array([1.0, -1.0])
    
    t_map = glm.compute_stats(Y, X, beta, contrast, stat_type='T')
    
    # 4. RFT Correction
    print("Applying RFT correction...")
    rft = RandomFieldTheory()
    
    # Estimate Resels (Placeholder value for sensor array topology)
    resels = 100.0 
    
    p_corrected = rft.correct_p_values(t_map, resels=resels, D=2)
    
    print(f"Max T-stat: {jnp.max(t_map)}")
    print(f"Min corrected p-value: {jnp.min(p_corrected)}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    mne.viz.plot_topomap(np.array(t_map), epochs.info, axes=axes[0], show=False)
    axes[0].set_title("SPM 25 T-Map (Famous - Scrambled)")
    
    # Plot P-values (1 - p) to show significance
    mne.viz.plot_topomap(1 - np.array(p_corrected), epochs.info, axes=axes[1], show=False, cmap='viridis')
    axes[1].set_title("1 - Corrected P-values")
    
    plt.savefig("wakeman_henson_spm25_results.png")
    print("Results saved to wakeman_henson_spm25_results.png")

import numpy as np # Import numpy for non-JAX ops
if __name__ == "__main__":
    main()
