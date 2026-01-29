"""
ASR Face Processing Demo (ds002718)
===================================
This script demonstrates Artifact Subspace Reconstruction (ASR) on real EEG data
from the "Face processing EEG dataset" (ds002718).

It performs the following steps:
1. Downloads subject 002 data from OpenNeuro (if not already present).
2. Loads the raw EEG data using MNE-Python.
3. Applies a standard 1Hz Highpass filter (crucial for ASR/ICA).
4. Calibrates ASR on the data.
5. Applies ASR to remove artifacts (blinks, muscle, etc.).
6. Visualizes the results.
"""

import os
import glob
import openneuro
import mne
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neurojax.preprocessing.asr import calibrate_asr, apply_asr

# Ensure JAX is on CPU for simple demo if GPU not available/needed
jax.config.update("jax_platform_name", "cpu")

def download_data(dataset_id='ds002718', subject='sub-002'):
    target_dir = os.path.join('downloads', dataset_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if data exists
    subject_dir = os.path.join(target_dir, subject)
    if os.path.exists(subject_dir):
        print(f"Data for {subject} found in {subject_dir}. Skipping download.")
        return target_dir
        
    print(f"Downloading {dataset_id} ({subject}) to {target_dir}...")
    openneuro.download(dataset=dataset_id, target_dir=target_dir, include=[subject])
    return target_dir

def load_data(data_dir, subject='sub-002'):
    # Find the EEG file
    # BIDS structure: sub-002/eeg/sub-002_task-face_eeg.set (usually EEGLAB format for this dataset)
    # or .bdf / .vhdr
    search_path = os.path.join(data_dir, subject, 'eeg', '*.set')
    files = glob.glob(search_path)
    if not files:
        # Try .bdf or .vhdr
        files = glob.glob(os.path.join(data_dir, subject, 'eeg', '*.bdf'))
        
    if not files:
        raise FileNotFoundError(f"No EEG files found in {search_path}")
        
    file_path = files[0]
    print(f"Loading {file_path}...")
    
    # Load raw data
    # Preload to memory
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw

def process_and_plot():
    # 1. Get Data
    data_dir = download_data()
    raw = load_data(data_dir)
    
    # 2. Preprocess
    # ASR requires stationarity, so Highpass is mandatory.
    # 1.0 Hz is standard for ASR/ICA to remove drift.
    print("Filtering (1.0 Hz Highpass)...")
    raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin')
    
    # Pick EEG channels
    picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False, stim=False)
    raw.pick_channels([raw.ch_names[p] for p in picks])
    
    # Convert to JAX array
    # (n_channels, n_times)
    data_np = raw.get_data()
    data_jax = jnp.array(data_np)
    
    print(f"Data Shape: {data_jax.shape}")
    
    # 3. ASR Calibration
    # Ideally, we find a "clean" section. 
    # For this demo, we use a robust calibration on the whole file 
    # (assuming the median/robust stats in calibration clean it up).
    # Since our simplified `calibrate_asr` uses standard Covariance, 
    # it might be sensitive to artifacts. 
    # Let's try to pick a cleaner start segment if possible, or just run it.
    # We'll use the first 60 seconds.
    print("Calibrating ASR (using first 60s as reference)...")
    sfreq = raw.info['sfreq']
    n_calib = int(60 * sfreq)
    calib_data = data_jax[:, :n_calib]
    
    asr_state = calibrate_asr(calib_data, cutoff=20.0) # Conservative cutoff for real data
    
    # 4. Apply ASR
    print("Applying ASR...")
    # Window size: 0.5s is typical
    win_len = int(0.5 * sfreq)
    step_len = int(win_len / 2)
    
    cleaned_jax = apply_asr(data_jax, asr_state, window_size=win_len, step_size=step_len)
    
    # 5. Visualization
    print("Plotting results...")
    
    # Create MNE objects for plotting
    raw_cleaned = raw.copy()
    raw_cleaned._data = np.array(cleaned_jax)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # A. Time Series (Snapshot)
    # Find a segment with potential artifacts (e.g. blink)
    # Arbitrary search or just plot a known segment
    start_t = 100.0
    duration = 5.0
    start_samp = int(start_t * sfreq)
    stop_samp = int((start_t + duration) * sfreq)
    times = raw.times[start_samp:stop_samp]
    
    # Pick a frontal channel (Fp1 or similar) usually impacted by blinks
    ch_name = raw.ch_names[0] # Usually Fp1 is early in list
    ch_idx = 0
    
    axes[0, 0].plot(times, data_np[ch_idx, start_samp:stop_samp] * 1e6, label='Raw', alpha=0.7)
    axes[0, 0].plot(times, np.array(cleaned_jax[ch_idx, start_samp:stop_samp]) * 1e6, label='ASR Cleaned', color='k')
    axes[0, 0].set_title(f"Time Series: {ch_name} (Snapshot)")
    axes[0, 0].set_ylabel("Amplitude (uV)")
    axes[0, 0].legend()
    
    # B. Removed Content
    removed = data_np[ch_idx, start_samp:stop_samp] - np.array(cleaned_jax[ch_idx, start_samp:stop_samp])
    axes[0, 1].plot(times, removed * 1e6, label='Removed Component', color='red')
    axes[0, 1].set_title(f"Removed Artifact: {ch_name}")
    axes[0, 1].legend()
    
    # C. PSD Comparison
    # We compute PSD for the whole file
    print("Computing PSD...")
    raw.compute_psd(fmax=80).plot(axes=axes[1, 0], show=False, average=True, spatial_colors=False)
    axes[1, 0].set_title("PSD - Raw Data")
    
    raw_cleaned.compute_psd(fmax=80).plot(axes=axes[1, 1], show=False, average=True, spatial_colors=False)
    axes[1, 1].set_title("PSD - ASR Cleaned")
    
    plt.tight_layout()
    plt.savefig('asr_face_processing.png')
    print("Saved plot to asr_face_processing.png")

if __name__ == "__main__":
    process_and_plot()
