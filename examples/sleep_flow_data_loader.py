
import os
import glob
import numpy as np
import mne
import jax.numpy as jnp
from scipy.signal import welch
from scipy.integrate import trapezoid

def load_and_process_data(subject_id='sub-01', data_dir='downloads/ds003768'):
    """
    Loads EEG data for a specific subject, preprocesses it, and extracts band powers.
    """
    # 1. Find Data File
    search_path = os.path.join(data_dir, subject_id, '**', '*.vhdr')
    files = glob.glob(search_path, recursive=True)
    if not files:
        # Try .edf if .vhdr not found
        search_path = os.path.join(data_dir, subject_id, '**', '*.edf')
        files = glob.glob(search_path, recursive=True)
    
    if not files:
        raise FileNotFoundError(f"No EEG data found for {subject_id} in {data_dir}")
    
    raw_path = files[0]
    print(f"Loading: {raw_path}")
    
    # 2. Load Data
    try:
        raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose=False)
    except:
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        
    # 3. Preprocessing
    # Resample to 100Hz as per skill
    raw.resample(100)
    
    # Filter (0.5 - 45 Hz)
    raw.filter(0.5, 45, verbose=False)
    
    # Select channels (Use common ones like C3, C4, O1, O2 if available, or just all)
    # providing a simple selection if specific channels aren't known, usually central/occipital are good for sleep
    available_chs = raw.ch_names
    target_chs = ['C3', 'C4', 'O1', 'O2', 'Cz', 'Pz']
    picks = [ch for ch in target_chs if ch in available_chs]
    if not picks:
        picks = 'eeg'
    
    print(f"Using channels: {picks}")
    raw.pick(picks)
    
    # 4. Extract Band Powers
    # Bands: Delta (0.5-4), Theta (4-8), Alpha (8-13), Sigma (12-16), Beta (16-30)
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Sigma': (12, 16),
        'Beta': (16, 30)
    }
    
    # Epoching: 30s epochs, but for continuous flow maybe sliding window?
    # Skill says: "every 30s epoch (or finer 5s resolution for continuity)"
    # Let's use 5s sliding window with some overlap or just non-overlapping 5s for smoother flow?
    # Actually, standard sleep scoring is 30s. For continuous flow, 30s is a bit coarse.
    # Let's use 30s window sliding every 5s.
    
    data = raw.get_data() # (n_channels, n_times)
    sfreq = raw.info['sfreq']
    
    window_size_sec = 30
    step_size_sec = 5
    window_samples = int(window_size_sec * sfreq)
    step_samples = int(step_size_sec * sfreq)
    
    n_samples = data.shape[1]
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    band_powers = []
    
    print(f"Extracting features from {n_windows} windows...")
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        segment = data[:, start:end]
        
        # Welch's method
        freqs, psd = welch(segment, fs=sfreq, nperseg=int(2*sfreq)) # 2s segments for Welch
        
        # Average across channels
        psd_mean = np.mean(psd, axis=0)
        
        # Integrate bands
        powers = []
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            power = trapezoid(psd_mean[idx], freqs[idx])
            powers.append(power)
        
        band_powers.append(powers)
        
    band_powers = np.array(band_powers) # (n_windows, 5)
    
    # 5. Log-transform and Z-score
    log_powers = np.log(band_powers + 1e-6)
    
    # Z-score normalization
    mean = np.mean(log_powers, axis=0)
    std = np.std(log_powers, axis=0)
    norm_log_powers = (log_powers - mean) / std
    
    print(f"Feature extraction complete. Shape: {norm_log_powers.shape}")
    
    return jnp.array(norm_log_powers), mean, std

if __name__ == "__main__":
    load_and_process_data()
