import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from neurojax.io.ds004745 import load_ds004745
from neurojax.preprocessing.asr import calibrate_asr, apply_asr

def generate_synthetic_ssvep(duration=60, sfreq=256, n_chans=8):
    """Generates synthetic SSVEP data with artifacts."""
    n_samples = int(duration * sfreq)
    time = np.linspace(0, duration, n_samples)
    
    # Signals: 2, 4, 8 Hz
    freqs = [2, 4, 8]
    signal = np.zeros((n_chans, n_samples))
    
    for i in range(n_chans):
        # Mix frequencies with random phases
        for f in freqs:
            phase = np.random.rand() * 2 * np.pi
            amp = 5.0 # 5 uV signal
            signal[i] += amp * np.sin(2 * np.pi * f * time + phase)
            
    # Add Noise (Pink noise approximation)
    noise = np.random.randn(n_chans, n_samples) * 2.0 
    # (Simple white noise for now, could filter for pink)
    
    data = signal + noise
    
    # Add Artifacts (High amplitude transients)
    # Every 10 seconds, add a 1 sec artifact
    n_artifacts = int(duration / 10)
    for k in range(n_artifacts):
        start = int((k * 10 + 5) * sfreq)
        end = start + int(1.0 * sfreq)
        if end > n_samples: break
        
        # Muscle artifact: High freq, high amp
        artifact = np.random.randn(n_chans, end-start) * 50.0 
        # Drift
        drift = np.linspace(0, 100, end-start)
        
        data[:, start:end] += artifact + drift
        
    return jnp.array(data, dtype=jnp.float32), {'sfreq': sfreq, 'ch_names': [f'Ch{i+1}' for i in range(n_chans)]}

def validate_asr():
    print("Loading Dataset ds004745 (Real Data)...")
    
    # Try loading real data first
    try:
        raw_data, info = load_ds004745('ds004745', subject_id='sub-001')
        print(f"Loaded Real Data: {raw_data.shape}")
    except Exception as e:
        print(f"Real data load failed: {e}")
        print("Detailed error. Check paths. Fallback to synthetic.")
        print("Generating Synthetic SSVEP + Artifact Data...")
        raw_data, info = generate_synthetic_ssvep(duration=120, sfreq=250)
        
    sfreq = info['sfreq']
    ch_names = info['ch_names']
    print(f"Data Ready: {raw_data.shape} @ {sfreq}Hz")
    
    # 1. Identify Calibration Data (First 30s is cleanish in synthetic construction usually,
    # except for the artifact every 10s. We should pick a clean segment.)
    # Synthetic: 0-5s is clean.
    calib_samples = int(30 * sfreq)
    clean_ref = raw_data[:, :calib_samples]
    
    print("Calibrating ASR...")
    # Cutoff 5-10 for synthetic might be strict
    asr_state = calibrate_asr(clean_ref, cutoff=20.0) 
    print(f"Calibration Complete. Cutoff: {asr_state.cutoff}")
    # ... rest of validation logic ...
    
    # 2. Apply ASR
    print("Applying ASR to full dataset...")
    cleaned_data = apply_asr(
        raw_data, 
        asr_state, 
        window_size=int(0.5 * sfreq), 
        step_size=int(0.25 * sfreq)
    )
    print("ASR Complete.")
    
    # 3. Validation Metrics
    # ... (Plotting Logic) ...
    # A. Time Domain Artifact Reduction
    chunk_len = int(2.0 * sfreq)
    n_chunks = raw_data.shape[1] // chunk_len
    
    max_amp = 0
    max_chunk_idx = 0
    
    for i in range(n_chunks):
        chunk = raw_data[:, i*chunk_len : (i+1)*chunk_len]
        amp = jnp.max(chunk) - jnp.min(chunk)
        if amp > max_amp:
            max_amp = amp
            max_chunk_idx = i
            
    print(f"Found artifact segment at chunk {max_chunk_idx} (max amp: {max_amp:.1f} uV)")
    
    # Plot this segment
    start_s = max_chunk_idx * 2.0
    s_idx = int(start_s * sfreq)
    e_idx = s_idx + chunk_len
    
    raw_seg = raw_data[:, s_idx:e_idx]
    clean_seg = cleaned_data[:, s_idx:e_idx]
    time_ax = np.linspace(0, 2.0, chunk_len)
    
    plt.figure(figsize=(12, 6))
    ch_idx = 0 # Synthetic Ch1
    
    plt.plot(time_ax, raw_seg[ch_idx], label='Raw', color='red', alpha=0.7)
    plt.plot(time_ax, clean_seg[ch_idx], label='ASR Cleaned', color='blue', linewidth=2)
    plt.title(f'Artifact Removal Validation: {start_s}s')
    plt.legend()
    plt.savefig('artifact_reduction.png')
    print("Saved artifact_reduction.png")
    
    # B. Frequency Domain SSVEP Preservation
    from scipy.signal import welch
    
    freqs, raw_psd = welch(np.array(raw_data), fs=sfreq, nperseg=int(4*sfreq))
    _, clean_psd = welch(np.array(cleaned_data), fs=sfreq, nperseg=int(4*sfreq))
    
    raw_psd_mean = np.mean(raw_psd, axis=0)
    clean_psd_mean = np.mean(clean_psd, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, raw_psd_mean, label='Raw PSD')
    plt.semilogy(freqs, clean_psd_mean, label='Cleaned PSD')
    plt.xlim(0, 20)
    plt.axvline(2, color='k', linestyle='--', alpha=0.3, label='2 Hz')
    plt.axvline(4, color='k', linestyle='--', alpha=0.3, label='4 Hz')
    plt.axvline(8, color='k', linestyle='--', alpha=0.3, label='8 Hz')
    plt.legend()
    plt.title("SSVEP Spectral Preservation")
    plt.savefig('ssvep_psd.png')
    print("Saved ssvep_psd.png")
    
    # C. Quantitative SNR
    def get_snr(psd, freqs, target_freq):
        idx = np.argmin(np.abs(freqs - target_freq))
        signal_power = psd[idx]
        mask = (freqs > target_freq - 1) & (freqs < target_freq + 1)
        mask[idx] = False
        noise_power = np.mean(psd[mask])
        return 10 * np.log10(signal_power / noise_power)
    
    raw_snr_8 = get_snr(raw_psd_mean, freqs, 8.0)
    clean_snr_8 = get_snr(clean_psd_mean, freqs, 8.0)
    
    print(f"SNR @ 8Hz: Raw={raw_snr_8:.2f}dB, Clean={clean_snr_8:.2f}dB")
    
    with open("validation_results.md", "w") as f:
        f.write("# ASR Validation Report (Synthetic Fallback)\n")
        f.write(f"## SNR Analysis (8 Hz)\n")
        f.write(f"- Raw SNR: **{raw_snr_8:.2f} dB**\n")
        f.write(f"- Cleaned SNR: **{clean_snr_8:.2f} dB**\n")
        f.write(f"- Improvement: **{clean_snr_8 - raw_snr_8:.2f} dB**\n\n")
        f.write("## Visual Validation\n")
        f.write("![Time Domain](artifact_reduction.png)\n")
        f.write("![Frequency Domain](ssvep_psd.png)\n")

if __name__ == "__main__":
    validate_asr()
