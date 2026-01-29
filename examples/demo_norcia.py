
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mne
from neurojax.io.cmi import CMILoader
from neurojax.analysis.mca import mca_decompose
from neurojax.analysis.superlet import superlet_transform
from neurojax.analysis.filtering import filter_fft, notch_filter_fft, robust_reference

SUBJECT_ID = "sub-NDARGU729WUR"

def demo_norcia():
    print("=== Norcia Demo: The Alpha-Free VEP ===")
    
    # 1. Load Data
    loader = CMILoader(SUBJECT_ID)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except:
        print("Data not found.")
        return
        
    # 2. Preprocess (Standard)
    # Filter 0.5 - 40Hz
    print("Preprocessing...")
    raw.load_data()
    raw.filter(1, 40, verbose=False) # Use MNE filter for ease on Raw
    
    # 3. Epoching
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # Event: 'contrastChangeB1_start' might be Block start.
    # 'contrastTrial_start': 4 might be the trials.
    print(f"Events: {event_id}")
    
    target_id = None
    # Try targets (visual stimuli)
    if 'left_target' in event_id:
        target_id = event_id['left_target']
    elif 'contrastTrial_start' in event_id:
        target_id = event_id['contrastTrial_start']
            
    if target_id is None:
        print("No contrast event found.")
        return
        
    print(f"Epoching around event ID {target_id}...")
    tmin, tmax = -0.2, 0.6
    # Relax artifact rejection for demo
    reject = dict(eeg=500e-6)
    epochs = mne.Epochs(raw, events, event_id=target_id, tmin=tmin, tmax=tmax, 
                        baseline=(None, 0), reject=reject, preload=True, verbose=False)
    
    print(f"Epochs dropped: {epochs.drop_log_stats()}")
    
    # Pick Occipital channels (Standard VEP: Oz, O1, O2)
    # HBN is GSN-129. 
    # E75=Oz, E70=O1, E83=O2 approx?
    # Let's auto-select 'Oz' if present, or channel with max variance
    # Or just average all 'Occipital' (Back of head)
    
    # Simple strategy: Find channel maximizing the evoked response later.
    # We define ROI later.
    
    data = epochs.get_data() # (N_trials, N_chan, N_time)
    n_trials, n_chan, n_time = data.shape
    times = epochs.times
    sfreq = epochs.info['sfreq']
    
    print(f"Data Shape: {data.shape} ({n_trials} trials)")
    
    # 4. MCA Decomposition (Single Trial Cleaning)
    # Input to MCA: (N_batch, N_time)
    # We reshape to (N_trials * N_chan, N_time) or loop?
    # MCA mca_decompose is vmapped over axis 0.
    
    # Select ROI for demo (One channel: Oz -> E75 approx)
    # Or find max Evoked
    evoked = epochs.average()
    # Find peak channel
    peak_ch_idx = np.argmax(np.max(np.abs(evoked.data), axis=1))
    peak_ch_name = epochs.ch_names[peak_ch_idx]
    print(f"Peak VEP Channel: {peak_ch_name}")
    
    roi_data = data[:, peak_ch_idx, :] # (N_trials, N_time)
    
    print("Running MCA on single trials (Removing Alpha)...")
    # Convert to JAX
    roi_jax = jnp.array(roi_data)
    
    # Thresholds:
    # Alpha (Osc) is sparse in FFT. Lambda_fft should be low enough to capture it, but high enough to ignore noise?
    # VEP (Trans) is sparse in Time. Lambda_ident should capture the P100 peak.
    # Magnitudes: EEG is uV. ~10-50. 
    # Need to normalize?
    scale = jnp.std(roi_jax)
    roi_norm = roi_jax / scale
    
    # Run MCA
    est_osc, est_trans = mca_decompose(
        roi_norm, 
        lambda_fft=1.0,    # Capture strong Alpha
        lambda_ident=1.5,  # Capture strong VEP peaks, ignore small noise
        n_iter=50
    )
    
    # Scale back
    est_osc = est_osc * scale
    est_trans = est_trans * scale
    
    # 5. Analysis
    # Compare "Raw Average" vs "Transient Average" (Cleaned VEP)
    vep_raw = np.mean(roi_data, axis=0) # Evoked
    vep_clean = np.mean(est_trans, axis=0) # De-Alpha'd Evoked
    
    # Check Pre-stimulus Noise (SNR)
    # t < 0
    t0_idx = np.searchsorted(times, 0)
    noise_raw = np.std(vep_raw[:t0_idx])
    peak_raw = np.max(np.abs(vep_raw[t0_idx:]))
    snr_raw = peak_raw / noise_raw
    
    noise_clean = np.std(vep_clean[:t0_idx])
    peak_clean = np.max(np.abs(vep_clean[t0_idx:]))
    snr_clean = peak_clean / noise_clean
    
    print(f"\n--- SNR Results ({peak_ch_name}) ---")
    print(f"Raw VEP SNR: {snr_raw:.2f} (Peak {peak_raw*1e6:.2f} uV)")
    print(f"MCA VEP SNR: {snr_clean:.2f} (Peak {peak_clean*1e6:.2f} uV)")
    print(f"Improvement: {snr_clean/snr_raw:.2f}x")
    
    # 6. Superlet on VEP
    print("\nComputing Superlet on Averaged VEPs...")
    # Add dim for (1, Time)
    freqs = tuple(np.arange(5, 60, 1.0).tolist())
    
    sl_raw = superlet_transform(jnp.array(vep_raw[None, :]), sfreq, freqs, order=5)
    sl_clean = superlet_transform(jnp.array(vep_clean[None, :]), sfreq, freqs, order=5)
    
    # Measure 'Alpha Blur' in TFR
    # Sum power in Alpha band (8-12Hz)
    idx_alpha = [i for i, f in enumerate(freqs) if 8 <= f <= 12]
    alpha_power_raw = jnp.mean(sl_raw[0, idx_alpha, :])
    alpha_power_clean = jnp.mean(sl_clean[0, idx_alpha, :])
    
    print(f"Alpha Power in TFR (Raw): {alpha_power_raw:.2e}")
    print(f"Alpha Power in TFR (Clean): {alpha_power_clean:.2e}")
    print(f"Alpha Reduction: {alpha_power_raw/alpha_power_clean:.1f}x")
    
    # 7. Visualization (ERP Image & JTF)
    print("\nGenering ERP Images and JTF Maps...")
    
    # Create EpochsArray for MNE plotting
    info_roi = mne.create_info([peak_ch_name], sfreq, ch_types='eeg')
    
    # Raw (ROI only)
    epochs_raw = mne.EpochsArray(roi_data[:, None, :], info_roi, tmin=tmin, verbose=False)
    
    # Cleaned (Transient)
    epochs_clean = mne.EpochsArray(est_trans[:, None, :], info_roi, tmin=tmin, verbose=False)
    
    # Alpha (Oscillatory)
    epochs_alpha = mne.EpochsArray(est_osc[:, None, :], info_roi, tmin=tmin, verbose=False)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: ERP Images
    # MNE plot_image returns figures, we need to capture axes or plot manually?
    # plot_image can take axes? No, standard mne plot_image creates fig.
    # We'll use plot_image(show=False) and manually manage if possible,
    # Or just use imshow for simplicity since we have the data.
    
    # Truncate Raw/Times to match Clean (MCA truncated to even)
    n_clean = est_trans.shape[-1]
    roi_data_plot = roi_data[..., :n_clean]
    times_plot = times[:n_clean]
    
    # Sort by nothing (Trial order)
    
    # Clim for ERP Image
    vmax = np.max(np.abs(roi_data_plot))
    vmin = -vmax
    
    def plot_erp_image(ax, data, title):
        # data: (N_trials, N_time)
        im = ax.imshow(data, aspect='auto', origin='lower', 
                       extent=[times_plot[0], times_plot[-1], 0, len(data)],
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trial")
        ax.axvline(0, color='k', linestyle='--')
        return im
        
    plot_erp_image(axes[0, 0], roi_data_plot, f"Raw ERP Image ({peak_ch_name})")
    plot_erp_image(axes[0, 1], est_trans, "MCA Cleaned (Transient)")
    im = plot_erp_image(axes[0, 2], est_osc, "Removed Alpha (Oscillatory)")
    plt.colorbar(im, ax=axes[0, :], fraction=0.02, pad=0.04, label="uV")
    
    # Row 2: JTF Maps (Superlets)
    # sl_raw: (1, Freq, Time)
    # Align shapes
    sl_raw_plot = sl_raw[..., :n_clean]
    sl_clean_plot = sl_clean
    
    def plot_jtf(ax, sl_data, title):
        # sl_data: (Freq, Time) Magnitude
        # Log scale power?
        power = sl_data
        # origin=lower puts freq 0 at bottom
        # extent=[t0, t1, f0, f1]
        extent = [times_plot[0], times_plot[-1], freqs[0], freqs[-1]]
        
        im = ax.imshow(power, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.axvline(0, color='white', linestyle='--')
        return im

    plot_jtf(axes[1, 0], sl_raw_plot[0], "Raw VEP JTF (Superlet)")
    plot_jtf(axes[1, 1], sl_clean_plot[0], "Cleaned VEP JTF (Superlet)")
    
    # Plot Difference
    sl_diff = sl_raw_plot[0] - sl_clean_plot[0]
    plot_jtf(axes[1, 2], sl_diff, "Difference (Removed Energy)")
    
    plt.tight_layout()
    plt.savefig("norcia_vis.png")
    print("Saved visualization to 'norcia_vis.png'")

if __name__ == "__main__":
    demo_norcia()
