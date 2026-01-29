
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import mne
import numpy as np
import matplotlib.pyplot as plt
from neurojax.io.cmi import CMILoader

SUBJECT_ID = "sub-NDARGU729WUR"

def analyze_ssvep():
    print(f"\n--- Analyzing Task Data for SSVEP ({SUBJECT_ID}) ---")
    loader = CMILoader(SUBJECT_ID)
    
    # Load Contrast Change Detection (Potential SSVEP)
    try:
        raw = loader.load_task("contrastChangeDetection", run=1)
    except Exception as e:
        print(f"Failed to load task: {e}")
        return

    print(raw.info)
    
    # 1. Events
    events, event_id = mne.events_from_annotations(raw)
    print(f"Events found: {len(events)}")
    print(event_id)
    
    # 2. Global PSD (Check for flicker peaks)
    print("Computing Global PSD to finding SSVEP frequencies...")
    raw.compute_psd(fmax=60).plot(average=True, show=False)
    # returning figure, but running in headless. 
    # Let's compute values and print peaks.
    psd = raw.compute_psd(fmax=60)
    psds, freqs = psd.get_data(return_freqs=True)
    mean_psd = np.mean(psds, axis=0) # Average across channels
    
    # Find peaks in mean PSD
    # Simple check: print high power freqs
    log_psd = np.log10(mean_psd)
    
    # Check specific SSVEP candidates (e.g. 6Hz, 10Hz, 12Hz, 15Hz, 20Hz)
    candidates = [6, 10, 12, 15, 20, 24, 30]
    print("\nPower at Potential SSVEP Freqs:")
    for f in candidates:
        # Find closest index
        idx = np.argmin(np.abs(freqs - f))
        power = log_psd[idx]
        
        # Local SNR (vs neighbors +/- 1Hz)
        idx_low = np.argmin(np.abs(freqs - (f-1)))
        idx_high = np.argmin(np.abs(freqs - (f+1)))
        neighborhood = np.concatenate([log_psd[idx_low:idx-2], log_psd[idx+3:idx_high]])
        noise = np.mean(neighborhood)
        
        snr = power - noise
        print(f" {f} Hz: Power={power:.2f}, SNR={snr:.2f} dB")
        
        if snr > 0.2: # Threshold for 'Peak'
             print(f"  --> POSSIBLE SSVEP PEAK at {f}Hz")

if __name__ == "__main__":
    analyze_ssvep()
