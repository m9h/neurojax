import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from neurojax.spectral import PowerSpectrumModel

def main():
    print("--- NeuroJAX: Visualizing the Spectral Slope Confound ---")
    
    # 1. Define Two Settings
    # Control: Steep Slope (Exp = 1.6), Offset = 2.0
    # Patient: Flat Slope (Exp = 1.1), Offset = 2.0
    # BOTH have Identical Alpha Peak: Amp=1.0 at 10Hz
    
    freqs = jnp.linspace(1, 40, 200)
    model = PowerSpectrumModel()
    
    # [off, exp, cf, pw, bw]
    # Note: PowerSpectrumModel adds Gaussian on top of Aperiodic.
    # Aperiodic L(f) = b - log(k + f^chi)
    # To compare visual "height", we should ensure they intersect or have similar scaling.
    # Usually flattest slope = higher high-freq power.
    
    params_control = jnp.array([2.0, 1.6, 10.0, 1.0, 1.5])
    params_patient = jnp.array([2.0, 1.1, 10.0, 1.0, 1.5]) # Same Peak Amp!
    
    psd_control = model(freqs, params_control)
    psd_patient = model(freqs, params_patient)
    
    # Aperiodic Only (Background)
    params_control_ap = jnp.array([2.0, 1.6, 10.0, -100.0, 1.5])
    params_patient_ap = jnp.array([2.0, 1.1, 10.0, -100.0, 1.5])
    
    ap_control = model(freqs, params_control_ap)
    ap_patient = model(freqs, params_patient_ap)
    
    # Alpha Mask (8-12 Hz)
    mask = (freqs >= 8) & (freqs <= 12)
    
    # Create Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Panel 1: The Raw Data (The Confusion) ---
    ax = axes[0]
    ax.plot(freqs, psd_control, 'b-', label='Control (Steep)', lw=2)
    ax.plot(freqs, psd_patient, 'r-', label='Patient (Flat)', lw=2)
    ax.set_title("1. Raw Power Spectra\n(Identical Oscillations!)", fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log Power")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Annotate the gap at 10Hz
    # Show that Red is "Higher" than Blue purely due to slope
    idx_10 = np.argmin(np.abs(freqs - 10.0))
    y_c = psd_control[idx_10]
    y_p = psd_patient[idx_10]
    ax.plot([10, 10], [y_c, y_p], 'k-', lw=1)
    ax.text(11, (y_c+y_p)/2, "Patient has\nHigher Absolute Power", fontsize=10, color='k')
    
    # --- Panel 2: The Naive Analysis (Band Power) ---
    ax = axes[1]
    ax.plot(freqs, psd_control, 'b-', alpha=0.3)
    ax.plot(freqs, psd_patient, 'r-', alpha=0.3)
    
    # Fill Area under curve (linear power proxy visualization)
    # We shade the region between y=-2 (arbitrary floor) and the curve
    floor = np.min(psd_control) - 0.5
    
    ax.fill_between(freqs, floor, psd_control, where=mask, color='b', alpha=0.3, label='Control Band Power')
    ax.fill_between(freqs, floor, psd_patient, where=mask, color='r', alpha=0.3, label='Patient Band Power')
    
    ax.set_title("2. Naive Analysis (Band Power)\n(False Positive Difference)", fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_yticks([])
    ax.legend(loc='lower left')
    ax.text(15, floor + 1.0, "Patient Area >> Control Area\n(Due to risen noise floor)", fontsize=11, fontweight='bold')
    
    # --- Panel 3: The Accurate Analysis (Flattened) ---
    ax = axes[2]
    
    # Flattened = Data - Aperiodic
    flat_control = psd_control - ap_control
    flat_patient = psd_patient - ap_patient
    
    ax.plot(freqs, flat_control, 'b-', label='Control (Flattened)', lw=2)
    ax.plot(freqs, flat_patient, 'r--', label='Patient (Flattened)', lw=2) # Dashed to show overlap
    
    ax.set_title("3. NeuroJAX Analysis (Flattened)\n(True Null Effect)", fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power vs Aperiodic")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.5)
    
    ax.text(12, 1.0, "Perfect Overlap!\nPeaks are Identical", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spectral_confound_viz.png')
    print("Saved visualization to spectral_confound_viz.png")

if __name__ == "__main__":
    main()
