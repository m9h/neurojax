import mne
from mne.datasets import eegbci
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optimistix as optx
from neurojax.spectral import fit_spectrum, PowerSpectrumModel

def load_real_data(subject=1):
    """
    Load EEGBCI data (PhysioNet).
    subject: int, subject id (1-109)
    """
    print(f"Loading EEGBCI data for Subject {subject}...")
    
    # Run 1: Baseline, Eyes Open (Resting State)
    # Run 2: Baseline, Eyes Closed (Strong Alpha)
    runs = [1, 2] 
    raw_fnames = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    
    # Concatenate or treat separately? Let's treat separately.
    # Rest: Eyes Open (Low Alpha, "Alert"?)
    # Task Proxy: Eyes Closed (Huge Alpha, "Relaxed") 
    #   (Note: In typical SSVEP, "Task" has the Peak. Here "Eyes Closed" has the Peak.)
    #   We will treat "Eyes Open" as our BASELINE (Resting Priors)
    #   We will treat "Eyes Closed" as our DATA with PEAKS (Task Loop)
    
    # Standardize montage (64 channel EEG)
    mne.datasets.eegbci.standardize(raws[0])
    mne.datasets.eegbci.standardize(raws[1])
    montage = mne.channels.make_standard_montage('standard_1005')
    raws[0].set_montage(montage)
    raws[1].set_montage(montage)
    
    return raws[0], raws[1]

def compute_psd(raw, fmin=1, fmax=40):
    """Compute Welch PSD for Oz channel (Visual)."""
    # Pick Oz (Visual Cortex)
    raw.pick_channels(['Oz'])
    
    # Welch
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), 
        sfreq=raw.info['sfreq'], 
        fmin=fmin, 
        fmax=fmax, 
        n_fft=256, 
        n_per_seg=256
    )
    
    # psd shape: [channels, freq] -> [1, freq]
    return jnp.array(freqs), jnp.array(psd[0])

def main():
    print("--- NeuroJAX: Real World Spectral Analysis (EEGBCI) ---")
    
    # 1. Load Data
    raw_open, raw_closed = load_real_data(subject=1)
    
    # 2. Compute Spectra (Log Power)
    print("\nComputing PSDs for Oz...")
    freqs, psd_open = compute_psd(raw_open)
    _, psd_closed = compute_psd(raw_closed)
    
    log_psd_open = jnp.log10(psd_open)
    log_psd_closed = jnp.log10(psd_closed)
    
    # 3. Strategy B: Resting-Informed Fit
    # Fit "Eyes Open" (Rest) to get aperiodic parameters
    print("\nFitting Resting State (Eyes Open) for Priors...")
    # fit 1 peak (Alpha might be small) or 0? Usually still has alpha.
    rest_params = fit_spectrum(freqs, log_psd_open, n_peaks=1)
    
    prior_offset = rest_params[0]
    prior_exponent = rest_params[1]
    
    print(f"  > Resting 1/f Exponent: {prior_exponent:.2f}")
    print(f"  > Resting Offset:       {prior_offset:.2f}")
    
    # 4. Compare Fits on "Eyes Closed" (Strong Alpha Peak)
    print("\nFitting 'Task' State (Eyes Closed - Strong Alpha)...")
    
    # Naive Fit
    naive_params = fit_spectrum(freqs, log_psd_closed, n_peaks=1)
    
    # Informed Fit
    def loss_with_prior(params, args):
        freqs, data, prior_exp = args
        model = PowerSpectrumModel()
        model_output = model(freqs, params)
        mse = jnp.mean((model_output - data)**2)
        # Regularization: Heavy penalty on exponent derivation
        reg = 1.0 * (params[1] - prior_exp)**2 
        return mse + reg

    solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
    # Init with prior + Naive peaks
    informed_init = jnp.array([prior_offset, prior_exponent, 10.0, 1.0, 1.0])
    
    sol = optx.least_squares(
        fn=loss_with_prior,
        y0=informed_init,
        args=(freqs, log_psd_closed, prior_exponent),
        solver=solver,
        max_steps=5000,
        throw=False
    )
    informed_params = sol.value
    
    # 5. Results
    print(f"  > Naive Exponent:    {naive_params[1]:.2f}")
    print(f"  > Informed Exponent: {informed_params[1]:.2f}")
    
    # Peak Amplitude (Alpha)
    # Params: [off, exp, cf, pw, bw]
    naive_amp = jax.nn.softplus(naive_params[3])
    informed_amp = jax.nn.softplus(informed_params[3])
    
    print(f"  > Naive Alpha Amp:    {naive_amp:.2f}")
    print(f"  > Informed Alpha Amp: {informed_amp:.2f}")
    
    print("\nVisualizing...")
    model = PowerSpectrumModel()
    
    # Generate full model fits
    fit_naive = model(freqs, naive_params)
    fit_informed = model(freqs, informed_params)
    
    # Generate APERIODIC ONLY fits (to flatten the data)
    # Params: [off, exp, cf, pw, bw] -> We zero out the Gaussian amp
    naive_aperiodic_params = jnp.array([naive_params[0], naive_params[1], 10, -100, 1])
    informed_aperiodic_params = jnp.array([informed_params[0], informed_params[1], 10, -100, 1])
    
    fit_naive_aperiodic = model(freqs, naive_aperiodic_params)
    fit_informed_aperiodic = model(freqs, informed_aperiodic_params)
    
    # Flattened Data (Residuals)
    flat_naive = log_psd_closed - fit_naive_aperiodic
    flat_informed = log_psd_closed - fit_informed_aperiodic

    # Calculate "Area Under Peak" (Sum of positive residuals in Alpha band 8-12Hz)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    auc_naive = jnp.sum(jnp.maximum(0, flat_naive[alpha_mask]))
    auc_informed = jnp.sum(jnp.maximum(0, flat_informed[alpha_mask]))
    
    print(f"\n--- Quantitative Impact ---")
    print(f"Alpha Band (8-13Hz) Excess Power (Area Under Curve):")
    print(f"  > Naive:    {auc_naive:.2f} (Likely overestimated due to tilt)")
    print(f"  > Informed: {auc_informed:.2f} (Corrected for resting 1/f)")
    print(f"  > Difference: {100 * (auc_naive - auc_informed) / auc_informed:.1f}% reduction in spuriously assigned power.")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Standard Log-Log Fit
    ax = axes[0]
    ax.plot(freqs, log_psd_closed, 'k', label='Real Data (Eyes Closed)', alpha=0.5, lw=2)
    ax.plot(freqs, fit_naive, 'r--', label=f'Naive Fit (Exp={naive_params[1]:.2f})')
    ax.plot(freqs, fit_informed, 'b--', label=f'Informed Fit (Exp={informed_params[1]:.2f})')
    ax.plot(freqs, fit_informed_aperiodic, 'g:', label=f'Informed Aperiodic Base (Exp={informed_params[1]:.2f})')
    ax.plot(freqs, fit_naive_aperiodic, 'm:', label=f'Naive Aperiodic Base (Exp={naive_params[1]:.2f})')
    ax.set_title("1. Raw PSD + Model Fits")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log Power")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 2: Flattened Spectra (Peak Visualization)
    ax = axes[1]
    ax.plot(freqs, flat_naive, 'r', label='Naive-Flattened Data', alpha=0.7)
    ax.plot(freqs, flat_informed, 'b', label='Informed-Flattened Data', alpha=0.7)
    ax.axhline(0, color='k', linestyle='-')
    ax.fill_between(freqs, 0, flat_naive, where=alpha_mask, color='r', alpha=0.2, label='Naive Excess Area')
    ax.fill_between(freqs, 0, flat_informed, where=alpha_mask, color='b', alpha=0.2, label='Informed Excess Area')
    
    ax.set_title("2. Flattened Spectra (Data - Aperiodic)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log Power (Above Background)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_data_fit.png')
    print("Saved plot to real_data_fit.png")

if __name__ == "__main__":
    main()
