
import os
import mne
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from neurojax.dynamics import SINDyOptimizer
from scipy.signal import savgol_filter

def load_and_preprocess_sleep_data():
    """
    Loads one run of sleep deprivation data, extracts Delta power.
    Returns:
        t_epochs: Time vector (hours)
        delta_power: Normalized delta power trajectory x(t)
        pressure_proxy: Modeled sleep pressure u(t)
    """
    # Path to verified file
    data_path = "/home/mhough/dev/neurojax/downloads/ds003768/sub-01/eeg/sub-01_task-sleep_run-1_eeg.vhdr"
    
    print(f"Loading EEG data from {data_path}...")
    # Load raw data (preload needed for filtering)
    raw = mne.io.read_raw_brainvision(data_path, preload=True, verbose=False)
    
    # Basic Preprocessing
    # Filter 0.5 - 40 Hz
    raw.filter(0.5, 40.0, n_jobs=1, verbose=False)
    
    # Re-reference to average (common for high-density, but this is 60ch approx)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    
    # Extract Delta Power in 30s epochs (standard sleep scoring window)
    sfreq = raw.info['sfreq']
    epoch_duration = 30.0 # seconds
    
    # Get data matrix (Channels x Time)
    data, times = raw.get_data(return_times=True)
    
    print(f"Data shape: {data.shape} (Channels, Timepoints)")
    
    # We will average power across all channels to get a global 'Sleep Depth' metric
    # In reality, frontal channels are best for Delta, but global avg is robust.
    
    # Window parameters
    n_samples_per_epoch = int(epoch_duration * sfreq)
    n_epochs = data.shape[1] // n_samples_per_epoch
    
    print(f"Extracting {n_epochs} epochs of {epoch_duration}s duration...")
    
    delta_powers = []
    
    # Frequency bands
    fmin, fmax = 1.0, 4.0 # Delta
    
    # Iterate epochs (simple reshaping would be faster but let's be explicit)
    # Using welch's method for PSD
    from scipy.signal import welch
    
    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        end = start + n_samples_per_epoch
        epoch_data = data[:, start:end]
        
        # Compute PSD (Channels x Freqs)
        freqs, psd = welch(epoch_data, fs=sfreq, nperseg=int(2*sfreq), axis=1)
        
        # Mask for Delta band
        idx_delta = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Mean power in delta band, averaged across channels
        # Shape: (Channels,) -> scalar
        power = np.mean(psd[:, idx_delta]) # Average over freqs and channels
        delta_powers.append(power)
        
    delta_powers = np.array(delta_powers)
    
    # Normalize Delta Power (x) to range [0, 1] or Z-score for numerical stability in SINDy
    # Let's clean it first: Remove artifacts (outliers)
    # Simple percentile clipping
    p95 = np.percentile(delta_powers, 95)
    delta_powers = np.clip(delta_powers, 0, p95)
    
    # Smooth the trajectory to get smooth derivatives
    # Savitzky-Golay filter
    window_length = 15 # epochs ~ 7.5 mins
    if window_length % 2 == 0: window_length += 1
    x_smooth = savgol_filter(delta_powers, window_length=window_length, polyorder=3)
    
    # Normalize mean to 1.0 for interpretable coefficients
    x_normalized = x_smooth / np.mean(x_smooth)
    
    # Construct Time Vector (hours)
    t_epochs = np.arange(n_epochs) * (epoch_duration / 3600.0) # Hours
    
    # Construct "Sleep Pressure" u(t)
    # Assumption: User starts with HIGH sleep pressure which decays exponentially during sleep task.
    # u(t) = exp(-t / tau)
    # Tau for Process S is approx 4-6 hours? Let's assume 4.0h.
    # Note: If valid data is < 1 hour, this might look linear.
    # The run duration is ~ 60 mins -> t ranges 0 to 1.
    # This is short used for a full exponential fit, but allows testing curvature.
    
    tau_decay = 2.0 # made quicker to see effect in 1 hour run
    u_pressure = np.exp(-t_epochs / tau_decay)
    
    # Add some random perturbation to u so it's not perfectly collinear with time? 
    # No, Process S is a theoretical construct, it is smooth.
    
    return t_epochs, x_normalized, u_pressure

def sleep_pressure_library(X_combined):
    """
    SINDy Library for Sleep Homeostasis
    Columns of X_combined: [x, u]
    """
    x = X_combined[:, 0:1]
    u = X_combined[:, 1:2]
    
    # 1. Bias
    bias = jnp.ones_like(x)
    
    # 2. Linear Terms
    # x, u are base
    
    # 3. Interactions
    x_u_linear = x * u
    
    # 4. Saturating Interaction (Michaelis-Menten)
    # u / (0.5 + u) -> Assuming Km is around 0.5 (half-max pressure)
    # Since u goes from 1.0 -> 0.6 in 1 hour, 0.5 is a reasonable pivot.
    x_u_sat = x * (u / (0.5 + u))
    
    return jnp.concatenate([bias, x, u, x_u_linear, x_u_sat], axis=1)

def recover_sleep_pressure():
    print("--- 1. LOADING & PREPROCESSING REAL EEG DATA ---")
    t, x, u = load_and_preprocess_sleep_data()
    
    print(f"Loaded {len(t)} epochs ({t[-1]:.2f} hours).")
    print(f"Mean Delta Power: {np.mean(x):.4f}")
    
    # Prepare Data for SINDy
    # We need dX/dt. Since data is real and noisy, we compute robust derivatives.
    dt = t[1] - t[0] # Time step in hours
    
    # Finite difference is noisy. Use spectral or TVDiff if available. 
    # We used savgol smoothing earlier, so simple diff is okay.
    dx = np.gradient(x, dt)
    
    # Smooth derivative again
    dx_smooth = savgol_filter(dx, window_length=11, polyorder=3)
    
    # Convert to JAX arrays
    X_combined = jnp.stack([x, u], axis=1)
    dX_target = jnp.expand_dims(dx_smooth, axis=1)
    
    print("\n--- 2. SINDY WITH CONTROL RECOVERY ---")
    
    # Optimizer
    # Real data requires higher threshold to ignore noise
    optimizer = SINDyOptimizer(threshold=0.05) 
    
    print("Fitting SINDy model to Real Sleep Data...")
    
    Xi = optimizer.fit(X_combined, dX_target, sleep_pressure_library)
    
    feature_names = ["1", "x", "u", "x*u", "x*u/(0.5+u)"]
    # library columns:
    # 0: 1
    # 1: x
    # 2: u
    # 3: x*u
    # 4: x*u/(0.5+u)
    
    print("\n--- Discovered Coefficients (Target: d(Delta)/dt) ---")
    coeffs = Xi[:, 0]
    
    for name, val in zip(feature_names, coeffs):
        print(f"{name:>12}: {val:.4f}")

    print("\n--- 3. VERIFICATION & ANALYSIS ---")
    
    coeff_linear_int = coeffs[3]
    coeff_sat_int = coeffs[4]
    
    print(f"Linear Coeff: {coeff_linear_int}")
    print(f"Saturating Coeff: {coeff_sat_int}")
    
    if abs(coeff_sat_int) > abs(coeff_linear_int):
        print("SUCCESS: Saturating dynamics dominant.")
        Emax_est = -coeff_sat_int
        print(f"Estimated Efficacy Emax: {Emax_est:.4f}")
    else:
        print("RESULT: Linear dynamics dominant (or indistinguishable).")
        print("Note: In short sleep runs, linear approximation is often sufficient.")

if __name__ == "__main__":
    recover_sleep_pressure()
