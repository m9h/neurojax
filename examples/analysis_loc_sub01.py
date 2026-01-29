
"""
Analysis of Sleep Onset (Wake -> NREM) stability using SINDy-based bifurcation analysis.
Dataset: ds003768 (Sleep Deprivation), Subject 01.
Tracks dynamical stability (max real eigenvalue) of the EEG manifold over time.
"""

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import mne
from sklearn.decomposition import PCA
from neurojax.dynamics.sindy import SINDyOptimizer, polynomial_library
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("/home/mhough/dev/neurojax/downloads/ds003768/sub-01/eeg")
OUTPUT_DIR = Path("results_loc_sub01")
TASK_FILE = "sub-01_task-sleep_run-1_eeg.vhdr"

WINDOW_SEC = 30.0
STEP_SEC = 5.0
SINDY_THRESHOLD = 0.05
POLY_DEGREE = 2
PCA_COMPONENTS = 3

# --- Helpers ---
def get_jacobian_fn(xi: jax.Array, degree: int = 2):
    """
    Returns a function J(z) that computes the Jacobian of the learned dynamics f(z) at state z.
    f(z) = Theta(z) @ Xi
    """
    def model_func(z):
        # z shape (dim,)
        # library expects (1, dim)
        theta = polynomial_library(z[None, :], degree=degree) # shape (1, n_lib)
        # Xi shape (n_lib, dim)
        dz = theta @ xi
        return dz[0] # (dim,)

    return jax.jacfwd(model_func)

def process_window(window_data, dt, optimizer):
    """
    Fits SINDy and computes stability metric (max real eigenvalue).
    """
    X = jnp.array(window_data)
    # Numerical differentiation
    dX = jnp.gradient(X, axis=0) / dt
    
    # Normalize for numerical stability in SINDy
    x_std = jnp.std(X, axis=0) + 1e-6
    X_norm = X / x_std
    dX_norm = dX / x_std
    
    # Fit
    Xi_norm = optimizer.fit(X_norm, dX_norm, lambda x: polynomial_library(x, degree=POLY_DEGREE))
    
    # Compute Jacobian at the centroid of the window
    z_center = jnp.mean(X_norm, axis=0)
    jac_fn = get_jacobian_fn(Xi_norm, degree=POLY_DEGREE)
    J = jac_fn(z_center)
    
    eigvals = jnp.linalg.eigvals(J)
    max_real_eig = jnp.max(jnp.real(eigvals))
    
    return max_real_eig, Xi_norm

# --- Main Pipeline ---
def main():
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
        
    vhdr_path = DATA_DIR / TASK_FILE
    if not vhdr_path.exists():
        print(f"ERROR: File {vhdr_path} not found.")
        print("Please ensure dataset ds003768 is downloaded to downloads/ds003768")
        return

    print(f"Loading {vhdr_path}...")
    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Basic Preprocessing
    print("Preprocessing...")
    raw.set_eeg_reference('average', projection=False)
    raw.filter(l_freq=0.5, h_freq=40.0, n_jobs=4) # 0.5Hz highpass for SINDy stability
    
    # Resample
    if raw.info['sfreq'] > 250.0:
        print(f"Resampling from {raw.info['sfreq']} to 250Hz...")
        raw.resample(250.0)
    
    SFREQ = raw.info['sfreq']
    print(f"Sampling Rate: {SFREQ} Hz")
    
    # Extract data (pick Posterior channels often good for Alpha/Sleep)
    # Just picking all EEG for PCA
    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False)
    data = raw.get_data(picks=picks).T
    print(f"Data shape: {data.shape}")
    
    # PCA
    print(f"PCA reduction to {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS)
    z_traj = pca.fit_transform(data)
    
    # Sliding Window Analysis
    n_samples = z_traj.shape[0]
    win_samples = int(WINDOW_SEC * SFREQ)
    step_samples = int(STEP_SEC * SFREQ)
    
    optimizer = SINDyOptimizer(threshold=SINDY_THRESHOLD)
    dt = 1.0 / SFREQ
    
    results = []
    time_points = []
    
    optimizer_jitted = eqx.filter_jit(process_window)
    
    print("Starting SINDy sliding window analysis...")
    # Limit number of windows for fast demo if needed, but let's run full run-1 (usually 10-20 mins)
    # Sleep Onset might happen in first 10 mins.
    
    for start_idx in range(0, n_samples - win_samples, step_samples):
        end_idx = start_idx + win_samples
        window_z = z_traj[start_idx:end_idx]
        t_center = (start_idx + end_idx) / 2.0 / SFREQ
        
        max_lambda, _ = optimizer_jitted(window_z, dt, optimizer)
        
        results.append(float(max_lambda))
        time_points.append(t_center)
        
        if len(results) % 20 == 0:
            print(f"t={t_center:.1f}s, lambda_max={max_lambda:.4f}")

    # Results processing
    results_arr = np.array(results)
    time_arr = np.array(time_points)
    metric_smooth = np.convolve(results_arr, np.ones(5)/5, mode='same')
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time_arr, results_arr, label='Max Re(lambda)', alpha=0.4)
    plt.plot(time_arr, metric_smooth, 'k-', linewidth=2, label='Smoothed')
    plt.axhline(0, color='r', linestyle='--', alpha=0.3)
    
    plt.title("Sleep Onset Stability Analysis: Max Real Eigenvalue")
    plt.xlabel("Time (s)")
    plt.ylabel("Max Real Eigenvalue")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "loc_trajectory_sub01.png")
    print(f"Plot saved to {OUTPUT_DIR / 'loc_trajectory_sub01.png'}")
    
    # Save results
    df_res = pd.DataFrame({"time": time_arr, "lambda_max": results_arr, "lambda_smooth": metric_smooth})
    df_res.to_csv(OUTPUT_DIR / "analysis_results.csv", index=False)

if __name__ == "__main__":
    main()
