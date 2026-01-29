import mne
import numpy as np
import jax
import jax.numpy as jnp
from neurojax.glm import GeneralLinearModel, run_permutation_test
from neurojax.utils.bridge import mne_to_jax
import time
import os

def main():
    print("--- NeuroJAX: GLM Permutation Demo ---")
    
    # 1. SETUP: Generate Dummy MNE Data (Simulating an MEG recording)
    info = mne.create_info(ch_names=['MEG%03d' % i for i in range(200)], sfreq=100, ch_types='mag')
    
    # Create a signal: 2 conditions, 30 seconds
    n_times = 3000
    n_sensors = 200
    times = np.linspace(0, 30, n_times)
    
    # Ground truth: Condition A has an effect on sensors 0-10
    design = np.zeros((n_times, 2))
    design[times < 15, 0] = 1  # Condition A
    design[times >= 15, 1] = 1 # Condition B
    
    # Add signal to Condition A
    signal = np.random.randn(n_sensors, n_times)
    signal[0:10, times < 15] += 2.0  # Strong effect
    
    raw = mne.io.RawArray(signal, info)
    
    # 2. BRIDGE: Move to JAX
    print("Loading data into JAX...")
    data_jax = mne_to_jax(raw)
    design_jax = jnp.array(design)
    
    # Define contrast: Condition A - Condition B
    contrast = jnp.array([1.0, -1.0])

    # 3. INIT: Create the OSL-JAX Model
    glm = GeneralLinearModel(design_jax, data_jax)

    # 4. RUN: Permutation Test (The "Killer App")
    n_perms = 5000
    print(f"Running {n_perms} permutations on {n_sensors} sensors...")
    
    start_time = time.time()
    
    # JIT compilation happens on the first call
    key = jax.random.PRNGKey(42)
    t_stats, p_values = run_permutation_test(glm, contrast, key, n_perms=n_perms)
    
    # Force execution for timing
    t_stats.block_until_ready()
    
    end_time = time.time()
    print(f"Done! Elapsed time: {end_time - start_time:.4f} seconds.")

    # 5. RESULTS: Check significant sensors
    sig_sensors = jnp.where(p_values < 0.05)[0]
    print(f"Significant sensors found: {sig_sensors}")
    
    expected = jnp.array([i for i in range(10)])
    print(f"Sensors 0-9 should be significant.")
    
    # 6. DISTRAX: Model Comparison
    fitted_glm = glm.fit()
    log_lik = fitted_glm.log_likelihood()
    print(f"Model Log Likelihood (via Distrax): {log_lik:.2f}")

if __name__ == "__main__":
    main()
