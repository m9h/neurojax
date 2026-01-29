import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import optimistix as optx
from neurojax.spectral import fit_spectrum, PowerSpectrumModel
import pandas as pd

# JIT-compiled generator for a whole group
# n_subjects (arg 1) must be static for shape generation
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def generate_group_spectra(key, n_subjects, exponent_mean, alpha_amp_mean=1.0):
    # Parameters for each subject
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Shape: (n_subjects,)
    exponents = exponent_mean + jax.random.normal(k1, (n_subjects,)) * 0.1
    alpha_amps = alpha_amp_mean + jax.random.normal(k2, (n_subjects,)) * 0.1
    
    offset = 2.0 
    
    # Construct params array: [offset, exponent, cf, pw, bw]
    # We need to stack them.
    # cf=10, bw=1.0 fixed
    cfs = jnp.full((n_subjects,), 10.0)
    bws = jnp.full((n_subjects,), 1.0)
    offsets = jnp.full((n_subjects,), offset)
    
    # Stack: (n_subjects, 5)
    true_params_batch = jnp.stack([offsets, exponents, cfs, alpha_amps, bws], axis=1)
    
    model = PowerSpectrumModel()
    freqs = jnp.linspace(1, 40, 100)
    
    # Vectorize model call over params [Batch, 5] (freqs broadcast)
    # Model expects (freqs, params). We want vmap over params.
    def get_spectrum(p):
        return model(freqs, p)
        
    clean_psds = jax.vmap(get_spectrum)(true_params_batch)
    
    # Add noise
    noise = jax.random.normal(k3, shape=clean_psds.shape) * 0.1
    data_psds = clean_psds + noise
    
    return freqs, data_psds, true_params_batch

# Naive Metric: Band Power
@jax.jit
def get_band_power_batch(freqs, psds, fmin=8, fmax=12):
    # psds: (n_subjects, n_freqs) log power
    mask = (freqs >= fmin) & (freqs <= fmax)
    # Convert to linear
    linear_psds = jnp.power(10.0, psds)
    # Sum over freqs (axis=1)
    # Boolean indexing is not JIT-compatible due to dynamic shape.
    # Use where to mask out non-band freqs
    masked_psds = jnp.where(mask, linear_psds, 0.0)
    
    # Mean power in band = Sum(masked) / Count(mask)
    return jnp.sum(masked_psds, axis=1) / jnp.sum(mask)

# Informed Metric: Fit Spectrum
# We wrap fit_spectrum to be vmap-able
# fit_spectrum(freqs, spectrum, n_peaks=1)
def fit_batch(freqs, psds):
    def fit_single(psd):
        return fit_spectrum(freqs, psd, n_peaks=1)
    
    # vmap over psds (axis 0)
    params_batch = jax.vmap(fit_single)(psds)
    
    # Extract Alpha Amp (idx 3 is Log Power, need Softplus)
    # Wait, fit_spectrum returns the raw params.
    # Model uses softplus inside.
    # We want the 'Amplitude' parameter (idx 3).
    # Ideally we'd return jax.nn.softplus(params[:, 3])
    # But let's verify indices. 0=off, 1=exp, 2=cf, 3=pw, 4=bw
    return jax.nn.softplus(params_batch[:, 3])

# JIT the fitting batch?
# optimistix compiles inside, but vmap helps.
fit_batch_jit = jax.jit(fit_batch)

def run_simulation():
    print("--- NeuroJAX: Group Power Analysis Comparison (Optimized) ---")
    print("Scenario: Control (Exp=1.5) vs Patient (Exp=1.2). True Alpha Diff = 0.")
    print("hypothesis: Naive BandPower will be significant (FP). Informed will be null.")
    
    sample_sizes = [10, 20, 30, 40, 50]
    n_sims = 100 # Fast enough with vmap?
    
    results = []
    key = jax.random.PRNGKey(42)
    
    for n in sample_sizes:
        print(f"Simulating N={n}/group ({n_sims} trials)...")
        
        # We can implement the trials loop in Python, measuring metrics in Batch
        # Or even batch the trials?
        # Let's batch trials too? Shape (n_sims, n_sub, ...)
        # Actually, memory might be tight if we do ALL.
        # Let's loop sims, but Batch the subjects (N<=50 is small).
        
        fp_naive = 0
        fp_informed = 0
        
        for i in range(n_sims):
            key, k1, k2 = jax.random.split(key, 3)
            
            # Generate Data
            _, psds_a, _ = generate_group_spectra(k1, n, exponent_mean=1.5)
            freqs, psds_b, _ = generate_group_spectra(k2, n, exponent_mean=1.2)
            
            # Naive Metric
            scores_a_naive = get_band_power_batch(freqs, psds_a)
            scores_b_naive = get_band_power_batch(freqs, psds_b)
            
            # Informed Metric
            scores_a_informed = fit_batch_jit(freqs, psds_a)
            scores_b_informed = fit_batch_jit(freqs, psds_b)
            
            # Convert to numpy for T-test (scipy is fast enough for 100 sims)
            a_n = np.array(scores_a_naive)
            b_n = np.array(scores_b_naive)
            a_i = np.array(scores_a_informed)
            b_i = np.array(scores_b_informed)
            
            # T-test (Ind)
            _, p_n = stats.ttest_ind(a_n, b_n, equal_var=False)
            _, p_i = stats.ttest_ind(a_i, b_i, equal_var=False)
            
            if p_n < 0.05: fp_naive += 1
            if p_i < 0.05: fp_informed += 1
            
        rate_naive = fp_naive / n_sims
        rate_informed = fp_informed / n_sims
        
        print(f"  > Naive FP Rate:    {rate_naive:.2f}")
        print(f"  > Informed FP Rate: {rate_informed:.2f}")
        
        results.append({
            "Sample Size": n,
            "Naive FPR": rate_naive,
            "Informed FPR": rate_informed
        })
        
    df = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    plt.plot(df["Sample Size"], df["Naive FPR"], 'r-o', label="Naive (Band Power)", lw=2)
    plt.plot(df["Sample Size"], df["Informed FPR"], 'b-s', label="Informed (NeuroJAX)", lw=2)
    plt.axhline(0.05, color='k', linestyle='--', label="Target Alpha (0.05)")
    plt.ylim(0, 1.05)
    plt.title("False Positive Rate by Analysis Method\n(Effect of 1/f Slope Confound)")
    plt.xlabel("Sample Size (N per group)")
    plt.ylabel("False Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("group_power_simulation.png")
    print("Saved plot to group_power_simulation.png")

if __name__ == "__main__":
    run_simulation()
