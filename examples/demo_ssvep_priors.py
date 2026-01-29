import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import optimistix as optx
from neurojax.spectral import fit_spectrum, PowerSpectrumModel

def generate_synthetic_data(freqs, aperiodic_params, peaks_params, noise_level=0.1, key=None):
    """Generates synthetic PSD Log Power."""
    model = PowerSpectrumModel()
    # Flatten peaks params
    flat_peaks = jnp.array(peaks_params).flatten()
    true_params = jnp.concatenate([jnp.array(aperiodic_params), flat_peaks])
    
    clean_spectrum = model(freqs, true_params)
    
    if key is not None:
        noise = jax.random.normal(key, shape=clean_spectrum.shape) * noise_level
        return clean_spectrum + noise, true_params
    return clean_spectrum, true_params

def main():
    print("--- 1/f Priors for SSVEP Analysis ---")
    
    freqs = jnp.linspace(1, 50, 100)
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    # ---------------------------------------------------------
    # 1. Subject "Ground Truth" Physiology (Resting State)
    # ---------------------------------------------------------
    # High Exponent (sleepy/relaxed) = 1.5, Alpha Peak at 11Hz
    rest_aperiodic = [1.0, 1.5] # Offset, Exponent
    rest_peaks = [[11.0, 0.5, 1.0]] # Alpha
    
    print(f"Generating Resting State Data (True 1/f exp={rest_aperiodic[1]})...")
    rest_psd, _ = generate_synthetic_data(freqs, rest_aperiodic, rest_peaks, noise_level=0.05, key=k1)
    
    # ---------------------------------------------------------
    # 2. SSVEP Task Data (Same Subject)
    # ---------------------------------------------------------
    # Same 1/f, Same Alpha, BUT added SSVEP at 20Hz
    # Crucially: We add a 'baseline shift' in the broad band power (arousal)
    # Task Exponent = 1.2 (Flatter, more aroused)
    task_aperiodic = [0.8, 1.2] 
    task_peaks = [[11.0, 0.3, 1.0], [20.0, 0.6, 0.5]] # Alpha (suppressed), SSVEP
    
    print(f"Generating Task Data (True SSVEP Amp=0.6, 1/f shift to 1.2)...")
    task_psd, true_task_params = generate_synthetic_data(freqs, task_aperiodic, task_peaks, noise_level=0.05, key=k2)

    # ---------------------------------------------------------
    # 3. Strategy A: Naive Fit (Fit Task blindly)
    # ---------------------------------------------------------
    print("\n--- Strategy A: Naive Fit (Task Only) ---")
    # Blind initialization is tricky. Let's give it a fair shot: 
    # One peak at 10 (Alpha), one at 20 (SSVEP guess) or just random spread.
    # If we init both at 10, they merge.
    # Initial: [max, 1.0, 10, 0.5, 1.0, 20, 0.5, 1.0]
    log_max = jnp.max(task_psd)
    naive_init = jnp.array([log_max, 1.0, 10.0, 0.5, 1.0, 22.0, 0.5, 1.0]) 
    
    naive_params = fit_spectrum(freqs, task_psd, n_peaks=2, initial_params=naive_init)
    
    # Check which peak is the 20Hz one.
    # Peak 1: indices 2,3,4. Peak 2: indices 5,6,7.
    # We want the amplitude (pw) of the SSVEP peak (Peak 2 if sorted, but here likely Peak 2).
    
    # Helper to find 20Hz peak
    def get_ssvep_amp(p):
        # p[2] is CF1, p[5] is CF2
        # Check which is closer to 20
        dist1 = abs(p[2] - 20.0)
        dist2 = abs(p[5] - 20.0)
        if dist1 < dist2:
             return jax.nn.softplus(p[3]) # Amp1
        else:
             return jax.nn.softplus(p[6]) # Amp2

    naive_amp = get_ssvep_amp(naive_params)
    print(f"Naive 1/f Exponent: {naive_params[1]:.2f} (True: {task_aperiodic[1]})")
    print(f"Naive SSVEP Amp:   {naive_amp:.2f} (True Softplus(0.6)~={jax.nn.softplus(0.6):.2f})")

    # ---------------------------------------------------------
    # 4. Strategy B: Resting-Informed Fit
    # ---------------------------------------------------------
    print("\n--- Strategy B: Resting-Informed Fit ---")
    
    # Step 1: Fit Rest
    rest_fit = fit_spectrum(freqs, rest_psd, n_peaks=1)
    prior_offset, prior_exponent = rest_fit[0], rest_fit[1]
    print(f"Step 1: Learned Priors from Rest -> Exp={prior_exponent:.2f}")
    
    def loss_with_prior(params, args):
        freqs, data, prior_exp = args
        model = PowerSpectrumModel()
        model_output = model(freqs, params)
        mse = jnp.mean((model_output - data)**2)
        # Regularization: Penalize deviation from Resting Exponent
        reg = 0.5 * (params[1] - prior_exp)**2 
        return mse + reg

    solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
    
    # Informed Init: Use Prior Aperiodic + Alpha (10) + SSVEP (20)
    # We know Alpha is around 10 from Rest fit? Ideally we use Rest peaks too.
    # But let's just use the Aperiodic prior + generic peaks.
    informed_init = jnp.array([prior_offset, prior_exponent, 10.0, 0.5, 1.0, 20.0, 0.5, 1.0])
    
    sol = optx.least_squares(
        fn=loss_with_prior,
        y0=informed_init,
        args=(freqs, task_psd, prior_exponent),
        solver=solver,
        max_steps=5000,
        throw=False
    )
    informed_params = sol.value
    informed_amp = get_ssvep_amp(informed_params)
    
    print(f"Informed 1/f Exponent: {informed_params[1]:.2f}")
    print(f"Informed SSVEP Amp:   {informed_amp:.2f}")
    
    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    # True value is Softplus(0.6) because model applies Softplus
    true_target = float(jax.nn.softplus(0.6))
    
    err_A = abs(naive_amp - true_target)
    err_B = abs(informed_amp - true_target)
    
    print(f"\nError Naive: {err_A:.4f}")
    print(f"Error Informed: {err_B:.4f}")
    
    if err_B < err_A:
        print("\nSUCCESS: Resting-Informed prior improved SSVEP estimation!")
    else:
        print("\nNote: Naive fit was already good enough (high SNR synthetic data).")

if __name__ == "__main__":
    main()
