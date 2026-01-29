import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxtyping import Array, Float

class PowerSpectrumModel(eqx.Module):
    """
    JAX-native parametrization of neural power spectra.
    Model: PSD(f) = Aperiodic(f) + Periodic(f)
    """
    # Aperiodic parameters: offset, exponent
    # Periodic parameters: center_freqs, powers, bandwidths
    
    def __call__(self, freqs, params):
        # Unpack params
        # params structure: [offset, exponent, peak1_cf, peak1_pw, peak1_bw, ...]
        offset, exponent = params[0], params[1]
        
        # Aperiodic component: L(f) = b - log(k + f^chi)
        # Simplified "fixed" mode: L(f) = b - chi * log(f)
        aperiodic_log = offset - exponent * jnp.log10(freqs)
        
        # Periodic component: Gaussians
        # We assume a fixed max number of peaks for JIT compatibility, 
        # or we pass specific params for specific peaks.
        # Here we implement a flexible "N peaks" version by reshaping the tail of params.
        
        peaks = params[2:]
        n_peak_params = 3
        # Reshape to [n_peaks, 3] -> (cf, pw, bw)
        peaks = peaks.reshape(-1, n_peak_params)
        
        periodic_log = jnp.zeros_like(freqs)
        
        
        def add_gaussian(current_sum, peak_params):
            cf, pw, bw = peak_params
            # Constraints:
            # Power must be positive (though in log space, 'power' usually means height above aperiodic)
            # Bandwidth must be positive
            pw = jax.nn.softplus(pw)
            bw = jax.nn.softplus(bw)
            
            # center frequency should technically be positive too, but usually safe
            
            gaussian = pw * jnp.exp(-((freqs - cf)**2) / (2 * bw**2))
            return current_sum + gaussian, None

        periodic_log, _ = jax.lax.scan(add_gaussian, periodic_log, peaks)
        
        # Total Log Power
        return aperiodic_log + periodic_log

def fit_spectrum(freqs, power_spectrum, n_peaks=1, initial_params=None):
    """
    Fits the PowerSpectrumModel to data using Optimistix.
    """
    model = PowerSpectrumModel()
    
    # Loss function: Error between Model(params) and Data
    def loss(params, args):
        freqs, data = args
        model_output = model(freqs, params)
        return jnp.mean((model_output - data)**2) # MSE

    # Initial guess
    if initial_params is None:
        # Heuristic guess
        # Offset: max power (log space)
        # Exponent: 1.0
        # Peaks: 10Hz, 0.5 power, 1.0 bw (generic Alpha)
        log_max = jnp.max(power_spectrum)
        initial_params = jnp.array([log_max, 1.0] + [10.0, 0.5, 1.0] * n_peaks)

    # Use a more robust solver for initial convergence if LM fails? 
    # LM is usually good. Increasing steps.
    solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
    sol = optx.least_squares(
        fn=loss,
        y0=initial_params,
        args=(freqs, power_spectrum),
        solver=solver,
        max_steps=5000, # Increased from 1000
        throw=False # Don't error out, just return best guess
    )
    
    return sol.value
