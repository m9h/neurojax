# © NeuroJAX developers
#
# License: BSD (3-clause)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from functools import partial

class SpecParam(eqx.Module):
    """
    Spectral Parameterization (SpecParam / FOOOF) in JAX.
    
    Model: P(f) = L(f) + G(f)
    L(f) = offset - log(f^chi + k) (Aperiodic with knee)
    G(f) = sum_n a_n * exp(-(f-c_n)^2 / (2w_n)^2) (Periodic)
    """
    # Parameters
    # Aperiodic
    aperiodic_params: jnp.ndarray # [offset, knee, exponent]
    
    # Periodic (n_peaks, 3) -> [center, amplitude, width]
    # We use a fixed max number of peaks and zero-out unused ones via amplitude mask?
    # Or just fit N peaks.
    peak_params: jnp.ndarray 
    
    # Config
    n_peaks: int = eqx.field(static=True)
    
    def __init__(self, n_peaks=3):
        self.n_peaks = n_peaks
        # Initialize
        # offset=1, knee=0, exp=2 (1/f^2)
        self.aperiodic_params = jnp.array([1.0, 0.0, 2.0]) 
        # Peaks: spread out
        # [10Hz, 1.0, 1.0], [20Hz, 1.0, 1.0], ...
        self.peak_params = jnp.zeros((n_peaks, 3))
        # Init logic usually requires data, but here we just construct
        
    def get_model(self, freqs):
        # Aperiodic
        off, knee, exp = self.aperiodic_params
        # L = off - log(freqs^exp + knee)
        # Note: FOOOF usually fits log10 power.
        # Let's assume input y is log10(power).
        # Robust log: log(freqs^exp + knee)
        
        # Ensure positivity constraints for computation
        # exponent > 0, knee >= 0
        exp_ = jax.nn.softplus(exp)
        knee_ = jax.nn.softplus(knee)
        
        # Avoid log(0) at f=0
        freqs_safe = jnp.maximum(freqs, 1e-5)
        
        # FOOOF 'knee' mode:
        # y = offset - log10(k + f^x)
        L = off - jnp.log10(knee_ + freqs_safe**exp_)
        
        # Periodic
        # Gaussian: a * exp(-(x-c)^2 / (2w^2))
        # params: [center, amp, width]
        
        def gaussian(f, p):
            c, a, w = p
            # constraints
            a_ = jax.nn.softplus(a) # positive amplitude
            w_ = jax.nn.softplus(w) + 0.1 # positive width, min 0.1 Hz
            
            return a_ * jnp.exp(-((f - c)**2) / (2 * w_**2))
            
        G = jax.vmap(gaussian, in_axes=(None, 0))(freqs, self.peak_params)
        G_sum = jnp.sum(G, axis=0)
        
        return L + G_sum
        
    @staticmethod
    def loss(model, freqs, power_spectrum):
        pred = model.get_model(freqs)
        # MSE loss
        return jnp.mean((pred - power_spectrum)**2)

    @staticmethod
    def fit(freqs, power_spectrum, n_peaks=3, key=None, steps=1000, lr=0.05):
        """
        Fit model to a single spectrum.
        freqs: (n_freqs,)
        power_spectrum: (n_freqs,) -- assumed simple (non-log), handled inside? 
        Wait, FOOOF implementation usually fits log-power.
        Let's assume input is log10(power).
        """
        model = SpecParam(n_peaks)
        
        # Smart Init
        # Guess offset ~ max(power)
        log_power = power_spectrum
        model = eqx.tree_at(
            lambda t: t.aperiodic_params, 
            model, 
            jnp.array([jnp.max(log_power), 0.0, 2.0])
        )
        
        # Initialize peaks?
        # A smart fit would find peaks in residual. 
        # Here we rely on gradient descent from random/grid or zero?
        # Let's start with peaks at alpha/beta/gamma centers roughly?
        # [10, 0.5, 2], [20, 0.1, 2], [50, 0.1, 5]
        init_peaks = jnp.array([
            [10.0, 0.5, 2.0],
            [22.0, 0.2, 3.0], 
            [50.0, 0.1, 5.0]
        ])
        # Pad or slice
        if n_peaks > 3:
            # fill rest with 0
            pad = jnp.zeros((n_peaks-3, 3))
            init_peaks = jnp.concatenate([init_peaks, pad])
        else:
            init_peaks = init_peaks[:n_peaks]
            
        model = eqx.tree_at(lambda t: t.peak_params, model, init_peaks)
        
        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(model)
        
        @eqx.filter_jit
        def step(model, opt_state, freqs, y):
            loss, grads = eqx.filter_value_and_grad(SpecParam.loss)(model, freqs, y)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss
            
        for _ in range(steps):
             model, opt_state, _ = step(model, opt_state, freqs, log_power)
             
        return model
