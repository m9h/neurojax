"""
CHAMPAGNE (CHArge Moment Projector Averaged over Generalized Noise Estimators).
Empirical Bayesian Beamformer / Sparse Bayesian Learning (SBL).

Reference: Wipf et al., "A unified Bayesian framework for MEG/EEG source imaging", 2008.

Key Idea: Iteratively update source power priors (Gamma) to maximize model evidence.
Results in sparse solutions that are robust to correlated sources.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax

@jit
def champagne_solver(cov: jnp.ndarray, gain: jnp.ndarray, noise_cov: jnp.ndarray = None, max_iter: int = 20, tol: float = 1e-4) -> jnp.ndarray:
    """
    Solve for Source Powers (Gamma) using SBL / CHAMPAGNE rules.

    Y = G X + E
    Cov_y = G Gamma G.T + Cov_noise

    Using the Convex Bounding rule (Wipf 2008):
    gamma_new = gamma * sqrt( diag( G.T @ C_inv @ C_data @ C_inv @ G ) / diag( G.T @ C_inv @ G ) )

    Parameters
    ----------
    cov: (n_chan, n_chan) Data covariance.
    gain: (n_chan, n_sources) Leadfield.
    noise_cov: (n_chan, n_chan) Noise covariance. If None, Identity.

    Returns
    -------
    gamma: (n_sources,) Estimated source powers.
    weights: (n_sources, n_chan) Posterior weights (Beamformer-like).
    """
    n_chan, n_src = gain.shape
    if noise_cov is None:
        noise_cov = jnp.eye(n_chan)

    # Regularise noise covariance for stable inversion
    noise_reg = noise_cov + jnp.eye(n_chan) * jnp.trace(noise_cov) * 1e-6

    # Initialise gamma from data: use diagonal of beamformer power
    # This is much more stable than ones initialisation
    C_inv_init = jnp.linalg.inv(cov + noise_reg)
    Z_init = C_inv_init @ gain
    init_power = jnp.sum((cov @ Z_init) * Z_init, axis=0)
    init_denom = jnp.sum(gain * Z_init, axis=0)
    gamma = jnp.maximum(init_power / jnp.maximum(init_denom, 1e-20), 1e-20)
    # Normalise to reasonable scale
    gamma = gamma / jnp.maximum(jnp.median(gamma), 1e-20)

    def body(val):
        i, gam, diff = val

        # 1. Model Covariance with regularisation
        Sigma_y = jnp.dot(gain * gam[None, :], gain.T) + noise_reg

        # Regularised inverse
        Sigma_inv = jnp.linalg.inv(Sigma_y)

        # 2. Compute update terms
        Z = jnp.dot(Sigma_inv, gain)

        # Numerator: diag(Z.T @ C_data @ Z)
        C_Z = jnp.dot(cov, Z)
        numer_diag = jnp.sum(C_Z * Z, axis=0)

        # Denominator: diag(G.T @ Sigma_inv @ G)
        denom_diag = jnp.sum(gain * Z, axis=0)

        # Convex bounding update with clipping for stability
        ratio = jnp.clip(numer_diag / jnp.maximum(denom_diag, 1e-20), 0.0, 1e6)
        gam_new = gam * jnp.sqrt(ratio)

        # Clip gamma to prevent divergence
        gam_new = jnp.clip(gam_new, 1e-20, 1e10)

        d = jnp.max(jnp.abs(gam - gam_new) / jnp.maximum(gam, 1e-20))
        return i + 1, gam_new, d

    def cond(val):
        i, gam, d = val
        return (i < max_iter) & (d > tol) & jnp.all(jnp.isfinite(gam))

    _, gamma_final, _ = lax.while_loop(cond, body, (0, gamma, 1.0))

    # Final weights: W = Gamma G.T Sigma_inv
    Sigma_y = jnp.dot(gain * gamma_final[None, :], gain.T) + noise_reg
    Sigma_inv = jnp.linalg.inv(Sigma_y)
    weights = jnp.dot(gamma_final[:, None] * gain.T, Sigma_inv)

    return gamma_final, weights

@jit
def imaginary_coherence(source_data: jnp.ndarray, ref_idx: int) -> jnp.ndarray:
    """
    Compute Imaginary Coherence between a reference source and all others.
    Robuts to volume conduction (which is real-valued / zero-lag).
    
    iCoh = Im( S_xy ) / sqrt( S_xx * S_yy )
    """
    n_src, n_time = source_data.shape
    
    # Compute Cross-Spectral Density (CSD) or just Analytic Signal Correlation?
    # If source_data is analytic (complex), we can use correlation.
    # Cov = E[ x y* ]
    
    ref = source_data[ref_idx]
    
    # Cross-product with reference
    # x * conj(ref)
    cross = source_data * jnp.conj(ref[None, :])
    
    # Mean over time -> Covariance/CSD
    csd = jnp.mean(cross, axis=1) # (n_src,) (Complex)
    
    # Power
    p_ref = jnp.mean(jnp.abs(ref)**2)
    p_src = jnp.mean(jnp.abs(source_data)**2, axis=1)
    
    # Coherency
    coh = csd / jnp.sqrt(p_src * p_ref + 1e-12)
    
    # Imaginary Coherence
    icoh = jnp.imag(coh)
    
    return icoh
