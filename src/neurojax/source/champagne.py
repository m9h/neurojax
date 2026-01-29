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
    
    Update rules (MacKay / Wipf):
    Gamma_new = Gamma * sqrt( diag( G.T @ C_y^-1 @ G ) ) / ...?
    Actually, simpler EM form:
    gamma_i_new = gamma_i * || w_i^T y ||^2 / (1 - gamma_i * w_i^T G_i) ... no.
    
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
        
    # Init Gamma (ones? or beamformer power?)
    gamma = jnp.ones(n_src)
    
    def body(val):
        i, gam, diff = val
        
        # 1. Model Covariance: Sigma_y = G Gamma G.T + Sigma_noise
        # Gamma is diag. G @ diag(gam) @ G.T = (G * gam) @ G.T
        Sigma_y = jnp.dot(gain * gam[None, :], gain.T) + noise_cov
        
        # Invert Sigma_y
        Sigma_inv = jnp.linalg.inv(Sigma_y)
        
        # 2. Compute auxiliary terms
        # W = Gamma G.T Sigma_inv
        # Posterior Mean X_bar = W Y.
        # But we work with covariances.
        # We need term: diag( G.T @ Sigma_inv @ C_data @ Sigma_inv @ G )
        # Let Z = Sigma_inv @ G
        Z = jnp.dot(Sigma_inv, gain) # (n_chan, n_src)
        
        # Numerator Term: diag( Z.T @ C_data @ Z )
        # Z.T @ C @ Z -> (n_src, n_src). We only need diag.
        # diag( A.T @ B @ A ) = sum( (B @ A) * A, axis=0 )
        C_Z = jnp.dot(cov, Z) # (n_chan, n_src)
        numer_diag = jnp.sum(C_Z * Z, axis=0) # (n_src,)
        
        # Denominator Term: diag( G.T @ Sigma_inv @ G ) = diag( G.T @ Z )
        denom_diag = jnp.sum(gain * Z, axis=0) # (n_src,)
        
        # Update Rule (Convex Bounding):
        # gam_new = gam * sqrt( numer / denom )
        gam_new = gam * jnp.sqrt(numer_diag / (denom_diag + 1e-12))
        
        # Change
        d = jnp.max(jnp.abs(gam - gam_new))
        
        return i + 1, gam_new, d
        
    def cond(val):
        i, _, d = val
        return (i < max_iter) & (d > tol)
        
    _, gamma_final, _ = lax.while_loop(cond, body, (0, gamma, 1.0))
    
    # Compute Final Weights
    # W = Gamma G.T Sigma_inv
    # Recompute Sigma
    Sigma_y = jnp.dot(gain * gamma_final[None, :], gain.T) + noise_cov
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
