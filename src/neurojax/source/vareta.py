"""VARETA — Variable Resolution Electromagnetic Tomography.

Spatially adaptive inverse solution that learns the resolution
(smoothness) from the data, rather than assuming a fixed prior.
Developed by Pedro Valdes-Sosa and colleagues.

The key innovation: instead of a fixed depth-weighted source covariance
prior (like MNE/dSPM/sLORETA), VARETA estimates a spatially-varying
precision matrix for the sources using Empirical Bayes. Sources in
regions with high SNR get high resolution (focal estimates); sources
in low-SNR regions get smoothed (regularized) estimates.

This is the precursor to the modern Bayesian inverse approaches
(CHAMPAGNE, hierarchical models) and inspired sLORETA and eLORETA.

Model:
    y = G @ x + e           (measurement equation)
    x ~ N(0, Σ_x)           (source prior)
    e ~ N(0, Σ_e)           (noise)
    Σ_x = diag(σ²_1, ..., σ²_N)  (diagonal, learned from data)

The spatial prior Σ_x is estimated iteratively via EM or evidence
maximization, yielding a data-driven resolution that adapts across
the cortex.

References
----------
Valdes-Sosa PA et al. (2000). Variable resolution electric-magnetic
tomography. Proceedings of the 12th International Conference on
Biomagnetism.

Trujillo-Barreto NJ, Aubert-Vazquez E, Valdes-Sosa PA (2004).
Bayesian model averaging in EEG/MEG imaging. NeuroImage 21(4):1300-1319.

Valdes-Sosa PA et al. (2009). Model driven EEG/fMRI fusion of brain
oscillations. Human Brain Mapping 30(9):2701-2721.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


from functools import partial

@partial(jax.jit, static_argnames=("n_iter",))
def vareta(
    data: jnp.ndarray,
    gain: jnp.ndarray,
    noise_cov: jnp.ndarray,
    n_iter: int = 50,
    spatial_smoothing: float = 0.5,
    min_variance: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """VARETA inverse solution with adaptive spatial resolution.

    Iteratively estimates per-source variance (precision) from data
    using Empirical Bayes / evidence maximization.

    Parameters
    ----------
    data : (n_sensors, n_timepoints) — sensor measurements.
    gain : (n_sensors, n_sources) — leadfield matrix.
    noise_cov : (n_sensors, n_sensors) — noise covariance.
    n_iter : int — number of EM iterations.
    spatial_smoothing : float — regularization on the spatial
        smoothness of the variance map. 0 = no smoothing (focal),
        1 = strong smoothing (distributed).
    min_variance : float — floor on source variance to prevent collapse.

    Returns
    -------
    source : (n_sources, n_timepoints) — estimated source activity.
    source_variance : (n_sources,) — learned per-source variance
        (the "variable resolution" — larger variance = more active region).
    evidence : (n_iter,) — marginal log-evidence per iteration
        (should increase monotonically).
    """
    n_sensors, n_sources = gain.shape
    n_timepoints = data.shape[1]

    # Initialize source variances uniformly
    sigma_sq = jnp.ones(n_sources)

    # Precompute noise inverse
    noise_reg = noise_cov + 1e-6 * jnp.trace(noise_cov) / n_sensors * jnp.eye(n_sensors)
    noise_inv = jnp.linalg.inv(noise_reg)

    # Data covariance (empirical)
    C_data = (data @ data.T) / n_timepoints

    def em_step(carry, _):
        sigma_sq = carry

        # Source prior: Σ_x = diag(σ²)
        Sigma_x = jnp.diag(sigma_sq)

        # Model covariance: C_model = G @ Σ_x @ G^T + Σ_noise
        C_model = gain @ Sigma_x @ gain.T + noise_reg

        # Posterior: W = Σ_x @ G^T @ C_model^{-1}
        C_model_inv = jnp.linalg.inv(C_model)
        W = Sigma_x @ gain.T @ C_model_inv  # (n_sources, n_sensors)

        # Posterior covariance (diagonal approximation)
        # Σ_post = Σ_x - W @ G @ Σ_x
        WG = W @ gain  # (n_sources, n_sources)
        post_var = sigma_sq - jnp.sum(WG * Sigma_x, axis=1)
        post_var = jnp.maximum(post_var, 0.0)

        # Source estimates
        source_est = W @ data  # (n_sources, n_timepoints)

        # Update source variances (M-step)
        # σ²_i = <x_i²> + Σ_post_ii
        source_power = jnp.mean(source_est ** 2, axis=1)  # (n_sources,)
        sigma_sq_new = source_power + post_var

        # Optional spatial smoothing of the variance map
        # Simple moving average with nearest neighbors (1D approximation)
        if spatial_smoothing > 0:
            kernel = jnp.array([spatial_smoothing / 2, 1 - spatial_smoothing, spatial_smoothing / 2])
            sigma_sq_smooth = jnp.convolve(sigma_sq_new, kernel, mode='same')
            sigma_sq_new = sigma_sq_smooth

        # Floor to prevent variance collapse
        sigma_sq_new = jnp.maximum(sigma_sq_new, min_variance)

        # Evidence (marginal log-likelihood, unnormalized)
        sign, logdet = jnp.linalg.slogdet(C_model)
        evidence = -0.5 * (jnp.trace(C_model_inv @ C_data) * n_timepoints +
                            logdet * n_timepoints)

        return sigma_sq_new, evidence

    # Run EM iterations
    sigma_sq, evidence_history = jax.lax.scan(em_step, sigma_sq, None, length=n_iter)

    # Final source estimate with converged variances
    Sigma_x = jnp.diag(sigma_sq)
    C_model = gain @ Sigma_x @ gain.T + noise_reg
    W = Sigma_x @ gain.T @ jnp.linalg.inv(C_model)
    source = W @ data

    return source, sigma_sq, evidence_history


@jax.jit
def vareta_resolution_map(
    sigma_sq: jnp.ndarray,
) -> jnp.ndarray:
    """Convert VARETA source variances to a resolution map.

    Higher variance = more signal detected = better resolution.
    This is the "variable resolution" that gives VARETA its name.

    Parameters
    ----------
    sigma_sq : (n_sources,) — learned per-source variance.

    Returns
    -------
    resolution : (n_sources,) — normalized resolution (0 = suppressed, 1 = peak).
    """
    return sigma_sq / jnp.maximum(jnp.max(sigma_sq), 1e-10)


def vareta_with_connectivity_prior(
    data: jnp.ndarray,
    gain: jnp.ndarray,
    noise_cov: jnp.ndarray,
    adjacency: jnp.ndarray,
    n_iter: int = 50,
    lambda_graph: float = 0.1,
    min_variance: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """VARETA with graph-Laplacian spatial prior from structural connectivity.

    Instead of simple 1D smoothing, uses the structural connectome
    (or cortical adjacency graph) to define which sources should have
    similar variances. This is the bridge between source reconstruction
    and whole-brain modeling.

    Parameters
    ----------
    data : (n_sensors, n_timepoints).
    gain : (n_sensors, n_sources).
    noise_cov : (n_sensors, n_sensors).
    adjacency : (n_sources, n_sources) — structural connectivity or
        cortical adjacency. Can be from:
        - FreeSurfer surface mesh (cortical adjacency)
        - DWI structural connectome (Cmat from neurojax)
        - Both combined
    n_iter : int.
    lambda_graph : float — strength of graph regularization.
    min_variance : float.

    Returns
    -------
    source, sigma_sq, evidence — same as vareta().
    """
    n_sensors, n_sources = gain.shape
    n_timepoints = data.shape[1]

    sigma_sq = jnp.ones(n_sources)

    noise_reg = noise_cov + 1e-6 * jnp.trace(noise_cov) / n_sensors * jnp.eye(n_sensors)

    # Graph Laplacian for spatial regularization
    degree = jnp.sum(adjacency, axis=1)
    L = jnp.diag(degree) - adjacency  # Graph Laplacian

    def em_step(sigma_sq, _):
        Sigma_x = jnp.diag(sigma_sq)
        C_model = gain @ Sigma_x @ gain.T + noise_reg
        C_model_inv = jnp.linalg.inv(C_model)
        W = Sigma_x @ gain.T @ C_model_inv

        source_est = W @ data
        post_var = sigma_sq - jnp.sum((W @ gain) * Sigma_x, axis=1)
        post_var = jnp.maximum(post_var, 0.0)

        source_power = jnp.mean(source_est ** 2, axis=1)
        sigma_sq_new = source_power + post_var

        # Graph Laplacian smoothing: minimize σ^T L σ
        # Gradient: ∂/∂σ (λ σ^T L σ) = 2λ L σ
        # Simple proximal step: σ ← (I + λ_graph L)^{-1} σ_new
        smooth_matrix = jnp.eye(n_sources) + lambda_graph * L
        sigma_sq_new = jnp.linalg.solve(smooth_matrix, sigma_sq_new)

        sigma_sq_new = jnp.maximum(sigma_sq_new, min_variance)

        sign, logdet = jnp.linalg.slogdet(C_model)
        C_data = (data @ data.T) / n_timepoints
        evidence = -0.5 * (jnp.trace(C_model_inv @ C_data) * n_timepoints +
                            logdet * n_timepoints)

        return sigma_sq_new, evidence

    sigma_sq, evidence_history = jax.lax.scan(em_step, sigma_sq, None, length=n_iter)

    Sigma_x = jnp.diag(sigma_sq)
    C_model = gain @ Sigma_x @ gain.T + noise_reg
    W = Sigma_x @ gain.T @ jnp.linalg.inv(C_model)
    source = W @ data

    return source, sigma_sq, evidence_history
