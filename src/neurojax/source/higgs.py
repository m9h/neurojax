"""HIGGS — Hidden Gaussian Graphical Spectral model.

Joint one-step source imaging and connectivity estimation from M/EEG
data using EM with a Hermitian graphical LASSO (hgLASSO).

Instead of the traditional two-step approach (source reconstruction
then connectivity estimation), HIGGS jointly estimates both the source
activity and the source-space precision matrix (inverse covariance,
encoding conditional independence / connectivity) in a single EM
framework.

The key insight: the source prior is parametrized as a Gaussian
Graphical Model (GGM) whose precision matrix is estimated with a
Hermitian graphical LASSO. This allows frequency-resolved connectivity
estimation in source space, where edges in the graphical model
correspond to direct (partial) connections between brain regions.

Model:
    y(f) = G @ x(f) + e(f)           (measurement equation, per frequency)
    x(f) ~ CN(0, Theta_x(f)^{-1})    (source GGM prior, frequency-dependent)
    e(f) ~ CN(0, Theta_e(f)^{-1})    (noise GGM)

EM algorithm:
    E-step: MAP source estimate given current precision matrices
    M-step: Update source and noise precision via Hermitian gLASSO
            on the sufficient statistics from the E-step

The Hermitian graphical LASSO uses an ADMM formulation to handle the
L1 penalty on complex-valued off-diagonal entries, promoting sparsity
in the estimated connectivity.

References
----------
Valdes-Sosa PA, Paz-Linares D, Gonzalez-Moreira E, et al. (2023).
Effective connectivity in the human connectome from the HIGGS
(Hidden Gaussian Graphical Spectral) model. Scientific Reports.

Gonzalez-Moreira E, Paz-Linares D, et al. (2021).
Population and individual level EEG/MEG source connectivity analysis
using the Hermitian graphical LASSO. bioRxiv.

Friedman J, Hastie T, Tibshirani R (2008). Sparse inverse covariance
estimation with the graphical lasso. Biostatistics 9(3):432-441.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Hermitian graphical LASSO
# ---------------------------------------------------------------------------


def _soft_threshold_complex(z: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Complex soft-thresholding operator.

    For complex z: ST(z, lam) = z * max(0, 1 - lam/|z|)
    This is the proximal operator of lam * |z| for complex z.
    """
    abs_z = jnp.abs(z)
    scale = jnp.maximum(1.0 - lam / jnp.maximum(abs_z, 1e-30), 0.0)
    return z * scale


def hermitian_glasso(
    S: jnp.ndarray,
    alpha: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> jnp.ndarray:
    """Hermitian graphical LASSO for complex-valued precision estimation.

    Estimates a sparse Hermitian precision matrix Theta from the sample
    covariance S by solving:

        minimize_Theta  -log det(Theta) + tr(S @ Theta) + alpha * sum_{i!=j} |Theta_ij|

    Uses ADMM: split into Theta (smooth) and Z (sparse) variables:
        minimize  -log det(Theta) + tr(S @ Theta) + alpha * ||Z_off||_1
        s.t.      Theta = Z

    ADMM updates:
        Theta <- argmin -log det(Theta) + (rho/2)||Theta - Z + U||_F^2
              =  eigendecomposition update
        Z     <- soft-threshold(Theta + U, alpha/rho) on off-diagonal
        U     <- U + Theta - Z

    Parameters
    ----------
    S : (n, n) complex — sample covariance (Hermitian positive definite).
    alpha : float — L1 penalty on off-diagonal entries.
    max_iter : int — maximum iterations.
    tol : float — convergence tolerance.

    Returns
    -------
    Theta : (n, n) complex — estimated sparse Hermitian precision matrix.
    """
    n = S.shape[0]
    is_complex = jnp.iscomplexobj(S)
    dtype = S.dtype

    # Regularize S for numerical stability
    S_reg = S + 1e-6 * jnp.trace(S).real / n * jnp.eye(n, dtype=dtype)

    # ADMM penalty parameter
    rho = jnp.maximum(jnp.real(jnp.trace(S_reg)) / n, 0.01)   # traced: jit/vmap-safe

    # Initialize
    s_diag = jnp.real(jnp.diag(S_reg))
    Theta = jnp.diag(1.0 / s_diag).astype(dtype)
    Z = Theta.copy()
    U = jnp.zeros((n, n), dtype=dtype)

    # Off-diagonal mask for L1 penalty
    off_diag = 1.0 - jnp.eye(n)

    def admm_step(carry, _):
        Theta, Z, U = carry

        # --- Theta update ---
        # minimize -log det(Theta) + tr(S @ Theta) + (rho/2)||Theta - (Z - U)||_F^2
        # = minimize -log det(Theta) + (rho/2)||Theta - A||_F^2
        # where A = Z - U - S/rho
        A = Z - U - S_reg / rho

        # Ensure A is Hermitian
        A = (A + A.conj().T) / 2

        # Eigendecompose A: A = Q @ diag(d) @ Q^H
        d, Q = jnp.linalg.eigh(A)

        # Theta update (diagonal in eigenbasis):
        # theta_i = (d_i + sqrt(d_i^2 + 4/rho)) / 2
        d_new = (d + jnp.sqrt(d ** 2 + 4.0 / rho)) / 2.0

        Theta_new = (Q * d_new[None, :]) @ Q.conj().T

        # Enforce Hermitian symmetry
        Theta_new = (Theta_new + Theta_new.conj().T) / 2

        # --- Z update ---
        # minimize alpha * ||Z_off||_1 + (rho/2)||Z - (Theta_new + U)||_F^2
        V = Theta_new + U
        # Soft-threshold off-diagonal, keep diagonal
        lam = (alpha / rho) * off_diag
        Z_new = _soft_threshold_complex(V, lam.astype(jnp.float32))
        # Diagonal: no penalty, just copy
        Z_new = Z_new * off_diag.astype(dtype) + jnp.diag(jnp.diag(V))
        # Enforce Hermitian
        Z_new = (Z_new + Z_new.conj().T) / 2

        # --- U update ---
        U_new = U + Theta_new - Z_new

        return (Theta_new, Z_new, U_new), None

    (Theta, Z, U), _ = jax.lax.scan(admm_step, (Theta, Z, U), None, length=max_iter)

    # Final result: use Z (the sparse variable) as the estimate
    # Enforce Hermitian symmetry and positive diagonal
    Theta_out = (Z + Z.conj().T) / 2
    diag_vals = jnp.maximum(jnp.real(jnp.diag(Theta_out)), 1e-6)
    Theta_out = Theta_out - jnp.diag(jnp.diag(Theta_out)) + jnp.diag(diag_vals.astype(dtype))

    return Theta_out


# ---------------------------------------------------------------------------
# Debiasing
# ---------------------------------------------------------------------------


def debias_precision(
    Theta: jnp.ndarray,
    S: jnp.ndarray,
) -> jnp.ndarray:
    """Debias a precision matrix estimate.

    Applies the one-step debiasing formula:
        Theta_debiased = 2 * Theta - Theta @ S @ Theta

    This removes the shrinkage bias introduced by the L1 penalty
    in the graphical LASSO, giving an asymptotically unbiased
    estimator of the true precision matrix.

    Parameters
    ----------
    Theta : (n, n) — estimated precision matrix (possibly complex).
    S : (n, n) — sample covariance matrix.

    Returns
    -------
    Theta_debiased : (n, n) — debiased precision estimate.
    """
    Theta_debiased = 2 * Theta - Theta @ S @ Theta
    # Enforce Hermitian symmetry
    Theta_debiased = (Theta_debiased + Theta_debiased.conj().T) / 2
    return Theta_debiased


# ---------------------------------------------------------------------------
# MAP source estimate
# ---------------------------------------------------------------------------


def higgs_source_estimate(
    data: jnp.ndarray,
    leadfield: jnp.ndarray,
    Theta_source: jnp.ndarray,
    Theta_noise: jnp.ndarray,
) -> jnp.ndarray:
    """MAP source estimate given source and noise precision matrices.

    The MAP estimator under the Gaussian model y = G @ x + e:
        x_hat = Sigma_source @ G^T @ (G @ Sigma_source @ G^T + Sigma_noise)^{-1} @ y

    where Sigma_source = Theta_source^{-1} and Sigma_noise = Theta_noise^{-1}.

    This is the E-step of the HIGGS EM algorithm.

    Parameters
    ----------
    data : (n_sensors, n_timepoints) — sensor measurements.
    leadfield : (n_sensors, n_sources) — forward model.
    Theta_source : (n_sources, n_sources) — source precision matrix.
    Theta_noise : (n_sensors, n_sensors) — noise precision matrix.

    Returns
    -------
    sources : (n_sources, n_timepoints) — MAP source estimates.
    """
    n_sensors, n_sources = leadfield.shape

    # Source covariance from precision
    Sigma_source = jnp.linalg.inv(
        Theta_source + 1e-8 * jnp.eye(n_sources, dtype=Theta_source.dtype)
    )
    Sigma_source = jnp.real(Sigma_source)

    # Noise covariance from precision
    Sigma_noise = jnp.linalg.inv(
        Theta_noise + 1e-8 * jnp.eye(n_sensors, dtype=Theta_noise.dtype)
    )
    Sigma_noise = jnp.real(Sigma_noise)

    # Model covariance in sensor space
    C_model = leadfield @ Sigma_source @ leadfield.T + Sigma_noise

    # Regularize
    C_model = C_model + 1e-6 * jnp.trace(C_model) / n_sensors * jnp.eye(n_sensors)

    # Wiener filter / MAP kernel
    W = Sigma_source @ leadfield.T @ jnp.linalg.inv(C_model)

    return W @ data


# ---------------------------------------------------------------------------
# Cross-spectral density computation
# ---------------------------------------------------------------------------


def _compute_csd(
    data: jnp.ndarray,
    freqs: jnp.ndarray,
    fs: float = 200.0,
    bandwidth: float = 4.0,
) -> jnp.ndarray:
    """Compute cross-spectral density at specified frequencies via DFT.

    Parameters
    ----------
    data : (n_channels, n_times) — sensor data.
    freqs : (n_freqs,) — frequencies of interest.
    fs : float — sampling rate.
    bandwidth : float — frequency smoothing bandwidth in Hz.

    Returns
    -------
    csd : (n_freqs, n_channels, n_channels) — cross-spectral density.
    """
    n_ch, n_times = data.shape

    # Compute DFT
    fft_data = jnp.fft.rfft(data, axis=1)  # (n_ch, n_fft)
    fft_freqs = jnp.fft.rfftfreq(n_times, d=1.0 / fs)  # (n_fft,)

    # For each frequency of interest, average CSD within bandwidth
    def compute_one_freq(f):
        # Frequency mask: bins within bandwidth/2 of target
        mask = jnp.abs(fft_freqs - f) < (bandwidth / 2.0)
        mask = mask.astype(jnp.float32)
        n_bins = jnp.maximum(jnp.sum(mask), 1.0)

        # Weighted FFT coefficients
        fft_weighted = fft_data * mask[None, :]  # (n_ch, n_fft)

        # CSD: average of outer products
        csd_f = (fft_weighted @ fft_weighted.conj().T) / (n_bins * n_times)
        return csd_f

    csd = jax.vmap(compute_one_freq)(freqs)  # (n_freqs, n_ch, n_ch)
    return csd


# ---------------------------------------------------------------------------
# HIGGS EM algorithm
# ---------------------------------------------------------------------------


def higgs_em(
    data: jnp.ndarray,
    leadfield: jnp.ndarray,
    n_sources: int,
    freqs: jnp.ndarray,
    alpha: float = 0.1,
    n_iter: int = 30,
    fs: float = 200.0,
    bandwidth: float = 4.0,
) -> dict:
    """Full HIGGS EM algorithm for joint source + connectivity estimation.

    Alternates between:
        E-step: MAP source estimate given current precision matrices
        M-step: Update source precision via Hermitian gLASSO on
                source-space cross-spectral density

    Parameters
    ----------
    data : (n_sensors, n_timepoints) — sensor measurements.
    leadfield : (n_sensors, n_sources) — forward model.
    n_sources : int — number of source locations.
    freqs : (n_freqs,) — frequencies of interest (Hz).
    alpha : float — sparsity penalty for graphical LASSO.
    n_iter : int — number of EM iterations.
    fs : float — sampling frequency (Hz).
    bandwidth : float — frequency bandwidth for CSD estimation.

    Returns
    -------
    dict with:
        sources : (n_sources, n_timepoints) — final source estimate.
        precision : (n_sources, n_sources) — average source precision.
        precision_per_freq : (n_freqs, n_sources, n_sources) — per-frequency.
        nll_history : (n_iter,) — negative log-likelihood per iteration.
    """
    n_sensors, n_times = data.shape
    n_freqs = freqs.shape[0]

    # Initialize source precision as identity (no connectivity assumed)
    Theta_source = jnp.eye(n_sources, dtype=jnp.complex64)
    Theta_per_freq = jnp.tile(Theta_source[None, :, :], (n_freqs, 1, 1))

    # Initialize noise covariance from data variance (avoid precision instability)
    data_var = jnp.mean(data ** 2)
    noise_var_init = jnp.maximum(data_var * 0.1, 1e-6)
    Sigma_noise = noise_var_init * jnp.eye(n_sensors)

    # Data covariance (time domain, for NLL computation)
    C_data = (data @ data.T) / n_times

    # EM iterations
    nll_history = []

    for it in range(n_iter):
        # ----- E-step: MAP source estimate -----
        # Average precision across frequencies for the time-domain estimate
        Theta_source_avg = jnp.mean(jnp.real(Theta_per_freq), axis=0)
        # Ensure symmetric and positive definite
        Theta_source_avg = (Theta_source_avg + Theta_source_avg.T) / 2
        eigvals = jnp.linalg.eigvalsh(Theta_source_avg)
        min_eig = jnp.min(eigvals)
        # Shift to ensure positive definiteness
        shift = jnp.where(min_eig < 1e-4, jnp.abs(min_eig) + 1e-4, 0.0)
        Theta_source_avg = Theta_source_avg + shift * jnp.eye(n_sources)

        # Use noise covariance directly (more stable than going through precision)
        Sigma_source = jnp.linalg.inv(
            Theta_source_avg + 1e-8 * jnp.eye(n_sources)
        )
        C_model = leadfield @ Sigma_source @ leadfield.T + Sigma_noise
        C_model_reg = C_model + 1e-6 * jnp.trace(C_model) / n_sensors * jnp.eye(n_sensors)
        W = Sigma_source @ leadfield.T @ jnp.linalg.inv(C_model_reg)
        sources_hat = W @ data

        # ----- Compute source-space CSD (sufficient statistic) -----
        # Also account for posterior uncertainty:
        # <x x^H> = x_hat x_hat^H + Sigma_post
        # Sigma_post = Sigma_source - W @ G @ Sigma_source
        Sigma_post = Sigma_source - W @ leadfield @ Sigma_source
        Sigma_post = (Sigma_post + Sigma_post.T) / 2  # symmetrize

        source_csd = _compute_csd(sources_hat, freqs, fs, bandwidth)

        # Add posterior uncertainty contribution to CSD (diagonal boost)
        post_diag = jnp.maximum(jnp.diag(Sigma_post), 0.0)
        post_correction = jnp.diag(post_diag).astype(jnp.complex64)
        source_csd = source_csd + post_correction[None, :, :]

        # ----- M-step: Update precision per frequency via hgLASSO -----
        new_precisions = []
        for f_idx in range(n_freqs):
            S_f = source_csd[f_idx]
            # Regularize the sample CSD
            S_f = S_f + 1e-4 * jnp.trace(S_f).real / n_sources * jnp.eye(
                n_sources, dtype=S_f.dtype
            )

            Theta_f = hermitian_glasso(S_f, alpha=alpha, max_iter=30)
            new_precisions.append(Theta_f)

        Theta_per_freq = jnp.stack(new_precisions, axis=0)

        # ----- Update noise covariance -----
        residuals = data - leadfield @ sources_hat
        C_residual = (residuals @ residuals.T) / n_times
        # Add posterior covariance propagated through leadfield
        C_residual = C_residual + leadfield @ jnp.diag(post_diag) @ leadfield.T
        Sigma_noise = C_residual + 1e-6 * jnp.trace(C_residual) / n_sensors * jnp.eye(
            n_sensors
        )

        # ----- Compute negative log-likelihood -----
        C_model_nll = leadfield @ Sigma_source @ leadfield.T + Sigma_noise
        C_model_nll = C_model_nll + 1e-6 * jnp.trace(C_model_nll) / n_sensors * jnp.eye(
            n_sensors
        )
        sign, logdet = jnp.linalg.slogdet(C_model_nll)
        C_model_nll_inv = jnp.linalg.inv(C_model_nll)
        nll = 0.5 * (
            jnp.trace(C_model_nll_inv @ C_data) * n_times + logdet * n_times
        )
        nll_history.append(nll)

    nll_history = jnp.array(nll_history)

    # Average precision across frequencies for the final output
    Theta_final = jnp.mean(jnp.real(Theta_per_freq), axis=0)
    Theta_final = (Theta_final + Theta_final.T) / 2

    return {
        "sources": sources_hat,
        "precision": Theta_final,
        "precision_per_freq": Theta_per_freq,
        "nll_history": nll_history,
    }


# ---------------------------------------------------------------------------
# Frequency-domain HIGGS on sensor cross-spectra (one jitted function)
# ---------------------------------------------------------------------------


def _hermitian(A):
    return 0.5 * (A + jnp.swapaxes(A, -1, -2).conj())


def _higgs_one_freq(S_y, G, alpha, n_iter, glasso_iter, noise_floor, noise_alpha=None):
    """HIGGS EM at a single frequency from the sensor CSD ``S_y`` (M, M).

    ``noise_alpha=None`` keeps a diagonal sensor noise; a float runs the full
    HIGGS noise model — an hgLASSO GGM on the expected residual cross-spectrum.
    """
    M, N = G.shape
    cdtype = S_y.dtype
    G_c = G.astype(cdtype)
    I_M, I_N = jnp.eye(M, dtype=cdtype), jnp.eye(N, dtype=cdtype)
    sensor_power = jnp.real(jnp.trace(S_y)) / M

    def e_step(Theta_x, Sigma_e):
        Sigma_x = _hermitian(jnp.linalg.inv(Theta_x + 1e-6 * I_N))
        C = _hermitian(G_c @ Sigma_x @ G_c.conj().T + Sigma_e)
        C = C + 1e-6 * sensor_power * I_M
        C_inv = jnp.linalg.inv(C)
        T = Sigma_x @ G_c.conj().T @ C_inv                       # MAP kernel (N, M)
        Sigma_post = _hermitian(Sigma_x - T @ G_c @ Sigma_x)
        S_x = _hermitian(T @ S_y @ T.conj().T + Sigma_post)       # E[x x^H]
        sign, logdet = jnp.linalg.slogdet(C)
        nll = jnp.real(logdet + jnp.trace(C_inv @ S_y))
        return T, S_x, nll

    def em_step(carry, _):
        Theta_x, Sigma_e = carry
        T, S_x, nll = e_step(Theta_x, Sigma_e)
        S_x_reg = S_x + 1e-4 * jnp.real(jnp.trace(S_x)) / N * I_N
        Theta_new = hermitian_glasso(S_x_reg, alpha=alpha, max_iter=glasso_iter)
        # Noise: diagonal of the expected residual cross-spectrum (kept diagonal
        # for stability; a full hgLASSO noise GGM is the HIGGS generalisation)
        GT_S = G_c @ T @ S_y
        R = _hermitian(S_y - GT_S - GT_S.conj().T + G_c @ S_x @ G_c.conj().T)
        r = jnp.maximum(jnp.real(jnp.diag(R)), noise_floor * sensor_power)
        if noise_alpha is None:
            Sigma_e_new = jnp.diag(r).astype(cdtype)
        else:
            # full noise GGM: floor the diagonal, then sparse Hermitian precision
            R_f = R + jnp.diag(r - jnp.real(jnp.diag(R))).astype(cdtype)
            # penalty relative to the residual's own power, as alpha is to the data's
            r_mean = jnp.mean(r)
            Theta_e = hermitian_glasso(R_f / r_mean, alpha=noise_alpha,
                                       max_iter=glasso_iter) / r_mean
            Sigma_e_new = _hermitian(jnp.linalg.inv(Theta_e))
        return (Theta_new, Sigma_e_new), nll

    Theta0 = I_N
    Sigma_e0 = (0.1 * sensor_power) * I_M
    (Theta_x, Sigma_e), nll = jax.lax.scan(em_step, (Theta0, Sigma_e0), None, length=n_iter)
    T, S_x, _ = e_step(Theta_x, Sigma_e)
    Theta_e = _hermitian(jnp.linalg.inv(Sigma_e))
    return Theta_x, S_x, Sigma_e, Theta_e, T, nll


@partial(jax.jit, static_argnames=("n_iter", "glasso_iter"))
def higgs_spectral(
    csd: jnp.ndarray,
    leadfield: jnp.ndarray,
    alpha: float = 0.05,
    n_iter: int = 20,
    glasso_iter: int = 50,
    noise_floor: float = 1e-3,
    noise_alpha: float | None = None,
) -> dict:
    """Frequency-domain HIGGS: joint source CSD + sparse source precision per frequency.

    The CiftiStorm module-3 operation in one jitted call: for every frequency,
    EM between the MAP source estimate from the sensor cross-spectrum
    (E-step, with posterior covariance folded into the source cross-spectrum)
    and a Hermitian graphical LASSO update of the source precision (M-step).
    ``vmap`` over frequencies, ``lax.scan`` over EM iterations; differentiable
    in ``alpha`` and the leadfield.

    Parameters
    ----------
    csd : (n_freqs, n_sensors, n_sensors) complex sensor cross-spectral density
        (e.g. multitaper; see :mod:`neurojax.analysis.multitaper`).
    leadfield : (n_sensors, n_sources) real gain matrix.
    alpha : hgLASSO sparsity penalty on off-diagonal source precision, relative
        to unit mean sensor power (each frequency's CSD is normalised internally).
    n_iter : EM iterations.
    glasso_iter : ADMM iterations per M-step.
    noise_floor : minimum noise variance as a fraction of mean sensor power.
    noise_alpha : ``None`` keeps the sensor noise diagonal; a float estimates
        the full HIGGS noise model, an hgLASSO GGM on the expected residual
        cross-spectrum with this off-diagonal penalty (relative to unit power).

    Returns
    -------
    dict with ``precision`` (F, N, N) sparse Hermitian source precision,
    ``source_csd`` (F, N, N) expected source cross-spectrum, ``noise_cov``
    (F, M, M) sensor noise (diagonal unless ``noise_alpha``), ``noise_precision``
    (F, M, M) its inverse (sparse with ``noise_alpha``), ``kernel`` (F, N, M) final MAP inverse
    operators, and ``nll`` (n_iter,) sensor negative log-likelihood summed over
    frequencies (Gaussian CSD likelihood, up to constants, on the unit-power scale).
    """
    csd = jnp.asarray(csd)
    if not jnp.iscomplexobj(csd):
        csd = csd.astype(jnp.complex64)
    leadfield = jnp.asarray(leadfield, dtype=jnp.real(csd).dtype)
    # The unit source prior (Theta0 = I), the hgLASSO penalty and the numerical
    # jitter are all absolute, so the EM runs on CSDs of unit mean sensor power
    # per frequency and the outputs are scaled back. Without this a CSD in
    # physical units (~1e-4 per MEG channel) oscillates instead of converging.
    scale = jnp.real(jnp.trace(csd, axis1=1, axis2=2)) / csd.shape[1]
    scale = jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)
    run = jax.vmap(lambda S: _higgs_one_freq(S, leadfield, alpha, n_iter, glasso_iter,
                                             noise_floor, noise_alpha))
    # GPUs default to TF32 matmuls for float32/complex64; 20 EM iterations of
    # 273x273 inverses on them give a different graph every run (WAND
    # sub-01187: edge density 0.88 vs 0.94 between identical runs). Full
    # precision makes GPU runs reproducible and equal to the CPU result.
    with jax.default_matmul_precision("highest"):
        Theta, S_x, Sigma_e, Theta_e, T, nll = run(csd / scale[:, None, None])
    s3 = scale[:, None, None].astype(csd.dtype)
    return {
        "precision": Theta / s3,
        "source_csd": S_x * s3,
        "noise_cov": Sigma_e * s3,
        "noise_precision": Theta_e / s3,
        "kernel": T,
        "nll": jnp.sum(nll, axis=0),
    }


def higgs_select_alpha(
    csd: jnp.ndarray,
    leadfield: jnp.ndarray,
    alphas,
    n_eff: int,
    gamma: float = 0.5,
    **kwargs,
) -> dict:
    """Choose the hgLASSO penalty of :func:`higgs_spectral` by extended BIC.

    For each candidate ``alpha`` the EM is run and scored with

        EBIC = 2 * n_eff * nll + k * log(n_eff) + 4 * gamma * k * log(P)

    where ``nll`` is the final sensor negative log-likelihood summed over
    frequencies (unit-power scale), ``k`` the number of non-zero off-diagonal
    source-precision edges summed over frequencies, ``P = N (N - 1) / 2`` the
    number of possible edges per frequency and ``n_eff`` the number of
    independent CSD estimates (windows × tapers for a multitaper CSD).
    ``gamma = 0`` is plain BIC; ``gamma = 0.5`` is the Foygel & Drton (2010)
    default for graphs with many more edges than samples.

    Returns
    -------
    dict with ``alphas``, ``ebic`` (per alpha), ``n_edges`` (per alpha),
    ``nll`` (per alpha), the selected ``alpha`` and its full ``result``.
    """
    alphas = list(alphas)
    N = leadfield.shape[1]
    off = ~jnp.eye(N, dtype=bool)
    P = N * (N - 1) / 2
    results, ebic, n_edges, nll = [], [], [], []
    for a in alphas:
        out = higgs_spectral(csd, leadfield, alpha=a, **kwargs)
        edges = jnp.sum((jnp.abs(out["precision"]) > 1e-6) & off[None]) / 2
        score = (2.0 * n_eff * out["nll"][-1] + edges * jnp.log(n_eff)
                 + 4.0 * gamma * edges * jnp.log(P))
        results.append(out); ebic.append(score); n_edges.append(edges); nll.append(out["nll"][-1])
    ebic = jnp.stack(ebic)
    best = int(jnp.argmin(ebic))
    return {
        "alphas": alphas,
        "ebic": ebic,
        "n_edges": jnp.stack(n_edges).astype(int),
        "nll": jnp.stack(nll),
        "alpha": alphas[best],
        "result": results[best],
    }
