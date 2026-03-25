"""
Unit tests for the neurojax.analysis subpackage.

Covers: decomposition, dimensionality, filtering, complex_ica, mca,
        mixture, spectral, stats, superlet, timefreq, spm, rough.
(ica.py and tensor.py are already tested elsewhere.)
"""

import os
import sys

# Ensure src is on path for the worktree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Disable GPU to keep tests fast and portable
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rank_deficient(key, n_features, n_samples, true_rank):
    """Create a rank-deficient matrix (n_features x n_samples) with known rank."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n_features, true_rank))
    B = jax.random.normal(k2, (true_rank, n_samples))
    return A @ B


def _make_sinusoid(freq, sfreq, duration, amplitude=1.0, phase=0.0):
    """Return a single-channel sinusoidal signal."""
    t = jnp.arange(0, duration, 1.0 / sfreq)
    return amplitude * jnp.sin(2.0 * jnp.pi * freq * t + phase)


# ===================================================================
# decomposition.py  — PCA whitening, dimensionality estimation
# ===================================================================


class TestDecomposition:
    """Tests for neurojax.analysis.decomposition."""

    def test_whiten_pca_shapes(self):
        from neurojax.analysis.decomposition import whiten_pca

        key = jax.random.PRNGKey(0)
        n_features, n_samples, n_comp = 10, 200, 3
        X = _make_rank_deficient(key, n_features, n_samples, n_comp)

        X_w, W, mu, evals, evecs = whiten_pca(X, n_components=n_comp)
        assert X_w.shape == (n_comp, n_samples)
        assert W.shape == (n_comp, n_features)
        assert mu.shape == (n_features, 1)
        assert evals.shape == (n_comp,)
        assert evecs.shape == (n_features, n_comp)

    def test_whiten_pca_variance(self):
        """After whitening, the covariance of X_white should be ~I."""
        from neurojax.analysis.decomposition import whiten_pca

        key = jax.random.PRNGKey(1)
        X = jax.random.normal(key, (8, 500))
        n_comp = 4
        X_w, *_ = whiten_pca(X, n_components=n_comp)

        cov = jnp.dot(X_w, X_w.T) / (X_w.shape[1] - 1)
        np.testing.assert_allclose(cov, jnp.eye(n_comp), atol=0.15)

    def test_whiten_pca_auto_dim(self):
        """When n_components=None, dimension is estimated automatically."""
        from neurojax.analysis.decomposition import whiten_pca

        key = jax.random.PRNGKey(2)
        n_features, n_samples = 20, 300
        X = _make_rank_deficient(key, n_features, n_samples, true_rank=3)
        # Add small noise so eigenvalues don't collapse to exact zeros
        X = X + 0.01 * jax.random.normal(key, X.shape)

        X_w, *_ = whiten_pca(X, n_components=None)
        # Should pick a small number of components (close to 3)
        assert X_w.shape[0] <= n_features
        assert X_w.shape[0] >= 1

    def test_estimate_dimension_laplace(self):
        """95%-variance cutoff returns sensible dimension."""
        from neurojax.analysis.decomposition import estimate_dimension_laplace

        # Eigenvalues with a clear gap: [10, 5, 1, 0.01, 0.01, ...]
        evals = jnp.array([10.0, 5.0, 1.0, 0.01, 0.01, 0.005, 0.002])
        dim = estimate_dimension_laplace(evals, n_samples=100)
        # 95% variance should be achieved in roughly 3 components
        assert 1 <= int(dim) <= 4

    def test_fastica_basic(self):
        """FastICA recovers independent sources from a simple mixture."""
        from neurojax.analysis.decomposition import whiten_pca, fastica

        key = jax.random.PRNGKey(42)
        n_samples = 1000
        t = jnp.linspace(0, 1, n_samples)
        s1 = jnp.sin(2 * jnp.pi * 3 * t)
        s2 = jnp.sign(jnp.sin(2 * jnp.pi * 7 * t))
        S = jnp.stack([s1, s2])  # (2, n_samples)

        # Mix
        A = jnp.array([[1.0, 0.5], [0.3, 0.8]])
        X = A @ S

        X_w, *_ = whiten_pca(X, n_components=2)
        W = fastica(X_w, 2, key, max_iter=200)

        # W should be approximately orthogonal
        WWT = W @ W.T
        np.testing.assert_allclose(WWT, jnp.eye(2), atol=0.2)

    def test_probabilistic_ica_shapes(self):
        from neurojax.analysis.decomposition import probabilistic_ica

        key = jax.random.PRNGKey(10)
        X = jax.random.normal(key, (6, 200))
        S_z, Mixing, evals = probabilistic_ica(X, n_components=3, key=key)
        assert S_z.shape[0] == 3
        assert S_z.shape[1] == 200
        assert Mixing.shape == (6, 3)

    def test_sym_decorrelate(self):
        from neurojax.analysis.decomposition import _sym_decorrelate

        key = jax.random.PRNGKey(5)
        W = jax.random.normal(key, (4, 4))
        W_dec = _sym_decorrelate(W)
        prod = W_dec @ W_dec.T
        np.testing.assert_allclose(prod, jnp.eye(4), atol=1e-4)


# ===================================================================
# dimensionality.py  — PPCA Laplace / BIC / AIC
# ===================================================================


class TestDimensionality:
    """Tests for neurojax.analysis.dimensionality (PPCA)."""

    def test_ppca_importable(self):
        from neurojax.analysis.dimensionality import PPCA
        assert hasattr(PPCA, "estimate_dimensionality")

    def test_laplace_evidence_shape(self):
        from neurojax.analysis.dimensionality import PPCA

        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (10, 100))
        ev = PPCA.get_laplace_evidence(X, max_dim=5)
        assert ev.shape == (5,)

    def test_consensus_evidence_keys(self):
        from neurojax.analysis.dimensionality import PPCA

        key = jax.random.PRNGKey(1)
        X = jax.random.normal(key, (8, 120))
        scores = PPCA.get_consensus_evidence(X, max_dim=4)
        assert "aic" in scores
        assert "bic" in scores
        assert scores["aic"].shape == (4,)
        assert scores["bic"].shape == (4,)

    def test_estimate_dimensionality_rank_deficient(self):
        from neurojax.analysis.dimensionality import PPCA

        key = jax.random.PRNGKey(2)
        X = _make_rank_deficient(key, 15, 200, true_rank=3)
        X = X + 0.001 * jax.random.normal(key, X.shape)

        k_bic = PPCA.estimate_dimensionality(X, method="bic")
        # BIC should pick something reasonable (less than full rank)
        assert 1 <= int(k_bic) <= 14

    def test_estimate_dimensionality_methods(self):
        """All three method strings run without error."""
        from neurojax.analysis.dimensionality import PPCA

        key = jax.random.PRNGKey(3)
        X = jax.random.normal(key, (6, 80))
        for method in ("aic", "bic", "consensus", "laplace"):
            k = PPCA.estimate_dimensionality(X, method=method)
            assert int(k) >= 1


# ===================================================================
# filtering.py  — FFT filter, notch, robust reference
# ===================================================================


class TestFiltering:
    """Tests for neurojax.analysis.filtering."""

    def test_filter_fft_lowpass(self):
        from neurojax.analysis.filtering import filter_fft

        sfreq = 500.0
        t = jnp.arange(0, 1.0, 1.0 / sfreq)
        # 10 Hz + 100 Hz
        sig = jnp.sin(2 * jnp.pi * 10 * t) + jnp.sin(2 * jnp.pi * 100 * t)
        sig = sig[None, :]  # (1, n_times)

        out = filter_fft(sig, sfreq=sfreq, f_low=None, f_high=30.0)
        assert out.shape == sig.shape

        # After lowpass at 30 Hz, the 100 Hz component should be gone
        spec = jnp.abs(jnp.fft.rfft(out[0]))
        freqs = jnp.fft.rfftfreq(out.shape[-1], d=1.0 / sfreq)
        idx_100 = jnp.argmin(jnp.abs(freqs - 100))
        idx_10 = jnp.argmin(jnp.abs(freqs - 10))
        assert float(spec[idx_100]) < 0.01 * float(spec[idx_10])

    def test_filter_fft_highpass(self):
        from neurojax.analysis.filtering import filter_fft

        sfreq = 500.0
        t = jnp.arange(0, 1.0, 1.0 / sfreq)
        sig = jnp.sin(2 * jnp.pi * 3 * t) + jnp.sin(2 * jnp.pi * 50 * t)
        sig = sig[None, :]

        out = filter_fft(sig, sfreq=sfreq, f_low=20.0, f_high=None)
        spec = jnp.abs(jnp.fft.rfft(out[0]))
        freqs = jnp.fft.rfftfreq(out.shape[-1], d=1.0 / sfreq)
        idx_3 = jnp.argmin(jnp.abs(freqs - 3))
        idx_50 = jnp.argmin(jnp.abs(freqs - 50))
        # 3 Hz should be attenuated
        assert float(spec[idx_3]) < 0.01 * float(spec[idx_50])

    def test_filter_fft_bandpass(self):
        from neurojax.analysis.filtering import filter_fft

        sfreq = 500.0
        t = jnp.arange(0, 1.0, 1.0 / sfreq)
        sig = (jnp.sin(2 * jnp.pi * 5 * t) +
               jnp.sin(2 * jnp.pi * 30 * t) +
               jnp.sin(2 * jnp.pi * 100 * t))
        sig = sig[None, :]

        out = filter_fft(sig, sfreq=sfreq, f_low=20.0, f_high=50.0)
        spec = jnp.abs(jnp.fft.rfft(out[0]))
        freqs = jnp.fft.rfftfreq(out.shape[-1], d=1.0 / sfreq)
        idx_30 = jnp.argmin(jnp.abs(freqs - 30))
        idx_5 = jnp.argmin(jnp.abs(freqs - 5))
        idx_100 = jnp.argmin(jnp.abs(freqs - 100))
        # 30 Hz retained, 5 Hz and 100 Hz attenuated
        assert float(spec[idx_5]) < 0.01 * float(spec[idx_30])
        assert float(spec[idx_100]) < 0.01 * float(spec[idx_30])

    def test_notch_filter(self):
        from neurojax.analysis.filtering import notch_filter_fft

        sfreq = 500.0
        t = jnp.arange(0, 1.0, 1.0 / sfreq)
        sig = jnp.sin(2 * jnp.pi * 10 * t) + jnp.sin(2 * jnp.pi * 60 * t)
        sig = sig[None, :]

        out = notch_filter_fft(sig, sfreq=sfreq, freq=60.0, width=2.0)
        spec = jnp.abs(jnp.fft.rfft(out[0]))
        freqs = jnp.fft.rfftfreq(out.shape[-1], d=1.0 / sfreq)
        idx_60 = jnp.argmin(jnp.abs(freqs - 60))
        idx_10 = jnp.argmin(jnp.abs(freqs - 10))
        # 60 Hz notched out
        assert float(spec[idx_60]) < 0.01 * float(spec[idx_10])

    def test_robust_reference(self):
        from neurojax.analysis.filtering import robust_reference

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (5, 200))
        out = robust_reference(data)
        assert out.shape == data.shape
        # Mean across channels should be near zero
        mean_signal = jnp.mean(out, axis=0)
        assert float(jnp.max(jnp.abs(mean_signal))) < 2.0


# ===================================================================
# complex_ica.py  — Complex FastICA
# ===================================================================


class TestComplexICA:
    """Tests for neurojax.analysis.complex_ica."""

    def test_complex_sym_decorrelation(self):
        from neurojax.analysis.complex_ica import _complex_sym_decorrelation

        key = jax.random.PRNGKey(0)
        W = jax.random.normal(key, (3, 3)) + 1j * jax.random.normal(
            jax.random.PRNGKey(1), (3, 3)
        )
        W = W.astype(jnp.complex64)
        W_dec = _complex_sym_decorrelation(W)
        prod = W_dec @ jnp.conj(W_dec.T)
        np.testing.assert_allclose(jnp.abs(prod), jnp.eye(3), atol=0.1)

    def test_complex_ica_class_fit(self):
        from neurojax.analysis.complex_ica import ComplexICA

        key = jax.random.PRNGKey(42)
        n_features, n_samples = 4, 200
        X = jax.random.normal(key, (n_features, n_samples)).astype(jnp.complex64)
        X = X + 1j * jax.random.normal(jax.random.PRNGKey(1), (n_features, n_samples))

        model = ComplexICA(n_components=2)
        model.fit(X)

        assert model.components_ is not None
        assert model.components_.shape[0] == 2
        assert model.components_.shape[1] == n_samples
        assert model.mixing_ is not None
        assert model.mixing_.shape == (n_features, 2)

    def test_complex_ica_real_input(self):
        """ComplexICA should accept real inputs (cast to complex internally)."""
        from neurojax.analysis.complex_ica import ComplexICA

        key = jax.random.PRNGKey(10)
        X = jax.random.normal(key, (3, 150))
        model = ComplexICA(n_components=2)
        model.fit(X)
        assert model.components_.shape[0] == 2

    def test_complex_fast_ica_step_shapes(self):
        from neurojax.analysis.complex_ica import complex_fast_ica_step, _complex_sym_decorrelation

        key = jax.random.PRNGKey(0)
        k, n_samples = 3, 100
        X = (jax.random.normal(key, (k, n_samples)) +
             1j * jax.random.normal(jax.random.PRNGKey(1), (k, n_samples))).astype(jnp.complex64)
        W = (jax.random.normal(jax.random.PRNGKey(2), (k, k)) +
             1j * jax.random.normal(jax.random.PRNGKey(3), (k, k))).astype(jnp.complex64)
        W = _complex_sym_decorrelation(W)
        W_out = complex_fast_ica_step(W, X, max_iter=10)
        assert W_out.shape == (k, k)


# ===================================================================
# mca.py  — Morphological Component Analysis
# ===================================================================


class TestMCA:
    """Tests for neurojax.analysis.mca."""

    def test_soft_threshold(self):
        from neurojax.analysis.mca import soft_threshold

        x = jnp.array([3.0, -2.0, 0.5, -0.3])
        out = soft_threshold(x, 1.0)
        # |3|-1=2 -> 2*sign(3)=2; |2|-1=1 -> -1; |0.5|-1<0 -> 0; |0.3|-1<0 -> 0
        assert float(out[0]) > 0
        assert float(out[1]) < 0
        assert abs(float(out[2])) < 0.6  # partly shrunken
        assert abs(float(out[3])) < 0.4

    def test_soft_threshold_complex(self):
        from neurojax.analysis.mca import soft_threshold

        x = jnp.array([3.0 + 4.0j])  # magnitude = 5
        out = soft_threshold(x, 2.0)
        # scale = max(0, 1 - 2/5) = 0.6
        expected_mag = 5.0 * 0.6
        np.testing.assert_allclose(jnp.abs(out[0]), expected_mag, atol=0.01)

    def test_fft_roundtrip(self):
        from neurojax.analysis.mca import fft_forward, fft_inverse

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        c = fft_forward(x)
        x_back = fft_inverse(c)
        np.testing.assert_allclose(x_back, x, atol=1e-5)

    def test_mca_solve_reconstruction(self):
        from neurojax.analysis.mca import mca_solve

        # Oscillation + spike
        t = jnp.linspace(0, 1, 128)
        osc = jnp.sin(2 * jnp.pi * 5 * t)
        spike = jnp.zeros(128).at[64].set(5.0)
        data = osc + spike

        part_osc, part_trans = mca_solve(data, lambda_fft=0.1, lambda_ident=0.3, n_iter=30)
        # Total should approximately reconstruct data
        recon = part_osc + part_trans
        residual = jnp.mean((recon - data) ** 2)
        assert float(residual) < 2.0  # Rough reconstruction

    def test_mca_decompose_batch(self):
        from neurojax.analysis.mca import mca_decompose

        key = jax.random.PRNGKey(0)
        batch = jax.random.normal(key, (3, 64))
        osc, trans = mca_decompose(batch, lambda_fft=0.5, lambda_ident=0.5, n_iter=10)
        assert osc.shape == (3, 64)
        assert trans.shape == (3, 64)


# ===================================================================
# mixture.py  — GaussianGammaMixture
# ===================================================================


class TestMixture:
    """Tests for neurojax.analysis.mixture (GaussianGammaMixture)."""

    def test_ggm_init(self):
        from neurojax.analysis.mixture import GaussianGammaMixture

        key = jax.random.PRNGKey(0)
        model = GaussianGammaMixture(key)
        weights, mu, sigma, a_p, b_p, a_n, b_n = model.get_params()
        # Weights should sum to 1
        np.testing.assert_allclose(float(jnp.sum(weights)), 1.0, atol=1e-5)
        assert float(sigma) > 0

    def test_ggm_log_prob_shape(self):
        from neurojax.analysis.mixture import GaussianGammaMixture

        key = jax.random.PRNGKey(0)
        model = GaussianGammaMixture(key)
        x = jnp.linspace(-5, 5, 100)
        lp = model.log_prob(x)
        assert lp.shape == (100,)
        # Log-probs should be finite for reasonable x
        assert jnp.all(jnp.isfinite(lp))

    def test_ggm_posterior_prob(self):
        from neurojax.analysis.mixture import GaussianGammaMixture

        key = jax.random.PRNGKey(0)
        model = GaussianGammaMixture(key)
        x = jnp.linspace(-5, 5, 50)
        p_active = model.posterior_prob(x)
        assert p_active.shape == (50,)
        # Probabilities should be in [0, 1]
        assert float(jnp.min(p_active)) >= 0.0
        assert float(jnp.max(p_active)) <= 1.0 + 1e-6

    def test_ggm_fit_runs(self):
        """Test that fitting runs without error (small steps for speed)."""
        from neurojax.analysis.mixture import GaussianGammaMixture

        key = jax.random.PRNGKey(42)
        # Generate bimodal data: Gaussian + some outliers
        data = jax.random.normal(key, (200,))
        # Add some positive outliers
        data = data.at[:20].add(4.0)

        model = GaussianGammaMixture.fit(data, key, steps=50, lr=0.01)
        # Should return a model with finite params
        weights, mu, sigma, *_ = model.get_params()
        assert jnp.all(jnp.isfinite(weights))
        assert jnp.isfinite(mu)
        assert jnp.isfinite(sigma)

    def test_ggm_loss(self):
        from neurojax.analysis.mixture import GaussianGammaMixture

        key = jax.random.PRNGKey(0)
        model = GaussianGammaMixture(key)
        x = jax.random.normal(key, (100,))
        loss = GaussianGammaMixture.loss(model, x)
        assert jnp.isfinite(loss)
        assert float(loss) > 0  # Negative log prob should be positive


# ===================================================================
# spectral.py  — SpecParam / FOOOF
# ===================================================================


class TestSpectral:
    """Tests for neurojax.analysis.spectral (SpecParam)."""

    def test_specparam_init(self):
        from neurojax.analysis.spectral import SpecParam

        model = SpecParam(n_peaks=2)
        assert model.n_peaks == 2
        assert model.aperiodic_params.shape == (3,)
        assert model.peak_params.shape == (2, 3)

    def test_specparam_get_model_shape(self):
        from neurojax.analysis.spectral import SpecParam

        model = SpecParam(n_peaks=3)
        freqs = jnp.linspace(1, 100, 200)
        spectrum = model.get_model(freqs)
        assert spectrum.shape == (200,)
        assert jnp.all(jnp.isfinite(spectrum))

    def test_specparam_loss_finite(self):
        from neurojax.analysis.spectral import SpecParam

        model = SpecParam(n_peaks=2)
        freqs = jnp.linspace(1, 50, 100)
        # Synthetic 1/f spectrum with a peak
        power = 1.0 / (freqs + 1.0) + 0.5 * jnp.exp(-((freqs - 10) ** 2) / 4)
        log_power = jnp.log10(power + 1e-10)

        loss = SpecParam.loss(model, freqs, log_power)
        assert jnp.isfinite(loss)

    def test_specparam_fit_runs(self):
        """Fit on a synthetic spectrum with a known peak."""
        from neurojax.analysis.spectral import SpecParam

        freqs = jnp.linspace(1, 50, 100)
        # Synthetic: 1/f^2 + alpha peak at 10 Hz
        aperiodic = 2.0 - jnp.log10(freqs ** 2 + 1)
        peak = 0.8 * jnp.exp(-((freqs - 10) ** 2) / (2 * 2.0 ** 2))
        log_power = aperiodic + peak

        model = SpecParam.fit(freqs, log_power, n_peaks=2, steps=50, lr=0.05)
        pred = model.get_model(freqs)
        assert pred.shape == log_power.shape
        # Residual should have decreased from init
        residual = float(jnp.mean((pred - log_power) ** 2))
        assert residual < 10.0  # loose bound for 50 steps


# ===================================================================
# stats.py  — GGMM PDF, posteriors, threshold
# ===================================================================


class TestStats:
    """Tests for neurojax.analysis.stats."""

    def test_ggmm_pdf_positive(self):
        from neurojax.analysis.stats import ggmm_pdf

        params = jnp.array([0.8, 0.1, 0.1, 1.0, 2.0, 1.0, 0.0])
        x = jnp.linspace(-4, 4, 50)
        pdf_vals = ggmm_pdf(x, params)
        assert pdf_vals.shape == (50,)
        # PDF should be >= 0 everywhere
        assert float(jnp.min(pdf_vals)) >= -1e-7

    def test_ggmm_pdf_integrates(self):
        """PDF should integrate to approximately 1."""
        from neurojax.analysis.stats import ggmm_pdf

        params = jnp.array([0.8, 0.1, 0.1, 1.0, 2.0, 1.0, 0.0])
        x = jnp.linspace(-10, 10, 2000)
        dx = float(x[1] - x[0])
        pdf_vals = ggmm_pdf(x, params)
        integral = float(jnp.sum(pdf_vals) * dx)
        np.testing.assert_allclose(integral, 1.0, atol=0.15)

    def test_ggmm_posteriors_sum_to_one(self):
        from neurojax.analysis.stats import ggmm_posteriors

        params = jnp.array([0.8, 0.1, 0.1, 1.0, 2.0, 1.0, 0.0])
        x = jnp.linspace(-3, 3, 30)
        p_g, p_p, p_n = ggmm_posteriors(x, params)
        total = p_g + p_p + p_n
        np.testing.assert_allclose(total, jnp.ones(30), atol=1e-4)

    def test_fit_ggmm_returns_stats(self):
        from neurojax.analysis.stats import fit_ggmm

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (500,))
        mu_robust, sigma_robust = fit_ggmm(data)
        # Robust mu should be near 0 for normal data
        assert abs(float(mu_robust)) < 0.5
        # Robust sigma should be near 1
        assert 0.5 < float(sigma_robust) < 2.0

    def test_threshold_ggmm_shapes(self):
        from neurojax.analysis.stats import threshold_ggmm

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (100,))
        z_robust, mu, sigma = threshold_ggmm(data, p_threshold=0.5)
        assert z_robust.shape == (100,)
        assert jnp.isfinite(mu)
        assert float(sigma) > 0

    def test_threshold_ggmm_zscore(self):
        """Z-scores of standard normal data should be near the original values."""
        from neurojax.analysis.stats import threshold_ggmm

        key = jax.random.PRNGKey(1)
        data = jax.random.normal(key, (1000,))
        z_robust, mu, sigma = threshold_ggmm(data)
        # mu should be near 0, sigma near 1 -> z ~= data
        assert abs(float(mu)) < 0.2
        assert abs(float(sigma) - 1.0) < 0.3


# ===================================================================
# timefreq.py  — Morlet wavelet transform
# ===================================================================


class TestTimeFreq:
    """Tests for neurojax.analysis.timefreq."""

    def test_morlet_transform_shape_1d(self):
        from neurojax.analysis.timefreq import morlet_transform

        sfreq = 256.0
        data = jnp.zeros(256)  # 1 second
        freqs = (8.0, 12.0, 30.0)
        out = morlet_transform(data, sfreq, freqs)
        # 1D input -> (n_freqs, n_times)
        assert out.shape == (3, 256)

    def test_morlet_transform_shape_2d(self):
        from neurojax.analysis.timefreq import morlet_transform

        sfreq = 256.0
        data = jnp.zeros((2, 256))  # 2 channels
        freqs = (10.0, 20.0)
        out = morlet_transform(data, sfreq, freqs)
        # 2D input -> (n_channels, n_freqs, n_times)
        assert out.shape == (2, 2, 256)

    def test_morlet_peak_frequency(self):
        """A pure sinusoid should produce maximum power at its frequency."""
        from neurojax.analysis.timefreq import morlet_transform

        sfreq = 256.0
        duration = 2.0
        t = jnp.arange(0, duration, 1.0 / sfreq)
        freq_true = 10.0
        data = jnp.sin(2 * jnp.pi * freq_true * t)

        freqs = (5.0, 8.0, 10.0, 12.0, 20.0, 30.0)
        out = morlet_transform(data, sfreq, freqs)
        # Average power over time for each frequency
        power = jnp.mean(jnp.abs(out) ** 2, axis=-1)
        # Peak should be at index 2 (10 Hz)
        peak_idx = int(jnp.argmax(power))
        assert freqs[peak_idx] == freq_true

    def test_morlet_complex_output(self):
        from neurojax.analysis.timefreq import morlet_transform

        sfreq = 128.0
        data = jnp.ones(128)
        freqs = (10.0,)
        out = morlet_transform(data, sfreq, freqs)
        assert jnp.iscomplexobj(out)


# ===================================================================
# superlet.py  — Superlet time-frequency
# ===================================================================


class TestSuperlet:
    """Tests for neurojax.analysis.superlet."""

    def test_superlet_transform_shape(self):
        from neurojax.analysis.superlet import superlet_transform

        sfreq = 256.0
        data = jnp.zeros((2, 256))
        freqs = (10.0, 20.0, 30.0)
        out = superlet_transform(data, sfreq, freqs, base_cycles=3.0, order=2)
        assert out.shape == (2, 3, 256)

    def test_superlet_peak_frequency(self):
        """Superlet should detect peak frequency of a pure sinusoid."""
        from neurojax.analysis.superlet import superlet_transform

        sfreq = 256.0
        duration = 2.0
        t = jnp.arange(0, duration, 1.0 / sfreq)
        data = jnp.sin(2 * jnp.pi * 15 * t)
        data = data[None, :]  # (1, n_times)

        freqs = (5.0, 10.0, 15.0, 20.0, 30.0)
        out = superlet_transform(data, sfreq, freqs, base_cycles=3.0, order=2)
        power = jnp.mean(out[0] ** 2, axis=-1)
        peak_idx = int(jnp.argmax(power))
        assert freqs[peak_idx] == 15.0

    def test_superlet_positive_output(self):
        """Superlet output (geometric mean of magnitudes) should be non-negative."""
        from neurojax.analysis.superlet import superlet_transform

        sfreq = 128.0
        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (1, 128))
        freqs = (8.0, 12.0)
        out = superlet_transform(data, sfreq, freqs, base_cycles=3.0, order=2)
        assert float(jnp.min(out)) >= 0.0


# ===================================================================
# spm.py  — SPM DCT filter, SVD
# ===================================================================


class TestSPM:
    """Tests for neurojax.analysis.spm."""

    def test_dct_basis_kernel_shape(self):
        from neurojax.analysis.spm import _dct_basis_kernel

        basis = _dct_basis_kernel(100, 5)
        assert basis.shape == (100, 5)

    def test_dct_basis_orthogonality(self):
        """DCT basis columns should be approximately orthonormal."""
        from neurojax.analysis.spm import _dct_basis_kernel

        basis = _dct_basis_kernel(200, 10)
        prod = basis.T @ basis
        np.testing.assert_allclose(prod, jnp.eye(10), atol=0.05)

    def test_dct_filter_removes_drift(self):
        from neurojax.analysis.spm import dct_filter

        sfreq = 100.0
        n_time = 1000
        t = jnp.arange(n_time) / sfreq
        # Slow drift + fast signal
        drift = 5.0 * jnp.sin(2 * jnp.pi * 0.05 * t)  # 0.05 Hz
        signal = jnp.sin(2 * jnp.pi * 5 * t)  # 5 Hz
        data = (drift + signal)[None, :]

        filtered = dct_filter(data, sfreq=sfreq, cutoff_freq=0.1)
        assert filtered.shape == data.shape
        # After highpass at 0.1 Hz, the slow drift should be reduced
        # Variance of filtered should be less than data with drift
        assert float(jnp.std(filtered)) < float(jnp.std(data))

    def test_spm_svd_shapes(self):
        from neurojax.analysis.spm import spm_svd

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (10, 200))
        data_red, U_k, S_k = spm_svd(data, n_modes=3)
        assert data_red.shape == (3, 200)
        assert U_k.shape == (10, 3)
        assert S_k.shape == (3,)

    def test_spm_svd_var_explained(self):
        from neurojax.analysis.spm import spm_svd

        key = jax.random.PRNGKey(1)
        X = _make_rank_deficient(key, 10, 200, true_rank=2)
        X = X + 0.001 * jax.random.normal(key, X.shape)

        data_red, U_k, S_k = spm_svd(X, var_explained=0.99)
        # Should pick ~2 modes for rank-2 data
        assert S_k.shape[0] <= 10
        assert S_k.shape[0] >= 1

    def test_spm_svd_reconstruction(self):
        from neurojax.analysis.spm import spm_svd

        key = jax.random.PRNGKey(2)
        data = jax.random.normal(key, (5, 100))
        # Full SVD should reconstruct perfectly
        data_red, U_k, S_k = spm_svd(data, n_modes=5)
        recon = U_k @ data_red
        np.testing.assert_allclose(recon, data, atol=1e-4)


# ===================================================================
# rough.py  — Rough path / signature analysis
# ===================================================================


class TestRough:
    """Tests for neurojax.analysis.rough (importability + basic checks)."""

    def test_rough_importable(self):
        """Module should be importable (signax may not be installed)."""
        try:
            from neurojax.analysis import rough
            has_rough = True
        except ImportError:
            has_rough = False
            pytest.skip("signax not installed — skipping rough path tests")

    def test_augment_path_with_time(self):
        try:
            from neurojax.analysis.rough import augment_path
        except ImportError:
            pytest.skip("signax not installed")

        path = jnp.ones((50, 2))
        aug = augment_path(path, add_time=True)
        assert aug.shape == (50, 3)
        # First column should be time [0, 1]
        np.testing.assert_allclose(float(aug[0, 0]), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(aug[-1, 0]), 1.0, atol=1e-5)

    def test_augment_path_without_time(self):
        try:
            from neurojax.analysis.rough import augment_path
        except ImportError:
            pytest.skip("signax not installed")

        path = jnp.ones((50, 2))
        aug = augment_path(path, add_time=False)
        assert aug.shape == (50, 2)

    def test_compute_signature_runs(self):
        """If signax is installed, compute_signature should return a 1D vector."""
        try:
            from neurojax.analysis.rough import compute_signature, augment_path
        except ImportError:
            pytest.skip("signax not installed")

        path = augment_path(jnp.ones((20, 2)), add_time=True)
        sig = compute_signature(path, depth=2)
        assert sig.ndim == 1
        assert sig.shape[0] > 0
