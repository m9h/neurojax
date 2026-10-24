"""Tests for higgs_spectral — the jitted, per-frequency HIGGS EM on sensor CSDs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.source.higgs import hermitian_glasso, higgs_spectral, higgs_select_alpha


@pytest.fixture
def spectral_problem():
    """Two frequencies with different sparse source GGMs; sensor CSD from the model."""
    key = jax.random.PRNGKey(3)
    n_sensors, n_sources, n_samples = 16, 6, 4000
    kG, k1, k2 = jax.random.split(key, 3)
    G = jax.random.normal(kG, (n_sensors, n_sources))
    G = G / jnp.linalg.norm(G, axis=0, keepdims=True)

    def precision(edges):
        Th = jnp.eye(n_sources, dtype=jnp.complex64) * 2.0
        for i, j, v in edges:
            Th = Th.at[i, j].set(v).at[j, i].set(jnp.conj(v))
        return Th

    Theta_true = jnp.stack([
        precision([(0, 1, -0.8 + 0.3j), (2, 3, 0.7 - 0.2j)]),
        precision([(4, 5, -0.8 - 0.3j), (1, 2, 0.7 + 0.2j)]),
    ])
    noise_var = 0.05
    csd, src_csd = [], []
    for f, k in enumerate((k1, k2)):
        Sigma = jnp.linalg.inv(Theta_true[f])
        L = jnp.linalg.cholesky(Sigma)
        ka, kb, kc, kd = jax.random.split(k, 4)
        z = (jax.random.normal(ka, (n_sources, n_samples))
             + 1j * jax.random.normal(kb, (n_sources, n_samples))) / jnp.sqrt(2.0)
        x = L @ z
        e = jnp.sqrt(noise_var) * (jax.random.normal(kc, (n_sensors, n_samples))
                                   + 1j * jax.random.normal(kd, (n_sensors, n_samples))) / jnp.sqrt(2.0)
        y = G @ x + e
        csd.append((y @ y.conj().T) / n_samples)
        src_csd.append((x @ x.conj().T) / n_samples)
    return dict(csd=jnp.stack(csd), src_csd=jnp.stack(src_csd), G=G,
                Theta_true=Theta_true, noise_var=noise_var,
                n_sensors=n_sensors, n_sources=n_sources)


def test_hermitian_glasso_is_jittable_and_vmappable():
    S = jnp.stack([jnp.eye(4, dtype=jnp.complex64) * (k + 1.0) for k in range(3)])
    out = jax.jit(jax.vmap(lambda s: hermitian_glasso(s, alpha=0.05, max_iter=10)))(S)
    assert out.shape == (3, 4, 4)
    assert bool(jnp.all(jnp.isfinite(out)))


class TestHiggsSpectral:
    def test_shapes_and_hermitian(self, spectral_problem):
        p = spectral_problem
        out = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=5)
        F, M, N = 2, p["n_sensors"], p["n_sources"]
        assert out["precision"].shape == (F, N, N)
        assert out["source_csd"].shape == (F, N, N)
        assert out["noise_cov"].shape == (F, M, M)
        assert out["kernel"].shape == (F, N, M)
        assert out["nll"].shape == (5,)
        for key in ("precision", "source_csd"):
            A = out[key]
            assert jnp.allclose(A, jnp.swapaxes(A, 1, 2).conj(), atol=1e-5)

    def test_nll_decreases(self, spectral_problem):
        p = spectral_problem
        nll = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=15)["nll"]
        assert bool(jnp.all(jnp.isfinite(nll)))
        assert float(nll[-1]) < float(nll[0])

    def test_recovers_frequency_specific_edges(self, spectral_problem):
        p = spectral_problem
        out = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=20)
        N = p["n_sources"]
        off = ~jnp.eye(N, dtype=bool)
        for f in range(2):
            est = jnp.abs(out["precision"][f]) * off
            true_edges = (jnp.abs(p["Theta_true"][f]) > 1e-6) & off
            # planted edges are the two largest off-diagonal (symmetric) entries
            thresh = jnp.sort(est[off])[-4]
            assert bool(jnp.all(est[true_edges] >= thresh)), f"freq {f}: edges not top-ranked"
        # the two frequencies must differ in their recovered pattern
        assert not jnp.allclose(jnp.abs(out["precision"][0]), jnp.abs(out["precision"][1]), atol=0.05)

    def test_source_csd_tracks_truth(self, spectral_problem):
        p = spectral_problem
        out = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=20)
        for f in range(2):
            est, true = jnp.diag(out["source_csd"][f]).real, jnp.diag(p["src_csd"][f]).real
            corr = jnp.corrcoef(est, true)[0, 1]
            assert float(corr) > 0.7

    def test_jit_and_grad_through_alpha(self, spectral_problem):
        p = spectral_problem

        def loss(alpha):
            return higgs_spectral(p["csd"], p["G"], alpha=alpha, n_iter=3)["nll"][-1]

        g = jax.jit(jax.grad(loss))(0.05)
        assert bool(jnp.isfinite(g))

    def test_scale_equivariant_in_sensor_power(self, spectral_problem):
        """A CSD in different physical units (e.g. T^2/Hz vs unit variance) must
        give the same graph: the EM works on unit-power CSDs internally, so the
        prior, the penalty and the jitter are all relative to the data scale."""
        p = spectral_problem
        s = 1e-4
        ref = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=10)
        out = higgs_spectral(p["csd"] * s, p["G"], alpha=0.05, n_iter=10)
        np.testing.assert_allclose(out["source_csd"], ref["source_csd"] * s, rtol=1e-3, atol=1e-6 * s)
        np.testing.assert_allclose(out["precision"], ref["precision"] / s, rtol=1e-3, atol=1e-6 / s)
        np.testing.assert_allclose(out["noise_cov"], ref["noise_cov"] * s, rtol=1e-3, atol=1e-6 * s)
        np.testing.assert_allclose(out["kernel"], ref["kernel"], rtol=1e-3, atol=1e-6)
        assert bool(jnp.all(jnp.diff(out["nll"]) <= 1e-4 * jnp.abs(out["nll"][0])))


class TestHiggsSelectAlpha:
    def test_ebic_picks_a_sparse_graph_that_keeps_the_planted_edges(self, spectral_problem):
        p = spectral_problem
        alphas = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5]
        sel = higgs_select_alpha(p["csd"], p["G"], alphas, n_eff=4000, gamma=0.5, n_iter=20)
        assert list(sel["alphas"]) == alphas
        assert sel["ebic"].shape == (len(alphas),) and sel["n_edges"].shape == (len(alphas),)
        assert bool(jnp.all(jnp.isfinite(sel["ebic"])))
        # the chosen penalty is the EBIC minimiser and is reported, not hidden
        assert sel["alpha"] == alphas[int(jnp.argmin(sel["ebic"]))]
        # complexity penalty: the selected graph is sparser than the densest one
        assert int(sel["n_edges"][alphas.index(sel["alpha"])]) <= int(sel["n_edges"][0])
        # ...but the four planted edges (two per frequency) survive
        out = sel["result"]
        N = p["n_sources"]
        off = ~jnp.eye(N, dtype=bool)
        for f in range(2):
            true_edges = (jnp.abs(p["Theta_true"][f]) > 1e-6) & off
            assert bool(jnp.all(jnp.abs(out["precision"][f])[true_edges] > 1e-6))


class TestHiggsNoiseGGM:
    """Step 6: full hgLASSO noise precision instead of a diagonal noise model."""

    @pytest.fixture
    def correlated_noise_problem(self, spectral_problem):
        """Re-generate the first frequency with spatially correlated sensor noise
        (nearest-neighbour coupling): a diagonal noise model cannot absorb it."""
        p = spectral_problem
        M, N, n = p["n_sensors"], p["n_sources"], 4000
        key = jax.random.PRNGKey(11)
        ka, kb, kc, kd = jax.random.split(key, 4)
        Sigma = jnp.linalg.inv(p["Theta_true"][0])
        x = jnp.linalg.cholesky(Sigma) @ ((jax.random.normal(ka, (N, n))
                                           + 1j * jax.random.normal(kb, (N, n))) / jnp.sqrt(2.0))
        Theta_e = jnp.eye(M, dtype=jnp.complex64) * 2.0
        for i in range(M - 1):
            Theta_e = Theta_e.at[i, i + 1].set(-0.9).at[i + 1, i].set(-0.9)
        Sigma_e = p["noise_var"] * jnp.linalg.inv(Theta_e)
        e = jnp.linalg.cholesky(Sigma_e) @ ((jax.random.normal(kc, (M, n))
                                             + 1j * jax.random.normal(kd, (M, n))) / jnp.sqrt(2.0))
        y = p["G"] @ x + e
        return dict(csd=((y @ y.conj().T) / n)[None], G=p["G"], Theta_e=Theta_e,
                    Theta_true=p["Theta_true"][:1], n_sources=N, n_sensors=M)

    def test_noise_precision_recovers_sensor_coupling(self, correlated_noise_problem):
        p = correlated_noise_problem
        M = p["n_sensors"]
        off = ~jnp.eye(M, dtype=bool)
        diag_model = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=20)
        full_model = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=20, noise_alpha=0.05)
        # diagonal model: no off-diagonal noise by construction
        assert float(jnp.max(jnp.abs(diag_model["noise_cov"][0] * off))) == 0.0
        # full model: a noise precision is returned and its off-diagonal pattern
        # ranks the planted nearest-neighbour couplings on top
        Th_e = full_model["noise_precision"][0]
        assert Th_e.shape == (M, M)
        est = jnp.abs(Th_e) * off
        true_edges = (jnp.abs(p["Theta_e"]) > 1e-6) & off
        n_true = int(true_edges.sum())
        thresh = jnp.sort(est[off])[-n_true]
        assert float(jnp.mean(est[true_edges] >= thresh)) > 0.8
        # and the fit is at least as good as the diagonal model's
        assert float(full_model["nll"][-1]) <= float(diag_model["nll"][-1]) + 1e-3

    def test_source_edges_survive_with_noise_ggm(self, correlated_noise_problem):
        p = correlated_noise_problem
        out = higgs_spectral(p["csd"], p["G"], alpha=0.05, n_iter=20, noise_alpha=0.05)
        N = p["n_sources"]
        off = ~jnp.eye(N, dtype=bool)
        est = jnp.abs(out["precision"][0]) * off
        true_edges = (jnp.abs(p["Theta_true"][0]) > 1e-6) & off
        thresh = jnp.sort(est[off])[-4]
        assert bool(jnp.all(est[true_edges] >= thresh))
