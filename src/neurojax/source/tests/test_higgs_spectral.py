"""Tests for higgs_spectral — the jitted, per-frequency HIGGS EM on sensor CSDs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neurojax.source.higgs import hermitian_glasso, higgs_spectral


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
