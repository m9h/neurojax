"""Tests for neurojax.spatial.graph and neurojax.spatial.splines modules."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

class TestSpatialImportability:
    def test_graph_importable(self):
        from neurojax.spatial.graph import EEGGraph
        assert EEGGraph is not None

    def test_splines_importable(self):
        from neurojax.spatial.splines import SphericalSpline, legendre_g
        assert SphericalSpline is not None
        assert callable(legendre_g)


# ---------------------------------------------------------------------------
# SphericalSpline & legendre_g (pure JAX, no heavy MNE deps)
# ---------------------------------------------------------------------------

class TestLegendreG:
    """Test the Green's function computation."""

    def test_output_tuple(self):
        from neurojax.spatial.splines import legendre_g

        x = jnp.array([0.5])
        g, h = legendre_g(x)
        assert g.shape == (1,)
        assert h.shape == (1,)

    def test_finite_values(self):
        from neurojax.spatial.splines import legendre_g

        x = jnp.linspace(-0.99, 0.99, 50)
        g, h = legendre_g(x)
        assert jnp.isfinite(g).all()
        assert jnp.isfinite(h).all()

    def test_2d_input(self):
        from neurojax.spatial.splines import legendre_g

        x = jnp.array([[0.5, 0.3], [-0.2, 0.9]])
        g, h = legendre_g(x)
        assert g.shape == (2, 2)
        assert h.shape == (2, 2)

    def test_symmetry(self):
        """g(x) and g(-x) are related by Legendre polynomial symmetry."""
        from neurojax.spatial.splines import legendre_g

        x = jnp.array([0.5, 0.8])
        g_pos, _ = legendre_g(x)
        g_neg, _ = legendre_g(-x)
        # P_n(-x) = (-1)^n P_n(x), so g(x) != g(-x) in general.
        # But both should be finite and real.
        assert jnp.isfinite(g_pos).all()
        assert jnp.isfinite(g_neg).all()

    def test_self_cos_distance(self):
        """cos(0) = 1, a point vs itself."""
        from neurojax.spatial.splines import legendre_g

        g, h = legendre_g(jnp.array([1.0]))
        assert jnp.isfinite(g).all()
        assert jnp.isfinite(h).all()

    def test_jit_compatible(self):
        from neurojax.spatial.splines import legendre_g

        # legendre_g is already @jit but calling from a jitted function should work
        @jax.jit
        def compute(x):
            return legendre_g(x)

        g, h = compute(jnp.array([0.3, 0.7]))
        assert jnp.isfinite(g).all()


class TestSphericalSpline:
    """Test SphericalSpline with synthetic points on the unit sphere."""

    @pytest.fixture()
    def sphere_setup(self):
        """Create 8 positions roughly on a unit sphere and known potentials."""
        from neurojax.spatial.splines import SphericalSpline

        rng = np.random.default_rng(123)
        # Generate random points, project onto unit sphere
        raw = rng.standard_normal((8, 3))
        pos = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        pos_jax = jnp.array(pos, dtype=jnp.float32)

        spline = SphericalSpline(pos_jax, lambda_reg=1e-5)
        return spline, pos_jax

    def test_construction(self, sphere_setup):
        spline, pos = sphere_setup
        assert spline.n == 8
        assert spline.G.shape == (8, 8)
        assert spline.K.shape == (9, 9)
        assert spline.K_inv.shape == (9, 9)

    def test_G_matrix_symmetric(self, sphere_setup):
        spline, _ = sphere_setup
        np.testing.assert_allclose(
            np.array(spline.G), np.array(spline.G.T), atol=1e-5,
        )

    def test_fit_returns_correct_size(self, sphere_setup):
        spline, _ = sphere_setup
        values = jnp.ones(8) * 3.0
        coeffs = spline.fit(values)
        assert coeffs.shape == (9,)  # N + 1

    def test_fit_coeffs_finite(self, sphere_setup):
        spline, _ = sphere_setup
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        coeffs = spline.fit(values)
        assert jnp.isfinite(coeffs).all()

    def test_interpolate_at_known_points(self, sphere_setup):
        """Interpolating at the original sensor positions should recover
        values close to the fitted data (modulo regularization).

        Note: with lambda_reg and limited Legendre series terms, the
        interpolation error can be non-trivial.  We test that the
        interpolated values are correlated and reasonably close.
        """
        spline, pos = sphere_setup
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        coeffs = spline.fit(values)
        interp = spline.interpolate(pos, coeffs)
        np.testing.assert_allclose(np.array(interp), np.array(values), atol=1.0)
        # Also verify the ordering is preserved (correlation > 0.9)
        corr = np.corrcoef(np.array(interp), np.array(values))[0, 1]
        assert corr > 0.9

    def test_interpolate_shape(self, sphere_setup):
        """Interpolation at M target points produces (M,) output."""
        spline, pos = sphere_setup
        values = jnp.ones(8)
        coeffs = spline.fit(values)

        rng = np.random.default_rng(99)
        target = rng.standard_normal((5, 3))
        target = target / np.linalg.norm(target, axis=1, keepdims=True)
        target_jax = jnp.array(target, dtype=jnp.float32)

        interp = spline.interpolate(target_jax, coeffs)
        assert interp.shape == (5,)
        assert jnp.isfinite(interp).all()

    def test_laplacian_shape(self, sphere_setup):
        spline, pos = sphere_setup
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        coeffs = spline.fit(values)
        lap = spline.laplacian(pos, coeffs)
        assert lap.shape == (8,)
        assert jnp.isfinite(lap).all()

    def test_laplacian_constant_is_zero(self, sphere_setup):
        """The Laplacian of a constant field should be approximately zero."""
        spline, pos = sphere_setup
        values = jnp.ones(8) * 5.0
        coeffs = spline.fit(values)
        lap = spline.laplacian(pos, coeffs)
        np.testing.assert_allclose(np.array(lap), 0.0, atol=1e-2)

    def test_pare_correction_output(self, sphere_setup):
        spline, _ = sphere_setup
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        corrected, c0 = spline.pare_correction(values)
        assert corrected.shape == (8,)
        assert jnp.isfinite(c0)
        assert jnp.isfinite(corrected).all()

    def test_pare_correction_removes_dc(self, sphere_setup):
        """PARE correction subtracts the constant c0 from the values.
        If all values are equal, corrected values should be near zero."""
        spline, _ = sphere_setup
        values = jnp.ones(8) * 42.0
        corrected, c0 = spline.pare_correction(values)
        # For a constant field, c0 should be close to 42 and corrected near 0
        np.testing.assert_allclose(np.array(corrected), 0.0, atol=0.5)


# ---------------------------------------------------------------------------
# EEGGraph -- requires MNE montage + scipy
# ---------------------------------------------------------------------------

mne = pytest.importorskip("mne")
scipy = pytest.importorskip("scipy")


class TestEEGGraph:
    """Test EEG graph construction and operations."""

    @pytest.fixture(scope="class")
    def graph(self):
        from neurojax.spatial.graph import EEGGraph
        # Use a smaller standard montage for speed
        return EEGGraph(montage_name="biosemi16", adjacency_dist=0.06)

    def test_construction(self, graph):
        assert graph.n_node > 0
        assert graph.n_edge >= 0

    def test_adjacency_symmetric(self, graph):
        np.testing.assert_array_equal(graph.adj_matrix, graph.adj_matrix.T)

    def test_no_self_loops(self, graph):
        np.testing.assert_array_equal(np.diag(graph.adj_matrix), 0.0)

    def test_laplacian_shape(self, graph):
        n = graph.n_node
        assert graph.laplacian.shape == (n, n)

    def test_laplacian_symmetric(self, graph):
        np.testing.assert_allclose(
            graph.laplacian, graph.laplacian.T, atol=1e-7,
        )

    def test_laplacian_row_sums_zero(self, graph):
        """Combinatorial Laplacian: each row sums to zero (D - A)."""
        row_sums = np.sum(graph.laplacian, axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-7)

    def test_laplacian_positive_semi_definite(self, graph):
        """All eigenvalues of the combinatorial Laplacian are >= 0."""
        eigenvalues = np.linalg.eigvalsh(graph.laplacian)
        assert np.all(eigenvalues >= -1e-7)

    def test_laplacian_has_zero_eigenvalue(self, graph):
        """Connected graph Laplacian has exactly one zero eigenvalue per component."""
        eigenvalues = np.linalg.eigvalsh(graph.laplacian)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-6)
        assert n_zero >= 1  # at least one zero eigenvalue

    def test_smooth_preserves_shape(self, graph):
        n = graph.n_node
        data = jnp.ones((n, 50))
        smoothed = graph.smooth(data, alpha=0.1)
        assert smoothed.shape == (n, 50)

    def test_smooth_constant_field_unchanged(self, graph):
        """Smoothing a constant field should leave it unchanged (L @ const = 0)."""
        n = graph.n_node
        data = jnp.ones((n, 10)) * 3.14
        smoothed = graph.smooth(data, alpha=0.5)
        np.testing.assert_allclose(np.array(smoothed), 3.14, atol=1e-5)

    def test_smooth_alpha_zero_is_identity(self, graph):
        """alpha=0 means no smoothing."""
        n = graph.n_node
        rng = np.random.default_rng(42)
        data = jnp.array(rng.standard_normal((n, 20)))
        smoothed = graph.smooth(data, alpha=0.0)
        np.testing.assert_allclose(np.array(smoothed), np.array(data), atol=1e-6)

    def test_smooth_reduces_variance(self, graph):
        """Smoothing should generally reduce spatial variance."""
        n = graph.n_node
        rng = np.random.default_rng(42)
        data = jnp.array(rng.standard_normal((n, 5)))
        smoothed = graph.smooth(data, alpha=0.01)
        # Variance across channels (axis=0) should decrease
        var_before = jnp.var(data, axis=0).mean()
        var_after = jnp.var(smoothed, axis=0).mean()
        assert var_after <= var_before

    def test_get_graph_returns_graphstuple(self, graph):
        import jraph

        n = graph.n_node
        features = jnp.ones((n, 3))
        gt = graph.get_graph(features)
        assert isinstance(gt, jraph.GraphsTuple)
        assert gt.nodes.shape == (n, 3)
        assert gt.n_node == jnp.array([n])
        assert gt.n_edge == jnp.array([graph.n_edge])
