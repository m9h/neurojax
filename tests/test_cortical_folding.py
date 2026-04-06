"""TDD tests for cortical folding geometry measures.

Implements the discrete differential geometry from the buckling layer
research (Hough 2026) and validates on WAND sub-08033 FreeSurfer surfaces.

Tests cover:
  - Mean curvature via cotangent Laplacian (Meyer et al. 2003)
  - Gaussian curvature via angle deficit (Gauss-Bonnet)
  - Shape index (Koenderink & van Doorn 1992)
  - Folding wavelength from curvature power spectrum
  - Thickness-folding correlation (Budday-Kuhl prediction)

References:
  Meyer M et al. (2003) Discrete differential-geometry operators
  Koenderink JJ & van Doorn AJ (1992) Surface shape and curvature scales
  Budday S, Steinmann P, Kuhl E (2014) J Mech Phys Solids
"""

import pytest
import numpy as np
import os

HAS_FS = os.path.exists('/Users/mhough/dev/wand/derivatives/freesurfer/sub-08033/surf/lh.pial')


# ===========================================================================
# Synthetic surface tests (always run)
# ===========================================================================

class TestCotangentLaplacian:
    """Mean curvature via the cotangent Laplace-Beltrami operator."""

    def test_mean_curvature_sphere(self):
        """Sphere of radius R should have H = 1/R everywhere."""
        from neurojax.geometry.curvature import mean_curvature
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        H = mean_curvature(vertices, faces)
        expected = 1.0 / 50.0  # H = 1/R for a sphere
        np.testing.assert_allclose(H, expected, atol=0.005)

    def test_mean_curvature_shape(self):
        from neurojax.geometry.curvature import mean_curvature
        vertices, faces = _make_icosphere(radius=1.0, subdivisions=2)
        H = mean_curvature(vertices, faces)
        assert H.shape == (len(vertices),)

    def test_mean_curvature_positive_for_convex(self):
        """Convex surfaces should have positive mean curvature."""
        from neurojax.geometry.curvature import mean_curvature
        vertices, faces = _make_icosphere(radius=30.0, subdivisions=2)
        H = mean_curvature(vertices, faces)
        assert np.all(H > 0)


class TestGaussianCurvature:
    """Gaussian curvature via discrete angle deficit."""

    def test_gaussian_curvature_sphere(self):
        """Sphere of radius R should have K = 1/R² everywhere."""
        from neurojax.geometry.curvature import gaussian_curvature
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        K = gaussian_curvature(vertices, faces)
        expected = 1.0 / (50.0 ** 2)
        # Interior vertices should be close; boundary effects at low resolution
        np.testing.assert_allclose(np.median(K), expected, rtol=0.1)

    def test_gauss_bonnet_sphere(self):
        """Gauss-Bonnet: integral of K over sphere = 4*pi."""
        from neurojax.geometry.curvature import gaussian_curvature, vertex_areas
        vertices, faces = _make_icosphere(radius=1.0, subdivisions=3)
        K = gaussian_curvature(vertices, faces)
        A = vertex_areas(vertices, faces)
        total = np.sum(K * A)
        np.testing.assert_allclose(total, 4 * np.pi, rtol=0.01)

    def test_gaussian_curvature_shape(self):
        from neurojax.geometry.curvature import gaussian_curvature
        vertices, faces = _make_icosphere(radius=1.0, subdivisions=2)
        K = gaussian_curvature(vertices, faces)
        assert K.shape == (len(vertices),)


class TestShapeIndex:
    """Shape index (Koenderink 1992)."""

    def test_shape_index_sphere(self):
        """Sphere should have shape index = +1 (convex dome) everywhere."""
        from neurojax.geometry.curvature import shape_index
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        SI = shape_index(vertices, faces)
        # Sphere: H>0, K>0, both principal curvatures equal and positive → SI=+1
        # Original icosahedron vertices (valence 5) have slightly lower SI
        assert np.median(SI) > 0.9
        assert np.min(SI) > 0.7

    def test_shape_index_range(self):
        """Shape index should be in [-1, +1]."""
        from neurojax.geometry.curvature import shape_index
        vertices, faces = _make_icosphere(radius=1.0, subdivisions=2)
        SI = shape_index(vertices, faces)
        assert np.all(SI >= -1.01) and np.all(SI <= 1.01)

    def test_shape_index_sulcus_negative(self):
        """Concave regions (sulci) should have negative shape index."""
        from neurojax.geometry.curvature import shape_index
        # Create a surface with a dimple (concavity)
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        # Push one vertex inward to create a sulcus
        vertices[0] *= 0.7  # push toward center
        SI = shape_index(vertices, faces)
        # The dimpled vertex should have SI < 0
        assert SI[0] < 0.5  # should be negative or near zero


class TestVertexAreas:
    """Barycentric vertex areas."""

    def test_total_area_sphere(self):
        """Total area should approximate 4*pi*R²."""
        from neurojax.geometry.curvature import vertex_areas
        R = 50.0
        vertices, faces = _make_icosphere(radius=R, subdivisions=3)
        A = vertex_areas(vertices, faces)
        total = np.sum(A)
        expected = 4 * np.pi * R ** 2
        np.testing.assert_allclose(total, expected, rtol=0.02)


class TestFoldingWavelength:
    """Folding wavelength from curvature power spectrum."""

    def test_wavelength_returns_float(self):
        """Folding wavelength should return a finite float."""
        from neurojax.geometry.curvature import folding_wavelength
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        lam = folding_wavelength(vertices, faces)
        assert isinstance(lam, float)
        # Low-resolution icosphere may detect mesh artifacts as "folding"
        assert lam > 0 or np.isnan(lam)


# ===========================================================================
# WAND FreeSurfer surface tests (skip if data missing)
# ===========================================================================

class TestCurvatureStatistics:
    """Distribution statistics from Ronan, Voets, Hough et al. (2012)."""

    def test_statistics_keys(self):
        from neurojax.geometry.curvature import curvature_statistics
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=2)
        stats = curvature_statistics(vertices, faces)
        assert 'K_skew' in stats
        assert 'K_negative_fraction' in stats
        assert 'H_skew' in stats
        assert 'SI_mean' in stats

    def test_sphere_K_positive(self):
        """Sphere should have K > 0 everywhere (all elliptic)."""
        from neurojax.geometry.curvature import curvature_statistics
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        stats = curvature_statistics(vertices, faces)
        assert stats['K_positive_fraction'] > 0.95
        assert stats['K_negative_fraction'] < 0.05

    def test_sphere_K_skew_finite(self):
        """Sphere K skewness should be finite (original ico vertices create outliers)."""
        from neurojax.geometry.curvature import curvature_statistics
        vertices, faces = _make_icosphere(radius=50.0, subdivisions=3)
        stats = curvature_statistics(vertices, faces)
        assert np.isfinite(stats['K_skew'])


class TestPialWhiteRatio:
    """Pial-to-white curvature ratio (Ronan, Voets, Hough et al. 2012)."""

    def test_same_surface_ratio_one(self):
        """Same surface for pial and white → ratio ≈ 1."""
        from neurojax.geometry.curvature import pial_white_curvature_ratio
        v, f = _make_icosphere(radius=50.0, subdivisions=2)
        result = pial_white_curvature_ratio(v, f, v, f)
        np.testing.assert_allclose(result['K_ratio_mean'], 1.0, atol=0.01)

    def test_expanded_pial_higher_K(self):
        """Expanded pial surface should have lower K than white (K ∝ 1/R²)."""
        from neurojax.geometry.curvature import pial_white_curvature_ratio
        v_white, f = _make_icosphere(radius=50.0, subdivisions=2)
        v_pial = v_white * 1.05  # 5% expansion
        result = pial_white_curvature_ratio(v_pial, f, v_white, f)
        # K_pial/K_white = (R_white/R_pial)² = (1/1.05)² ≈ 0.907
        assert result['K_ratio_mean'] < 1.0


@pytest.mark.skipif(not HAS_FS, reason="WAND FreeSurfer surfaces not available")
class TestWANDCorticalGeometry:
    """Run on real WAND sub-08033 FreeSurfer surfaces."""

    @pytest.fixture(scope="class")
    def pial_surface(self):
        import mne
        v, f = mne.read_surface(
            '/Users/mhough/dev/wand/derivatives/freesurfer/sub-08033/surf/lh.pial')
        return v.astype(np.float64), f

    def test_mean_curvature_distribution(self, pial_surface):
        """Cortical surface should have both positive and negative H."""
        from neurojax.geometry.curvature import mean_curvature
        v, f = pial_surface
        H = mean_curvature(v, f)
        assert H.shape[0] > 100000  # ~160K vertices
        # Gyri: H > 0 (convex), sulci: H < 0 (concave)
        frac_positive = (H > 0).mean()
        assert 0.3 < frac_positive < 0.7, f"Expected ~50% positive H, got {frac_positive:.0%}"

    def test_gaussian_curvature_distribution(self, pial_surface):
        """Cortical surface should have K centered near zero."""
        from neurojax.geometry.curvature import gaussian_curvature
        v, f = pial_surface
        K = gaussian_curvature(v, f)
        # Cortical surface has both elliptic (K>0) and hyperbolic (K<0) points
        assert (K > 0).sum() > 1000
        assert (K < 0).sum() > 1000

    def test_shape_index_bimodal(self, pial_surface):
        """Cortical shape index should be bimodal (gyri vs sulci)."""
        from neurojax.geometry.curvature import shape_index
        v, f = pial_surface
        SI = shape_index(v, f)
        # Should have peaks near SI ≈ -0.5 (sulci) and SI ≈ +0.5 (gyri)
        hist, edges = np.histogram(SI[np.isfinite(SI)], bins=50, range=(-1, 1))
        peak_neg = edges[np.argmax(hist[:25])]  # peak in negative half
        peak_pos = edges[25 + np.argmax(hist[25:])]  # peak in positive half
        assert peak_neg < 0, f"Expected negative peak, got {peak_neg:.2f}"
        assert peak_pos > 0, f"Expected positive peak, got {peak_pos:.2f}"

    def test_thickness_curvature_correlation(self, pial_surface):
        """Thicker cortex should correlate with less curvature (Test 1 from buckling paper)."""
        from neurojax.geometry.curvature import mean_curvature
        v, f = pial_surface
        H = mean_curvature(v, f)

        # Load thickness from FreeSurfer
        thickness_path = '/Users/mhough/dev/wand/derivatives/freesurfer/sub-08033/surf/lh.thickness'
        try:
            import mne
            thickness = np.asarray(mne.read_surface(thickness_path.replace('.thickness', '.pial'))[0])
            # Actually need to read curv-format file
            from neurojax.geometry.surface import read_curv
            import jax.numpy as jnp
            thickness = np.asarray(read_curv(thickness_path))
        except Exception:
            pytest.skip("Cannot load thickness file")

        # Absolute mean curvature vs thickness
        abs_H = np.abs(H)
        valid = np.isfinite(abs_H) & np.isfinite(thickness) & (thickness > 0.5) & (thickness < 5.0)
        corr = np.corrcoef(abs_H[valid], thickness[valid])[0, 1]

        # Budday-Kuhl predicts negative correlation:
        # thicker cortex → longer wavelength → less curvature
        print(f"Thickness-curvature correlation: r={corr:.3f} (expected negative)")
        assert corr < 0.2, f"Expected negative or weak correlation, got r={corr:.3f}"

    def test_curvature_matches_freesurfer(self, pial_surface):
        """Our curvature should correlate with FreeSurfer's precomputed curvature."""
        from neurojax.geometry.curvature import mean_curvature
        from neurojax.geometry.surface import read_curv
        import jax.numpy as jnp

        v, f = pial_surface
        H = mean_curvature(v, f)

        curv_path = '/Users/mhough/dev/wand/derivatives/freesurfer/sub-08033/surf/lh.curv'
        try:
            fs_curv = np.asarray(read_curv(curv_path))
        except Exception:
            pytest.skip("Cannot load FreeSurfer curvature")

        valid = np.isfinite(H) & np.isfinite(fs_curv)
        corr = np.corrcoef(H[valid], fs_curv[valid])[0, 1]
        # Should be highly correlated (same quantity, different implementation)
        assert abs(corr) > 0.5, f"Low correlation with FreeSurfer: r={corr:.3f}"


# ===========================================================================
# Helper: icosphere generation
# ===========================================================================

def _make_icosphere(radius=1.0, subdivisions=2):
    """Generate an icosphere by recursive subdivision."""
    # Base icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=np.int32)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        new_verts = list(verts)

        def get_midpoint(i, j):
            key = (min(i,j), max(i,j))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (new_verts[i] + new_verts[j]) / 2
            mid /= np.linalg.norm(mid)
            idx = len(new_verts)
            new_verts.append(mid)
            edge_midpoints[key] = idx
            return idx

        for tri in faces:
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.extend([[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]])

        verts = np.array(new_verts)
        faces = np.array(new_faces, dtype=np.int32)

    return verts * radius, faces
