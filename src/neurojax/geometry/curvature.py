"""Discrete differential geometry for cortical surface analysis.

Computes mean curvature (cotangent Laplacian), Gaussian curvature
(angle deficit), shape index (Koenderink 1992), and folding wavelength
on triangulated surfaces.

Implements the geometry from the buckling layer research (Hough 2026),
following Meyer et al. (2003) for the discrete operators.

References:
    Meyer M et al. (2003) Discrete differential-geometry operators
        for triangulated 2-manifolds. In: Visualization and Mathematics III.
    Koenderink JJ & van Doorn AJ (1992) Surface shape and curvature scales.
        Image Vision Comput 10(8):557-564.
"""

import numpy as np
from typing import Tuple


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle indices

    Returns:
        (N, 3) unit normals per vertex
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3), area-weighted

    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return normals / norms


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric vertex areas (1/3 of each incident face area).

    Args:
        vertices: (N, 3)
        faces: (F, 3)

    Returns:
        (N,) area per vertex
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    areas = np.zeros(len(vertices))
    for i in range(3):
        np.add.at(areas, faces[:, i], face_areas / 3.0)

    return areas


def _cotangent_weights(vertices: np.ndarray, faces: np.ndarray):
    """Compute cotangent weights for the Laplace-Beltrami operator.

    For each edge (i,j), the weight is (cot α_ij + cot β_ij) / 2
    where α, β are the angles opposite edge (i,j) in the two
    incident triangles.

    Returns:
        (N, N) sparse-like weight matrix as dense array (for simplicity)
    """
    N = len(vertices)
    W = np.zeros((N, N))

    for tri in faces:
        for local in range(3):
            i = tri[local]
            j = tri[(local + 1) % 3]
            k = tri[(local + 2) % 3]

            # Angle at vertex k opposite edge (i,j)
            vi = vertices[i] - vertices[k]
            vj = vertices[j] - vertices[k]
            cos_angle = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
            cos_angle = np.clip(cos_angle, -0.999, 0.999)
            sin_angle = np.sqrt(1 - cos_angle ** 2)
            cot = cos_angle / (sin_angle + 1e-10)

            W[i, j] += cot / 2.0
            W[j, i] += cot / 2.0

    return W


def mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mean curvature via the cotangent Laplace-Beltrami operator.

    H_i = -0.5 * (Laplacian x_i) · n_i

    where the discrete Laplacian uses cotangent weights (Meyer et al. 2003).

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) triangle indices

    Returns:
        (N,) mean curvature per vertex. Positive for convex (gyral),
        negative for concave (sulcal).
    """
    N = len(vertices)
    A = vertex_areas(vertices, faces)
    A = np.maximum(A, 1e-10)
    nn = vertex_normals(vertices, faces)
    W = _cotangent_weights(vertices, faces)

    # Laplacian: L_i = (1/A_i) * sum_j w_ij * (x_j - x_i)
    H = np.zeros(N)
    for i in range(N):
        laplacian = np.zeros(3)
        for j in range(N):
            if W[i, j] != 0:
                laplacian += W[i, j] * (vertices[j] - vertices[i])
        laplacian /= A[i]
        H[i] = -0.5 * np.dot(laplacian, nn[i])

    return H


def gaussian_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Gaussian curvature via the discrete angle deficit (Gauss-Bonnet).

    K_i = (2π - Σ_f θ_i^f) / A_i

    Args:
        vertices: (N, 3)
        faces: (F, 3)

    Returns:
        (N,) Gaussian curvature per vertex
    """
    N = len(vertices)
    angle_sum = np.zeros(N)

    for tri in faces:
        for local in range(3):
            i = tri[local]
            j = tri[(local + 1) % 3]
            k = tri[(local + 2) % 3]

            vi = vertices[j] - vertices[i]
            vk = vertices[k] - vertices[i]
            cos_angle = np.dot(vi, vk) / (np.linalg.norm(vi) * np.linalg.norm(vk) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_sum[i] += np.arccos(cos_angle)

    A = vertex_areas(vertices, faces)
    A = np.maximum(A, 1e-10)

    return (2 * np.pi - angle_sum) / A


def shape_index(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Shape index (Koenderink & van Doorn 1992).

    SI = (2/π) * arctan(H / sqrt(H² - K))

    Ranges from -1 (concave cup, sulcal fundus) through 0 (saddle,
    sulcal wall) to +1 (convex dome, gyral crown).

    Args:
        vertices: (N, 3)
        faces: (F, 3)

    Returns:
        (N,) shape index in [-1, 1]
    """
    H = mean_curvature(vertices, faces)
    K = gaussian_curvature(vertices, faces)

    # Principal curvatures from H and K:
    # κ1,2 = H ± sqrt(H² - K)
    discriminant = H ** 2 - K
    discriminant = np.maximum(discriminant, 0)  # numerical safety
    sqrt_disc = np.sqrt(discriminant)

    # SI = (2/π) * arctan(H / sqrt(H² - K))
    # When H² ≈ K (umbilical point), use sign of H
    SI = np.where(
        sqrt_disc > 1e-10,
        (2.0 / np.pi) * np.arctan2(H, sqrt_disc),
        np.sign(H)
    )

    return np.clip(SI, -1.0, 1.0)


def curvature_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Summary statistics of the curvature distribution.

    Computes the measures from Ronan, Voets, Hough et al. (2012)
    NeuroImage: intrinsic curvature skewness, negative K fraction,
    and distribution moments that detect cortical changes missed by
    extrinsic measures.

    Args:
        vertices: (N, 3)
        faces: (F, 3)

    Returns:
        dict with keys:
          H_mean, H_std, H_skew: mean curvature statistics
          K_mean, K_std, K_skew: Gaussian curvature statistics
          K_negative_fraction: fraction of vertices with K < 0 (hyperbolic)
          K_positive_fraction: fraction with K > 0 (elliptic)
          SI_mean, SI_std: shape index statistics
    """
    from scipy.stats import skew as scipy_skew

    H = mean_curvature(vertices, faces)
    K = gaussian_curvature(vertices, faces)
    SI = shape_index(vertices, faces)

    valid_H = H[np.isfinite(H)]
    valid_K = K[np.isfinite(K)]
    valid_SI = SI[np.isfinite(SI)]

    return {
        'H_mean': float(np.mean(valid_H)),
        'H_std': float(np.std(valid_H)),
        'H_skew': float(scipy_skew(valid_H)),
        'K_mean': float(np.mean(valid_K)),
        'K_std': float(np.std(valid_K)),
        'K_skew': float(scipy_skew(valid_K)),
        'K_negative_fraction': float((valid_K < 0).mean()),
        'K_positive_fraction': float((valid_K > 0).mean()),
        'SI_mean': float(np.mean(valid_SI)),
        'SI_std': float(np.std(valid_SI)),
        'n_vertices': len(vertices),
    }


def pial_white_curvature_ratio(pial_vertices: np.ndarray,
                                pial_faces: np.ndarray,
                                white_vertices: np.ndarray,
                                white_faces: np.ndarray) -> dict:
    """Ratio of pial to white matter intrinsic curvature.

    From Ronan, Voets, Hough et al. (2012): the pial/white Gaussian
    curvature ratio indexes differential expansion of cortical layers.
    Reduced ratio indicates under-expansion of superficial layers,
    associated with reduced short-range connectivity.

    Args:
        pial_vertices, pial_faces: pial surface mesh
        white_vertices, white_faces: white matter surface mesh

    Returns:
        dict with K_ratio_mean, K_ratio_median, and per-vertex ratio
    """
    K_pial = gaussian_curvature(pial_vertices, pial_faces)
    K_white = gaussian_curvature(white_vertices, white_faces)

    # Vertex correspondence assumed (same topology, FreeSurfer convention)
    assert len(K_pial) == len(K_white), \
        f"Surfaces must have same vertex count: {len(K_pial)} vs {len(K_white)}"

    # Ratio of absolute K (avoids sign issues)
    abs_K_pial = np.abs(K_pial)
    abs_K_white = np.abs(K_white)
    valid = (abs_K_white > 1e-10) & np.isfinite(K_pial) & np.isfinite(K_white)

    ratio = np.full(len(K_pial), np.nan)
    ratio[valid] = abs_K_pial[valid] / abs_K_white[valid]

    return {
        'K_ratio_mean': float(np.nanmean(ratio)),
        'K_ratio_median': float(np.nanmedian(ratio)),
        'K_ratio_per_vertex': ratio,
        'K_pial_skew': float(np.nan) if len(K_pial[np.isfinite(K_pial)]) == 0
                        else float(__import__('scipy.stats', fromlist=['skew']).skew(K_pial[np.isfinite(K_pial)])),
        'K_white_skew': float(np.nan) if len(K_white[np.isfinite(K_white)]) == 0
                        else float(__import__('scipy.stats', fromlist=['skew']).skew(K_white[np.isfinite(K_white)])),
    }


def folding_wavelength(vertices: np.ndarray, faces: np.ndarray,
                       n_bins: int = 50) -> float:
    """Estimate dominant folding wavelength from curvature power spectrum.

    Computes the mean curvature, takes its spatial power spectrum
    (via vertex-to-vertex distance binning), and finds the peak
    spatial frequency.

    Args:
        vertices: (N, 3)
        faces: (F, 3)
        n_bins: number of spatial frequency bins

    Returns:
        float: dominant folding wavelength in mm. Returns NaN if
        no clear folding peak (smooth surface).
    """
    H = mean_curvature(vertices, faces)

    # Estimate spatial autocorrelation of H
    # Sample random vertex pairs and bin by distance
    N = len(vertices)
    n_samples = min(100000, N * 10)
    rng = np.random.RandomState(0)
    idx_i = rng.randint(0, N, n_samples)
    idx_j = rng.randint(0, N, n_samples)

    dists = np.linalg.norm(vertices[idx_i] - vertices[idx_j], axis=1)
    products = H[idx_i] * H[idx_j]

    max_dist = np.percentile(dists, 95)
    bins = np.linspace(0, max_dist, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    autocorr = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (dists >= bins[b]) & (dists < bins[b + 1])
        if mask.sum() > 10:
            autocorr[b] = np.mean(products[mask])

    # Find first zero-crossing (half wavelength)
    autocorr_norm = autocorr / (np.abs(autocorr[0]) + 1e-10)
    zero_crossings = np.where(np.diff(np.sign(autocorr_norm)))[0]

    if len(zero_crossings) > 0:
        half_wavelength = bin_centers[zero_crossings[0]]
        return 2.0 * half_wavelength
    else:
        return float('nan')
