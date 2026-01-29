"""JAX-native Spherical Spline Interpolation."""
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['n_terms'])
def _calc_legendre_coeffs(n_terms=50):
    """
    Calculate the coefficients for the spherical spline function g(x).
    Based on Perrin et al. (1989).
    
    g(x) = (1/(4*pi)) * sum_{n=1}^inf ( (2n+1) / (n^m * (n+1)^m) ) * P_n(x)
    where m is the order of the spline (usually m=4).
    """
    m = 4
    # Ensure n_terms is concrete int for arange
    N = int(n_terms)
    n = jnp.arange(1, N + 1, dtype=jnp.float32)
    coeffs = (2 * n + 1) / (n ** m * (n + 1) ** m)
    return coeffs / (4 * jnp.pi)

@partial(jax.jit, static_argnames=['n_terms'])
def _evaluate_legendre(x, n_terms=50):
    """
    Evaluate Legendre polynomials P_n(x) up to order n_terms.
    Returns array of shape (n_terms, ...).
    Uses recursive relation:
    (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
    """
    N = int(n_terms)
    
    # Initialize P0=1, P1=x
    p0 = jnp.ones_like(x)
    p1 = x
    
    def step(carry, n):
        prev, curr = carry
        # n is the index corresponding to P_n (so curr is P_n)
        # We compute P_{n+1}
        # (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
        
        # Note: In scan loop, `n` will be from `ns` array.
        # But we need to use float math.
        
        n_float = n.astype(jnp.float32)
        
        val = ((2 * n_float + 1) * x * curr - n_float * prev) / (n_float + 1)
        return (curr, val), curr 
        
    ns = jnp.arange(1, N + 1)
    
    # We output P1...PN
    # Start with (P0, P1)
    _, p_values = jax.lax.scan(step, (p0, p1), ns)
    
    return p_values

@partial(jax.jit, static_argnames=['n_terms'])
def _g_function(x, n_terms=50):
    """
    Compute g(x) using the Legendre expansion.
    x is cosine of angle between points.
    """
    coeffs = _calc_legendre_coeffs(n_terms)
    polys = _evaluate_legendre(x, n_terms)
    # Sum over n terms
    # coeffs shape: (n_terms,)
    # polys shape: (n_terms, ...)
    # expand coeffs
    coeffs_exp = coeffs.reshape((-1,) + (1,) * (x.ndim))
    return jnp.sum(coeffs_exp * polys, axis=0)


def spherical_spline_interpolate(data, bad_idx, sensor_coords, n_terms=50, lambda_reg=1e-5):
    """
    Interpolate bad channels using spherical splines.
    
    Args:
        data: shape (n_channels, n_times)
        bad_idx: List or array of indices of bad channels to interpolate.
        sensor_coords: (n_channels, 3) XYZ coordinates on the unit sphere.
                       (Must be normalized to unit sphere!)
        n_terms: Number of terms in Legendre expansion.
        lambda_reg: Regularization parameter.
        
    Returns:
        Interpolated data (n_channels, n_times).
    """
    n_channels = sensor_coords.shape[0]
    good_idx = jnp.setdiff1d(jnp.arange(n_channels), bad_idx)
    
    if len(bad_idx) == 0:
        return data
        
    # Get coordinates
    pos_good = sensor_coords[good_idx]
    pos_bad = sensor_coords[bad_idx]
    
    # Compute G (good, good)
    # Cosine distance between good sensors
    # dot product of unit vectors
    cos_gg = jnp.dot(pos_good, pos_good.T)
    # Clip for numerical stability
    cos_gg = jnp.clip(cos_gg, -1.0, 1.0)
    
    G_gg = _g_function(cos_gg, n_terms)
    
    # Add regularization to diagonal
    eye = jnp.eye(len(good_idx))
    C = G_gg + lambda_reg * eye
    
    # C matrix extended with constant term row/col for sum(c_i) = 0 constraint
    # [ C  1 ]
    # [ 1' 0 ]
    
    n_good = len(good_idx)
    C_ext = jnp.zeros((n_good + 1, n_good + 1))
    C_ext = C_ext.at[:n_good, :n_good].set(C)
    C_ext = C_ext.at[:n_good, n_good].set(1.0)
    C_ext = C_ext.at[n_good, :n_good].set(1.0)
    
    # Compute inverse (pseudoinverse for stability)
    C_inv_ext = jnp.linalg.pinv(C_ext)
    
    # M_proj = C_inv_ext[:n_good, :n_good]
    # M_c0 = C_inv_ext[n_good, :n_good] # shape (n_good,)
    
    M_proj = C_inv_ext[:n_good, :n_good]
    M_c0 = C_inv_ext[n_good, :n_good] 
    
    # Now we compute G (bad, good) to interpolate
    cos_bg = jnp.dot(pos_bad, pos_good.T)
    cos_bg = jnp.clip(cos_bg, -1.0, 1.0)
    G_bg = _g_function(cos_bg, n_terms)
    
    # V_bad = (G_bg @ M_proj + M_c0) @ V_good
    
    W = G_bg @ M_proj + M_c0.reshape(1, -1)
    
    # Apply interpolation
    data_good = data[good_idx]
    data_bad_interp = jnp.dot(W, data_good)
    
    # Reconstruct full data
    # Create output array
    out = data.at[bad_idx].set(data_bad_interp)
    
    return out
