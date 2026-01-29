"""
Spherical Spline Interpolation for EEG.

Implements Perrin et al. (1989) Spherical Splines for:
1. Interpolation (filling bad channels / topography).
2. Surface Laplacian (CSD) - Analytical 2nd derivative.
3. PARE Correction - Integrating potential over the sphere to finding the true zero (infinity reference).
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# Constants for Perrin Splines
M = 4  # Order of spline (typically 2, 3, or 4). m=4 is standard.

@jit
def legendre_g(x: jnp.ndarray, m: int = 4, n_terms: int = 50) -> jnp.ndarray:
    """
    Compute the g_m(x) function (Green's function) for Spherical Splines.
    Sum of Legendre Polynomials.
    
    g_m(x) = (1/4pi) * Sum_{n=1}^inf [ (2n+1) / (n(n+1))^m * P_n(x) ]
    
    Args:
        x: Cosine of angle between points [-1, 1].
        m: Order of spline (smoothness). Higher m = smoother.
        n_terms: Truncation of infinite series.
    """
    # Recurrence relation for Legendre Polynomials P_n(x)
    # P_0 = 1, P_1 = x
    # (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
    
    def loop_body(n, state):
        p_prev, p_curr, accum_g, accum_h = state
        
        # Compute terms for n
        # Term coefficient for g_m
        if n == 0:
             # n=0 term is usually skipped in summation n=1..inf for g_m
             # But let's follow Perrin exactly. usually n=1.
             term_coef = 0.0
             term_h_coef = 0.0
        else:
             denom = (n * (n + 1)) ** m
             term_coef = (2 * n + 1) / denom
             
             # Term for Laplacian (h_m)
             # h_m(x) coefficients are derived from -n(n+1) * coef
             # Laplacian eigenvalue on sphere is -n(n+1)
             term_h_coef = -n * (n + 1) * term_coef

        accum_g = accum_g + term_coef * p_curr
        accum_h = accum_h + term_h_coef * p_curr
        
        # Next Legendre P_{n+1}
        # (n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}
        # P_{n+1} = ((2n+1)xP_n - nP_{n-1}) / (n+1)
        p_next = ((2 * n + 1) * x * p_curr - n * p_prev) / (n + 1)
        
        return p_curr, p_next, accum_g, accum_h

    # Initial state: P_0=1, P_1=x
    init_state = (1.0, x, 0.0, 0.0) 
    # Start loop from n=1
    
    # Actually, scan over n=1 to n_terms
    # We need to carry P_n and P_{n-1}
    # Let's fix loop to be clean.
    
    P0 = jnp.ones_like(x)
    P1 = x
    
    # We sum from n=1...
    val_g = jnp.zeros_like(x)
    val_h = jnp.zeros_like(x)
    
    # Manual unroll or scan? Scan is better for n_terms=50
    # State: (P_{n-1}, P_n, sum_g, sum_h)
    
    def body_fun(carry, n):
        p_prev, p_curr, sg, sh = carry
        
        denom = (n * (n + 1)) ** m
        k = (2 * n + 1) / denom
        
        sg = sg + k * p_curr
        sh = sh - n * (n + 1) * k * p_curr
        
        p_next = ((2 * n + 1) * x * p_curr - n * p_prev) / (n + 1)
        
        return (p_curr, p_next, sg, sh), None

    (pn, P_next, g, h), _ = jax.lax.scan(body_fun, (P0, P1, val_g, val_h), jnp.arange(1, n_terms + 1))
    
    const = 1.0 / (4 * jnp.pi)
    return g * const, h * const

class SphericalSpline:
    def __init__(self, pos: jnp.ndarray, lambda_reg: float = 1e-5):
        """
        Args:
            pos: Electrode positions on unit sphere (N, 3).
            lambda_reg: Regularization parameter.
        """
        self.pos = pos
        self.lambda_reg = lambda_reg
        self.n = pos.shape[0]
        
        # Compute G matrix (pairwise Gram matrix)
        # Cosine distance: x . y (since on unit sphere)
        cos_dist = jnp.dot(pos, pos.T)
        # Clip for numerical stability
        cos_dist = jnp.clip(cos_dist, -1.0, 1.0)
        
        self.G, _ = legendre_g(cos_dist, m=M)
        
        # Setup linear system for coefficients C = [c_1 ... c_n, c_0]
        # [ G + lambda*I   1 ] [ C ] = [ V ]
        # [      1^T       0 ] [ c0]   [ 0 ]
        
        eye = jnp.eye(self.n) * lambda_reg
        row1 = jnp.concatenate([self.G + eye, jnp.ones((self.n, 1))], axis=1)
        row2 = jnp.concatenate([jnp.ones((1, self.n)), jnp.zeros((1, 1))], axis=1)
        self.K = jnp.concatenate([row1, row2], axis=0) # (N+1, N+1)
        
        # Precompute inverse? Or solve on fly?
        # Inverse is fine for fixed geometry.
        self.K_inv = jnp.linalg.inv(self.K)

    def fit(self, values: jnp.ndarray):
        """
        Fit spline coefficients to potential values.
        Args:
            values: (N_channels,) potentials.
        Returns:
            Coefficients (N+1,)
        """
        # RHS = [V; 0]
        rhs = jnp.concatenate([values, jnp.array([0.0])])
        return jnp.dot(self.K_inv, rhs)

    def interpolate(self, target_pos: jnp.ndarray, coeffs: jnp.ndarray):
        """
        Interpolate potentials at target positions.
        """
        # Compute g(x) between targets and sensors
        coss = jnp.dot(target_pos, self.pos.T)
        coss = jnp.clip(coss, -1.0, 1.0)
        
        gx, _ = legendre_g(coss, m=M)
        
        # V(r) = sum c_i g(r, r_i) + c_0
        ci = coeffs[:-1]
        c0 = coeffs[-1]
        
        return jnp.dot(gx, ci) + c0
        
    def laplacian(self, target_pos: jnp.ndarray, coeffs: jnp.ndarray):
        """
        Compute Surface Laplacian (CSD) at target positions.
        """
        coss = jnp.dot(target_pos, self.pos.T)
        coss = jnp.clip(coss, -1.0, 1.0)
        
        _, hx = legendre_g(coss, m=M)
        
        ci = coeffs[:-1]
        # Laplacian kills constant c0 term
        return jnp.dot(hx, ci)

    def pare_correction(self, values: jnp.ndarray):
        """
        Compute PARE-corrected potentials.
        1. Fit spline.
        2. Integrate V over sphere (C_integral).
        3. Subtract C_integral from V.
        """
        coeffs = self.fit(values)
        
        # Integral of Spline over sphere:
        # Int(V) = Int(c0) + sum ci * Int(g_m)
        # Int(c0) over 4pi sphere = c0 * 4pi
        # Int(g_m(cos(theta))) over sphere is 0 for n>=1 because P_n are orthogonal to P_0=1
        # WAIT: g_m starts at n=1. 
        # Int(P_n(x)) from -1 to 1 is 0 for n>=1.
        # So the integral of the g_m part is ZERO!
        # Thus, the integral over the sphere is just c0 * 4pi.
        # The average potential is c0.
        
        c0 = coeffs[-1]
        
        # So PARE correction is just removing c0?
        # Ideally yes, if c0 represents the DC offset.
        # Let's verify: V_pare = V - c0.
        
        return values - c0, c0

