"""
Inverse Solver using Native JAX ADMM.
Solves Y = LX + n with TV or Sparse priors using Alternating Direction Method of Multipliers.
Replaces the 'scico' dependency with a lightweight native implementation.
"""

from typing import NamedTuple, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import equinox as eqx

class InverseResult(NamedTuple):
    sources: jax.Array
    residuals: jax.Array
    resolution_matrix: Optional[jax.Array] = None

def soft_threshold(x: jax.Array, threshold: float) -> jax.Array:
    """Soft thresholding operator for L1 prox."""
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)

def compute_depth_prior(L: jax.Array, gamma: float = 0.8, limit: float = 1e-4) -> jax.Array:
    """
    Computes depth weighting prior (whitener).
    W = diag(1 / ||L_i||^gamma)
    
    Args:
        L: Leadfield (n_sensors, n_sources)
        gamma: Weighting exponent (0.8 is standard for loose orientation)
        limit: Stability limit for very small norms
        
    Returns:
        W: Weighting matrix (n_sources, n_sources) - represented as diagonal vector (n_sources,)
    """
    # Column norms
    norms = jnp.linalg.norm(L, axis=0)
    # Clip for stability
    norms = jnp.maximum(norms, limit * jnp.max(norms))
    # Weight is inverse power
    weights = 1.0 / (norms ** gamma)
    return weights

def solve_inverse_admm(
    Y: jax.Array,
    L: jax.Array,
    lambda_reg: float = 0.1,
    rho: float = 1.0,
    maxiter: int = 100,
    penalty: str = 'l1',
    depth: float = 0.0 # 0.0 = off, 0.8 = standard
) -> InverseResult:
    """
    Solves the inverse problem using ADMM.
    Minimize 0.5 * ||Y - LX||_2^2 + lambda * R(X)
    
    If depth > 0, applies depth weighting:
    Problem becomes: Minimize ||Y - L W^-1 (WX)||^2 + lambda |WX|
    Let X_w = WX (Weighted source)
    L_w = L W^-1 (Whitened Leadfield)
    Solve for X_w, then X = W^-1 X_w
    """
    n_sensors, n_sources = L.shape
    
    # 0. Depth Weighting
    if depth > 0.0:
        # W vector (diagonal)
        W_diag = compute_depth_prior(L, gamma=depth)
        # We want to solve for X_w such that X = X_w / W
        # Y = L (X_w / W) = (L / W) X_w
        # This means we scale columns of L by 1/W
        L_eff = L / W_diag[None, :] # Broadcast divide columns
        
        # We solve the standard problem for L_eff and X_w
        L_solve = L_eff
    else:
        L_solve = L
        W_diag = None
    
    # Precompute Matrix Inversion for the x-update (Ridge-like step)
    # (L'L + rho I)^-1 L'
    # This is static for the loop
    Lt = L_solve.T
    LtL = Lt @ L_solve
    Identity = jnp.eye(n_sources)
    LinearOp = LtL + rho * Identity
    
    use_woodbury = True
    if use_woodbury:
        # Woodbury Identity setup (same as before but using L_solve)
        inv_rho = 1.0 / rho
        LLt = L_solve @ Lt
        mid_term = jnp.linalg.inv(jnp.eye(n_sensors) + inv_rho * LLt)
        
        def apply_inverse(b):
            term1 = inv_rho * b
            term2 = (inv_rho**2) * (Lt @ (mid_term @ (L_solve @ b)))
            return term1 - term2
    else:
        InvOp = jnp.linalg.inv(LinearOp)
        def apply_inverse(b):
            return InvOp @ b

    # ADMM State
    x = jnp.zeros((n_sources, Y.shape[1])) # Primal
    z = jnp.zeros_like(x)                  # Split variable
    u = jnp.zeros_like(x)                  # Dual
    
    def body_fun(carry):
        i, x, z, u = carry
        
        # 1. x-update: Minimize 0.5||Y - Lx||^2 + (rho/2)||x - z + u||^2
        rhs = Lt @ Y + rho * (z - u)
        x_new = apply_inverse(rhs)
        
        # 2. z-update: Proximal operator of R evaluated at (x + u)
        v = x_new + u
        threshold = lambda_reg / rho
        z_new = soft_threshold(v, threshold)
        
        # 3. u-update: Dual ascent
        u_new = u + x_new - z_new
        
        return (i + 1, x_new, z_new, u_new)

    init_val = (0, x, z, u)
    _, x_final, z_final, u_final = jax.lax.fori_loop(0, maxiter, lambda i, val: body_fun(val), init_val)
    
    # 4. Un-weight if needed
    if W_diag is not None:
        # X = X_w / W
        sources_final = z_final / W_diag[:, None]
    else:
        sources_final = z_final
        
    return InverseResult(sources=sources_final, residuals=Y - L @ sources_final)

def compute_resolution_matrix(
    L: jax.Array,
    G: jax.Array
) -> jax.Array:
    """Computes Resolution Matrix R = G @ L."""
    return jnp.dot(G, L)
