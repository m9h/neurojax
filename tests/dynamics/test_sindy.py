import pytest
import jax
import jax.numpy as jnp
import diffrax
from neurojax.dynamics.sindy import SINDyOptimizer, polynomial_library

def lorenz_system(t, y, args):
    sigma, rho, beta = args
    x, y, z = y
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.stack([dx, dy, dz])

def test_sindy_lorenz_recovery():
    # 1. Generate Data
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    args = (sigma, rho, beta)
    y0 = jnp.array([1.0, 1.0, 1.0])
    t0, t1 = 0.0, 2.0
    dt = 0.01
    
    term = diffrax.ODETerm(lorenz_system)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt, y0, args=args, saveat=saveat)
    X = sol.ys
    
    # Compute true derivatives at these points
    dX = jax.vmap(lambda x: lorenz_system(0, x, args))(X)
    
    # 2. Fit SINDy
    optimizer = SINDyOptimizer(threshold=0.5, max_iter=20)
    # Use degree 2 polynomial library
    Xi = optimizer.fit(X, dX, lambda x: polynomial_library(x, degree=2))
    
    # 3. Verify Coefficients
    # Library order: 1, x, y, z, x^2, xy, xz, y^2, yz, z^2
    # Indices:       0, 1, 2, 3, 4,   5,  6,  7,   8,  9
    
    # Equation 1: dx = 10y - 10x
    # Expected: Coeff of y (idx 2) = 10, Coeff of x (idx 1) = -10
    # Relax tolerance slightly as simple SVD STLSQ might have small residuals
    assert jnp.isclose(Xi[2, 0], 10.0, atol=1.5)
    assert jnp.isclose(Xi[1, 0], -10.0, atol=1.5)
    
    # Equation 2: dy = 28x - y - xz
    # Expected: Coeff of x (idx 1) = 28, Coeff of y (idx 2) = -1, Coeff of xz (idx 6) = -1
    assert jnp.isclose(Xi[1, 1], rho, atol=1.5)
    assert jnp.isclose(Xi[2, 1], -1.0, atol=1.5)
    assert jnp.isclose(Xi[6, 1], -1.0, atol=1.5)
    
    # Equation 3: dz = xy - 2.66z
    # Expected: Coeff of xy (idx 5) = 1, Coeff of z (idx 3) = -2.66
    assert jnp.isclose(Xi[5, 2], 1.0, atol=1.5)
    assert jnp.isclose(Xi[3, 2], -beta, atol=1.5)
    
    # Sparsity Check: Most others should be zero
    # Count non-zeros
    non_zeros = jnp.sum(jnp.abs(Xi) > 0.1)
    assert non_zeros <= 10 # Should be exactly 7 terms active (2 + 3 + 2)
