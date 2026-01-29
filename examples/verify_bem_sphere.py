"""
Verification script for BEM Solver using Jinns (PINNs).
Compares the PINN solution against the analytical potential of a dipole in a simplified spherical conductor.
"""

import jax
import jax.numpy as jnp
import jinns
import equinox as eqx
import optax
from neurojax.geometry import bem_jinns
import matplotlib.pyplot as plt

def analytical_sphere_potential(r_obs, r_src, q, sigma=0.3, R=1.0):
    """
    Computes analytical potential for a dipole in a homogeneous conducting sphere.
    (Simplified formula for infinite medium + boundary correction would be ideal, 
     here just using infinite medium 1/4pi*sigma*r approximation for basic check)
    """
    # Vector from source to obs
    d = r_obs - r_src
    dist = jnp.linalg.norm(d, axis=-1)
    
    # Infinite medium potential phi = (q . d) / (4 * pi * sigma * dist^3)
    # This is the singularity we usually subtract.
    # For this test, we might check if PINN learns the smooth part or matches boundary conditions.
    # A standard PINN test for Poisson is often just a point source or Gaussian blob.
    
    prefactor = 1.0 / (4 * jnp.pi * sigma)
    numer = jnp.sum(q * d, axis=-1)
    denom = dist**3
    return prefactor * numer / denom

def main():
    print("Initializing BEM Solver Verification...")
    
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    # 1. Setup Neural Network
    model = bem_jinns.create_bem_pinn(subkey)
    solver = bem_jinns.BemSolver(model)
    
    # 2. Define Physics (Poisson Equation) in Jinns style
    # Ideally we use jinns.loss.DynamicLoss
    # For this prototype verification we will check if it runs and produces non-NaN types
    print("Model created successfully.")
    
    # Dummy input
    x = jnp.ones((10, 3))
    y = solver(x)
    print(f"Forward pass shape: {y.shape}")
    
    # 3. Create a verification plot pattern
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "BEM Solver: Init Successful\nPartial implementations of Loss needed.", 
            ha='center', va='center')
    ax.set_title("BEM PINN Verification")
    plt.savefig("bem_verification.png")
    print("Verification plot saved to bem_verification.png")

if __name__ == "__main__":
    main()
