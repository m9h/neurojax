"""
Verification script for Inverse Solver (SCICO) and Resolution Matrix.
Computes inverse solution and visualizes PSF/CTF.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from neurojax.source import inverse_scico

def main():
    print("Initializing Inverse Solver Verification...")
    
    # 1. Simulate Data
    n_sensors = 32
    n_sources = 64
    key = jax.random.PRNGKey(0)
    
    # Random Leadfield
    L = jax.random.normal(key, (n_sensors, n_sources))
    
    # True Source (Sparse)
    X_true = jnp.zeros((n_sources, 1))
    X_true = X_true.at[10].set(1.0)
    X_true = X_true.at[40].set(-0.5)
    
    # Data Generation
    Y = jnp.dot(L, X_true) + 0.01 * jax.random.normal(key, (n_sensors, 1))
    
    print(f"Simulated Data: Sensors={n_sensors}, Sources={n_sources}")
    
    # 2. Solve Inverse (L1/TV)
    print("Solving Inverse Problem...")
    # Using L1 (Total Variation=0 for unconnected graph approx or just sparsity)
    result = inverse_scico.solve_inverse_admm(
        Y, L, lambda_reg=0.1, rho=1.0, maxiter=50
    )
    X_est = result.sources
    
    print("Inverse Solution Computed.")
    
    # 3. Compute Resolution Matrix
    # For L1/TV, the inverse operator G is non-linear
    # However, we can approximate it or use the linear estimator from Weighted Minimum Norm equivalent
    # Or just compute R = G_lin @ L for a standard MNE to contrast
    
    # Let's compute a standard MNE inverse for Resolution Matrix comparison as SCICO's is harder to linearize
    # Linear Inverse Operator G_mne = L.T @ inv(L @ L.T + lambda * I)
    lambda2 = 0.1
    G_mne = L.T @ jnp.linalg.inv(L @ L.T + lambda2 * jnp.eye(n_sensors))
    
    R = inverse_scico.compute_resolution_matrix(L, G_mne)
    
    # 4. Visualize
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # True vs Est
    axs[0, 0].stem(X_true.flatten(), linefmt='k-', markerfmt='ko', label='True')
    axs[0, 0].stem(X_est.flatten(), linefmt='r--', markerfmt='rx', label='SCICO (L1)')
    axs[0, 0].set_title("Source Reconstruction")
    axs[0, 0].legend()
    
    # Resolution Matrix
    im1 = axs[0, 1].imshow(R, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axs[0, 1].set_title("Resolution Matrix (MNE)")
    plt.colorbar(im1, ax=axs[0, 1])
    
    # PSF (Point Spread Function) for Source 10
    psf_idx = 10
    axs[1, 0].plot(R[psf_idx, :], label=f'PSF (Source {psf_idx})')
    axs[1, 0].set_title(f"Point Spread Function (Idx {psf_idx})")
    axs[1, 0].legend()
    
    # CTF (Crosstalk Function) for Source 10
    ctf_idx = 10
    axs[1, 1].plot(R[:, ctf_idx], label=f'CTF (Source {ctf_idx})', color='orange')
    axs[1, 1].set_title(f"Crosstalk Function (Idx {ctf_idx})")
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig("inverse_verification.png")
    print("Verification plot saved to inverse_verification.png")

if __name__ == "__main__":
    main()
