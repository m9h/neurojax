"""
Synthetic Phantom Validation: Leakage and Resolution Analysis.
Computes Resolution Matrix and Leakage metrics for Native ADMM Inverse Solver.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from neurojax.source import inverse_scico

def compute_metrics(R, true_indices):
    """
    Computes Point Spread Function (PSF) metrics.
    R: Resolution Matrix (Sources x Sources)
    """
    n_sources = R.shape[0]
    metrics = []
    
    for idx in true_indices:
        psf = R[idx, :] # Row of R is PSF
        # Peak Localization Error (PLE)
        peak_idx = jnp.argmax(jnp.abs(psf))
        ple = jnp.abs(peak_idx - idx) # In indices (1D distance)
        
        # Spatial Dispersion (SD)
        # Weighted standard deviation around peak
        # Normalized PDF
        p = jnp.abs(psf)
        p = p / jnp.sum(p)
        grid = jnp.arange(n_sources)
        mean_pos = jnp.sum(grid * p)
        sd = jnp.sqrt(jnp.sum(p * (grid - mean_pos)**2))
        
        # Crosstalk (CTF peak vs PSF peak)
        # CTF is column R[:, idx]
        ctf = R[:, idx]
        ctf_peak = jnp.max(jnp.abs(ctf))
        
        metrics.append({
            "idx": int(idx),
            "ple": float(ple),
            "sd": float(sd),
            "peak_amp": float(psf[idx]),
            "ctf_peak": float(ctf_peak)
        })
    return metrics

def main():
    print("Running Synthetic Phantom Validation...")
    key = jax.random.PRNGKey(42)
    
    # Setup
    n_sensors = 64
    n_sources = 256
    
    # 1. Generate Random Leadfield (Simulating random geometry for this demo)
    # in real app, load a forward solution
    L = jax.random.normal(key, (n_sensors, n_sources))
    # Normalize L columns
    L = L / jnp.linalg.norm(L, axis=0, keepdims=True)
    
    # 2. Compute Inverse Operator G (Approximate Linear for Resolution Matrix)
    # For L1/TV, G is non-linear. 
    # Standard practice for Resolution Matrix in non-linear solvers:
    # Compute Reconstructed map for point sources (Impulse Response)
    # This builds R column by column (or row by row depending on def)
    
    # Let's compute PSF for a few probe sources by actually solving the inverse
    probe_indices = [20, 100, 200]
    R_rows = []
    
    print("\nComputing Point Spread Functions (Impulse Responses)...")
    for idx in probe_indices:
        print(f"  Probing Source {idx}...")
        # X_true = delta function
        X_true = jnp.zeros((n_sources, 1))
        X_true = X_true.at[idx].set(1.0)
        
        # Y = L X
        Y = L @ X_true
        
        # Solve Inverse
        result = inverse_scico.solve_inverse_admm(
            Y, L, lambda_reg=0.1, rho=1.0, maxiter=50
        )
        X_est = result.sources
        
        # The estimated X_est IS the PSF for source idx
        R_rows.append(X_est.flatten())
    
    # Stack to visualize compared PSFs
    R_subset = jnp.stack(R_rows)
    
    # 3. Compute Metrics
    # Since we only computed specific PSFs, we compute metrics on them
    # For a full R, we would need to run N_sources times (expensive) or use MNE approx
    
    # Just for metric calc
    # Reconstruct a partial R for these rows
    # We can pass this to our metric function
    # But let's just do it manually here for the report
    
    fig, axs = plt.subplots(len(probe_indices), 1, figsize=(10, 8))
    
    print("\nMetrics:")
    for i, (idx, psf) in enumerate(zip(probe_indices, R_rows)):
        peak_idx = jnp.argmax(jnp.abs(psf))
        ple = jnp.abs(peak_idx - idx)
        
        # Dispersion
        p = jnp.abs(psf) / jnp.sum(jnp.abs(psf))
        grid = jnp.arange(n_sources)
        center = jnp.sum(grid * p)
        sd = jnp.sqrt(jnp.sum(p * (grid - center)**2))
        
        print(f"Source {idx}: Peak={peak_idx}, PLE={ple:.1f}, SD={sd:.2f}")
        
        axs[i].stem(psf, linefmt='b-', markerfmt='bo', label='PSF')
        axs[i].axvline(idx, color='r', linestyle='--', label='True')
        axs[i].set_title(f"PSF for Source {idx} (PLE={ple:.1f}, SD={sd:.2f})")
        axs[i].legend()
        
    plt.tight_layout()
    plt.savefig("synthetic_leakage.png")
    print("Leakage plot saved to synthetic_leakage.png")

if __name__ == "__main__":
    main()
