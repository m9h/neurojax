"""
Demo of Riemannian Artifact Detection.
"""
import jax
import jax.numpy as jnp
import numpy as np  # For synthetic data generation
from neurojax.geometry.riemann import covariance_mean, riemannian_distance
from neurojax.preprocessing.artifact import detect_artifacts_riemann

def generate_spd_matrix(n, seed):
    """Generate a random SPD matrix."""
    np.random.seed(seed)
    A = np.random.randn(n, n)
    spd = A @ A.T + np.eye(n) * 0.1
    return spd

def main():
    print("Generating synthetic data...")
    n_channels = 5
    n_epochs_clean = 40
    n_epochs_artifact = 2
    
    # 1. Generate 'clean' epochs (centered around a common mean)
    # We generate a base covariance and perturb it slightly using Wishart-like process
    # Or simplified: generate random SPD matrices with small variance
    
    clean_covs = []
    base_spd = generate_spd_matrix(n_channels, 42)
    
    for i in range(n_epochs_clean):
        # Small perturbation: S = A * M * A^T where A is close to identity
        perturb = np.eye(n_channels) + 0.1 * np.random.randn(n_channels, n_channels)
        c = perturb @ base_spd @ perturb.T
        clean_covs.append(c)
        
    # 2. Generate 'artifact' epochs (far from the mean)
    artifact_covs = []
    for i in range(n_epochs_artifact):
        # Artifact: Scale the whole matrix (simulating high amplitude noise)
        scale = 10.0 + np.random.rand() * 5.0
        c = base_spd * scale
        artifact_covs.append(c)
        
    all_covs = jnp.array(clean_covs + artifact_covs)
    
    print(f"Total epochs: {len(all_covs)}")
    print(f"Clean: {n_epochs_clean}, Artifacts: {n_epochs_artifact}")
    
    # 3. Apply Riemannian Artifact Detection
    print("\nRunning Riemannian Artifact Detection...")
    is_artifact = detect_artifacts_riemann(all_covs, n_std=3.0)
    
    # 4. Analyze Results
    detected_indices = jnp.where(is_artifact)[0]
    print(f"Detected {len(detected_indices)} artifacts at indices: {detected_indices}")
    
    n_detected_clean = jnp.sum(is_artifact[:n_epochs_clean])
    n_detected_artifact = jnp.sum(is_artifact[n_epochs_clean:])
    
    print(f"False Positives (Clean marked as Artifact): {n_detected_clean}")
    print(f"True Positives (Artifact marked as Artifact): {n_detected_artifact}")
    
    if n_detected_artifact == n_epochs_artifact and n_detected_clean == 0:
        print("\nSUCCESS: Perfectly separated artifacts from clean data.")
    else:
        print("\nWARNING: Detection was not perfect (check thresholds).")

    # 5. Check Mean Computation
    print("\nChecking Fréchet Mean...")
    mean_cov = covariance_mean(all_covs)
    print("Mean Covariance shape:", mean_cov.shape)
    
    # Quick sanity check: Distance from mean to itself should be 0
    d_self = riemannian_distance(mean_cov, mean_cov)
    print(f"Distance(Mean, Mean) = {d_self:.6f} (Should be 0.0)")

if __name__ == "__main__":
    main()
