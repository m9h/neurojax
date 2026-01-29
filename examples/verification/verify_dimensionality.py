import jax
import jax.numpy as jnp
from neurojax.analysis.dimensionality import PPCA

def verify():
    print("Verifying PPCA Dimensionality Estimation...")
    
    # 1. Generate Synthetic Data
    # True dim = 5, Noise dim = 50
    # Samples = 1000
    key = jax.random.PRNGKey(0)
    d_true = 5
    n_features = 50
    n_samples = 1000
    
    # Latent variables S (5, 1000)
    key, subkey = jax.random.split(key)
    S = jax.random.normal(subkey, (d_true, n_samples)) * 5.0 # Strong signal
    
    # Mixing A (50, 5)
    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, (n_features, d_true))
    
    # Signal
    X_signal = jnp.dot(A, S)
    
    # Noise (50, 1000)
    key, subkey = jax.random.split(key)
    E = jax.random.normal(subkey, (n_features, n_samples)) * 1.0
    
    X = X_signal + E
    
    # 2. Estimate
    evidences = PPCA.get_laplace_evidence(X, max_dim=20)
    k_star = jnp.argmax(evidences) + 1
    
    print(f"True Dimension: {d_true}")
    print(f"Estimated Dimension: {k_star}")
    print(f"Evidences (first 10): {evidences[:10]}")
    
    if k_star == d_true:
        print("[SUCCESS] Correctly recovered dimensionality.")
    else:
        print(f"[FAILURE] Inferred {k_star} instead of {d_true}.")

if __name__ == "__main__":
    verify()
