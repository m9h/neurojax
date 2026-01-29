# © MNELAB developers
#
# License: BSD (3-clause)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from neurojax.preprocessing.ica import FastICA

def test_fastica_synthetic():
    """Test FastICA recovery of synthetic signals."""
    # 1. Generate sources
    n_samples = 2000
    time = jnp.linspace(0, 8, n_samples)
    
    s1 = jnp.sin(2 * time)  # Sine wave
    s2 = jnp.sign(jnp.sin(3 * time))  # Square wave
    s3 = jax.random.normal(jax.random.PRNGKey(0), (n_samples,))  # Noise
    
    # Scale
    S = jnp.stack([s1, s2, s3])
    S /= S.std(axis=1, keepdims=True)
    
    # 2. Mix
    n_features = 3
    A = jnp.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = jnp.dot(A, S)
    
    # 3. Fit ICA
    ica = FastICA(n_components=3, max_iter=1000, tol=1e-5)
    
    # Equinox models are immutable, fit returns new instance
    model = ica.fit(X, key=jax.random.PRNGKey(42))
    
    # 4. Recover
    S_rec = model.apply(X)
    
    # 5. Check correlation
    # ICA recovers components with arbitrary sign and permutation.
    # We check max correlation for each source.
    
    correlation_matrix = jnp.corrcoef(S, S_rec)
    # Top-left 3x3 is S vs S, Bottom-right 3x3 is S_rec vs S_rec
    # Top-right 3x3 is S vs S_rec
    cross_corr = correlation_matrix[:3, 3:]
    
    # For each source, there should be one recovered component with high abs correlation
    max_corrs = jnp.max(jnp.abs(cross_corr), axis=1)
    
    print("Max correlations:", max_corrs)
    assert jnp.all(max_corrs > 0.9), "ICA failed to recover sources"

if __name__ == "__main__":
    test_fastica_synthetic()
