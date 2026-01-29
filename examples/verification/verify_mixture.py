import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from neurojax.analysis.mixture import GaussianGammaMixture

def verify():
    print("Verifying Gaussian-Gamma Mixture Model...")
    key = jax.random.PRNGKey(0)
    
    # 1. Generate Synthetic Data
    # 80% Gaussian(0, 1)
    # 10% Pos Gamma (shape=2.0) + shift
    # 10% Neg Gamma 
    n_samples = 2000
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    x_g = jax.random.normal(k1, (int(0.8*n_samples),))
    x_p = jax.random.gamma(k2, 2.0, (int(0.1*n_samples),)) + 2.0 # shift well spread
    x_n = -jax.random.gamma(k3, 2.0, (int(0.1*n_samples),)) - 2.0
    
    x = jnp.concatenate([x_g, x_p, x_n])
    # Shuffle
    x = jax.random.permutation(k4, x)
    
    print(f"Data Stats: mean={x.mean():.2f}, std={x.std():.2f}")
    
    # 2. Fit
    print("Fitting model using optax (adam)...")
    model = GaussianGammaMixture.fit(x, jax.random.PRNGKey(42), steps=3000, lr=0.05)
    
    weights, mu, sigma, ap, bp, an, bn = model.get_params()
    print("Fitted Parameters:")
    print(f" Weights: {weights}")
    print(f" Gaussian: mu={mu:.2f}, sigma={sigma:.2f}")
    print(f" PosTail: alpha={ap:.2f}, beta={bp:.2f}")
    
    # 3. Check Thresholding
    # Theoretical intersection?
    # Let's check P(Active|x) for a point in the tail
    test_pts = jnp.array([0.0, 5.0, -5.0])
    probs = model.posterior_prob(test_pts)
    print(f"P(Active | 0.0) = {probs[0]:.4f} (Expected low)")
    print(f"P(Active | 5.0) = {probs[1]:.4f} (Expected high)")
    print(f"P(Active | -5.0) = {probs[2]:.4f} (Expected high)")
    
    assert probs[0] < 0.2, "Zero should be background"
    assert probs[1] > 0.8, "Tail should be active"
    print("[SUCCESS] GGM Fit and Thresholding Verified.")

if __name__ == "__main__":
    verify()
