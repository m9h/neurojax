import pytest
import jax
import jax.numpy as jnp
from neurojax.dynamics.koopman import KoopmanEstimator

def test_koopman_linear_rotation():
    # Linear system: rotation matrix
    # [ x_t+1 ] = [ cos w  -sin w ] [ x_t ]
    # [ y_t+1 ]   [ sin w   cos w ] [ y_t ]
    
    theta = jnp.pi / 4 # 45 degrees
    K_true = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                        [jnp.sin(theta), jnp.cos(theta)]])
    
    # Generate data
    n_samples = 100
    X = jax.random.normal(jax.random.PRNGKey(0), (2, n_samples))
    Y = K_true @ X
    
    # Estimate K
    estimator = KoopmanEstimator(rank=0) # Full rank
    K_est, evals, modes = estimator.fit(X, Y)
    
    # Check if K_est matches K_true
    assert jnp.allclose(K_est, K_true, atol=1e-5)
    
    # Check eigenvalues (should be e^(+/- i theta))
    expected_evals = jnp.array([jnp.exp(1j * theta), jnp.exp(-1j * theta)])
    # Sort for comparison
    evals_sorted = jnp.sort(jnp.angle(evals))
    expected_sorted = jnp.sort(jnp.angle(expected_evals))
    
    assert jnp.allclose(evals_sorted, expected_sorted, atol=1e-5)

def test_koopman_prediction():
    theta = 0.1
    K_true = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                        [jnp.sin(theta), jnp.cos(theta)]])
    
    x0 = jnp.array([1.0, 0.0])
    
    # Estimator
    estimator = KoopmanEstimator()
    # We cheat and use the true K for prediction testing logic
    pred = estimator.predict(x0, t=10, K=K_true)
    
    # True n-step
    Kt = jnp.linalg.matrix_power(K_true, 10)
    true_pred = Kt @ x0
    
    assert jnp.allclose(pred, true_pred)
