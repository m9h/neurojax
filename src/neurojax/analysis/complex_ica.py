"""
Complex FastICA Implementation in JAX.

References:
- Bingham & Hyvarinen, "A fast fixed-point algorithm for independent component analysis of complex valued signals", 2000.
- Novel et al., "Complex FastICA for Circular and Non-Circular Sources".

We assume circularity for simplicity (G depends on |y|), which fits oscillatory EEG data well.
Contrast: G(y) = sqrt(0.1 + |y|^2) or log(0.1 + |y|^2)
g(y) = G'(|y|^2) * y  [Not exactly, careful with complex derivs]
Actually, standard Complex FastICA fixed point:
W+ = E[ x * (g(|w^H x|) * w^H x)^* ] - E[ g'(...) * |w^H x| + ... ] W
Let's use the standard "G2" contrast: G(u) = log(0.1 + u), where u = |y|^2.
g(u) = 1 / (0.1 + u).
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

def _complex_sym_decorrelation(W):
    """ W <- W (W^H W)^(-1/2) """
    # W is (n_comp, n_feat)
    # P = W @ W.H
    P = jnp.dot(W, jnp.conj(W.T))
    s, u = jnp.linalg.eigh(P)
    # s are real
    inv_sqrt_s = jnp.diag(1.0 / jnp.sqrt(jnp.maximum(s, 1e-12)))
    # P^(-1/2) = U S^-1/2 U^H
    P_inv_sqrt = jnp.dot(jnp.dot(u, inv_sqrt_s), jnp.conj(u.T))
    return jnp.dot(P_inv_sqrt, W)

@partial(jit, static_argnames=['max_iter', 'tol'])
def complex_fast_ica_step(W, X, max_iter=200, tol=1e-4):
    """
    X: (n_features, n_samples) Complex whitened data.
    W: (n_components, n_features) Complex weights.
    """
    n_samples = X.shape[1]
    
    def body(val):
        i, W_curr, err = val
        
        # y = W^H * x ? No, usually W * x. Let's stick to W @ X.
        # usually W is (k, n). X is (n, T). y is (k, T).
        y = jnp.dot(W_curr, X)
        
        # Non-linearity G(|y|^2) = log(0.1 + |y|^2)
        # g(u) = 1/(0.1 + u) where u = |y|^2
        u = jnp.abs(y)**2
        g_u = 1.0 / (0.1 + u)
        
        # Fixed point rule for circular sources (Bingham 2000):
        # W_new = E[ x * (g(|y|^2) * y)^* ] - E[ g(|y|^2) + g'(|y|^2)|y|^2 ] * W
        # But wait, derivative terms are tricky.
        # Let's use the simpler rule:
        # W_new = E [ x * (y * g(u))^* ] - E[ g(u) + u g'(u) ] * W
        
        # Term 1: E[ x * conj(y * g) ] = E[ x * conj(y) * g ]
        # y_conj_g = conj(y) * g_u
        y_conj_g = jnp.conj(y) * g_u
        term1 = jnp.dot(y_conj_g, X.T).T / n_samples # (k, n_feat) ??
        # Wait shapes:
        # X: (n, T), y: (k, T).
        # E[ x * ... ] should be (n, k). We want (k, n) for W.
        # So E [ (y * g)^* * x^T ] ?
        # Bingham W^H x notation implies W iscol vector.
        # Here W is row vectors.
        # y = W X. 
        # Update: W = E[ (y g)^* X ] - ...
        # (k, T) * (T, n) -> (k, n). This works.
        term1 = jnp.dot(y_conj_g, X.T) / n_samples # (k, n) ?
        # y_conj_g: (k, T). X.T: (T, n). dot -> (k, n). Correct.
        
        # Term 2: Expectation of scalar coeff
        # E [ g(u) + u * g'(u) ]
        # g'(u) = -1 / (0.1 + u)^2
        g_prime_u = -1.0 / ((0.1 + u)**2)
        
        m_term = g_u + u * g_prime_u # (k, T)
        beta = jnp.mean(m_term, axis=1, keepdims=True) # (k, 1)
        # beta is real.
        
        term2 = beta * W_curr # (k, n)
        
        W_new = term1 - term2
        
        # Decorrelate
        W_new = _complex_sym_decorrelation(W_new)
        
        # Convergence
        # 1 - |corr|. Real part of dot product of conjugate?
        # | <w_new, w_old> | should be 1.
        # dot row-wise.
        dots = jnp.sum(W_curr * jnp.conj(W_new), axis=1)
        lim = jnp.abs(dots)
        error = jnp.max(1.0 - lim)
        
        return i + 1, W_new, error
    
    def cond(val):
        i, _, err = val
        return (i < max_iter) & (err > tol)
        
    _, W_final, error = lax.while_loop(cond, body, (0, W, 1.0))
    return W_final

class ComplexICA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mixing_ = None
        self.components_ = None
        self.mean_ = None
        
    def fit(self, X):
        """
        X: (n_features, n_samples) Complex or Real.
        If Real, will cast to complex (analytic signal should be passed externally if desired).
        """
        if not jnp.iscomplexobj(X):
            # Warn? Or assume analytic signal generation is done outside.
            # Just cast for now.
            X = X.astype(jnp.complex64)
            
        n_features, n_samples = X.shape
        self.mean_ = jnp.mean(X, axis=1, keepdims=True)
        X_centered = X - self.mean_
        
        # 1. PCA Whiten (Complex)
        k = self.n_components if self.n_components else n_features
        
        # SVD of Centered Complex Data
        U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        X_white = jnp.sqrt(n_samples) * Vt_k
        
        # 2. Complex FastICA
        key = jax.random.PRNGKey(42)
        # Init W (Real + Imag random)
        W_init = jax.random.normal(key, (k, k), dtype=jnp.complex64)
        W_init = _complex_sym_decorrelation(W_init)
        
        W_ica = complex_fast_ica_step(W_init, X_white)
        
        # 3. Reconstruct
        S_ica = jnp.dot(W_ica, X_white)
        
        # Mixing
        # A = U_k @ diag(S_k / sqrt(N)) @ W^H ?
        # X = A S.
        # X_white = W S ?? No S = W X_white. X_white = W^H S (if unitary W).
        # X_white = W^-1 S. Since W is unitary (decorrelated), W^-1 = W^H.
        # X_cent = U S Vt = U S/sqrt(N) X_white = U S/sqrt(N) W^H S.
        
        dewhitening = jnp.dot(U_k, jnp.diag(S_k/jnp.sqrt(n_samples)))
        mixing = jnp.dot(dewhitening, jnp.conj(W_ica.T))
        
        self.mixing_ = mixing
        self.components_ = S_ica
        
        return self
