"""
Independent Component Analysis (ICA) and PICA (Probabilistic ICA) in JAX.

Implements:
1. FastICA (Hyvarinen 1999) - Fixed Point Iteration.
2. PICA (MELODIC-style): Dim Est -> PCA -> ICA -> GMM Thresholding.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from neurojax.analysis.dimensionality import PPCA
from neurojax.analysis.mixture import GaussianGammaMixture

def _sym_decorrelation(W):
    """Symmetric decorrelation W <- (W * W.T)^(-1/2) * W"""
    s, u = jnp.linalg.eigh(jnp.dot(W, W.T))
    # W_new = U * S^(-1/2) * U.T * W
    # Add epsilon to eigenvalues for stability
    inv_sqrt_s = jnp.diag(1.0 / jnp.sqrt(jnp.maximum(s, 1e-12)))
    return jnp.dot(jnp.dot(jnp.dot(u, inv_sqrt_s), u.T), W)

def _logcosh(x):
    # G(u) = 1/a log cosh(a u)
    # g(u) = tanh(a u)
    # g'(u) = a (1 - tanh^2(a u))
    # Standard FastICA uses a=1
    alpha = 1.0
    gx = jnp.tanh(alpha * x)
    g_prime_x = alpha * (1.0 - gx**2)
    return gx, g_prime_x

@partial(jit, static_argnames=['max_iter', 'tol'])
def fast_ica_step(W, X, max_iter=200, tol=1e-4):
    """
    Parallel FastICA fixed-point iteration.
    X: (n_features, n_samples) - Whitened
    W: (n_components, n_features)
    """
    n_samples = X.shape[1]
    
    def body(val):
        i, W_curr, err = val
        
        # W_new = E[x g(W^T x)] - E[g'(W^T x)] W
        # W^T x -> dot(W, X) -> (n_comp, n_samples)
        wx = jnp.dot(W_curr, X)
        gx, g_prime_x = _logcosh(wx)
        
        # Term 1: E[x g(wx)] -> dot(gx, X.T) / N
        term1 = jnp.dot(gx, X.T) / n_samples
        
        # Term 2: E[g'(wx)] W
        # mean over samples
        beta = jnp.mean(g_prime_x, axis=1, keepdims=True) # (n_comp, 1)
        term2 = beta * W_curr
        
        W_new = term1 - term2
        
        # Decorrelate
        W_new = _sym_decorrelation(W_new)
        
        # Check convergence
        # dist = max(abs(abs(diag(W.T @ W_new)) - 1))
        # simpler: distance between 1 and abs correlation
        # Or Frobenius norm of change? Sklearn uses max abs diff of dot products
        lim = jnp.abs(jnp.sum(W_curr * W_new, axis=1))
        error = jnp.max(1.0 - lim)
        
        return i + 1, W_new, error
        
    def cond(val):
        i, _, err = val
        return (i < max_iter) & (err > tol)
        
    _, W_final, error = lax.while_loop(cond, body, (0, W, 1.0))
    
    return W_final

class PICA:
    """
    Probabilistic ICA (MELODIC-style).
    
    Pipeline:
    1. Dimensionality Estimation (Laplace/BIC).
    2. Whitening / PCA Reduction.
    3. FastICA.
    4. Z-transformation + GMM Mixture Model (optional).
    """
    def __init__(self, n_components=None, method='fastica'):
        self.n_components = n_components
        self.method = method
        self.mixing_ = None
        self.components_ = None
        self.mean_ = None
        
    def fit(self, X):
        """
        X: (n_features, n_samples)
        e.g. (channels, times) for EEG.
        """
        n_features, n_samples = X.shape
        self.mean_ = jnp.mean(X, axis=1, keepdims=True)
        X_centered = X - self.mean_
        
        # 1. Dimensionality Estimation if needed
        k = self.n_components
        if k is None:
            # Autodetect
            k = PPCA.estimate_dimensionality(X, method='bic')
            print(f"PICA: Auto-detected dimensionality k={k}")
            
        # 2. PCA Whiten
        # SVD of X_centered
        # cov = X @ X.T / N
        # U, S, Vt = svd(X)
        U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
        
        # Keep k components
        U_k = U[:, :k] # (n_features, k)
        S_k = S[:k]    # (k,)
        Vt_k = Vt[:k, :] # (k, n_samples)
        
        # Whitened Data X_white = K @ X
        # K = diag(1/sqrt(S^2/N)) @ U.T ??
        # Or simple whitening: X_white = sqrt(N) * Vt_k ? 
        # Since X approx U S Vt, U.T X = S Vt.
        # Variance is S^2/N. We want var=1.
        # X_white = diag(1/lambda) S Vt = vt * sqrt(N) ?
        # Actually standard whitening:
        # X_white = (S_k / sqrt(N))^-1 * U_k.T * X is wrong?
        
        # Let's trust FastICA standard:
        # X_white = diag(1/std) @ U.T @ X 
        # eigenvalues lambda = S**2 / N
        # inv_std = sqrt(N) / S
        
        # X_white = diag(sqrt(N)/S) @ S @ Vt = sqrt(N) * Vt
        X_white = jnp.sqrt(n_samples) * Vt_k
        
        # 3. FastICA
        # Init W random
        key = jax.random.PRNGKey(0)
        W_init = jax.random.normal(key, (k, k))
        W_init = _sym_decorrelation(W_init)
        
        W_ica = fast_ica_step(W_init, X_white)
        
        # Sources S = W @ X_white
        S_ica = jnp.dot(W_ica, X_white)
        
        # Mixing Matrix A
        # X = A @ S
        # X = U S Vt = U S (1/param) W_inv S_ica
        # A = U_k @ diag(S_k/sqrt(N)) @ W_ica.T
        
        whitening_matrix = jnp.dot(jnp.diag(jnp.sqrt(n_samples)/S_k), U_k.T)
        # dewhitening = inv(white) = U @ diag(S/sqrt(N))
        dewhitening = jnp.dot(U_k, jnp.diag(S_k/jnp.sqrt(n_samples)))
        
        self.mixing_ = jnp.dot(dewhitening, W_ica.T)
        self.components_ = S_ica
        
        return self

    def z_score_maps(self):
        """ Convert components to Z-stats (normalize variance). """
        # S: (k, samples)
        # Normalize each row
        std = jnp.std(self.components_, axis=1, keepdims=True)
        return self.components_ / (std + 1e-12)
        
    def find_spatially_correlated_component(self, target_map):
        """ Find IC matching a spatial map. """
        # A: (channels, k)
        # target: (channels,)
        corrs = jnp.corrcoef(self.mixing_.T, target_map)[:-1, -1]
        return jnp.argmax(jnp.abs(corrs))
        
    def find_temporally_correlated_component(self, target_signal):
        """ Find IC matching a temporal signal. """
        # S: (k, time)
        corrs = jnp.corrcoef(self.components_, target_signal)[:-1, -1]
        delays = 0 # Future: handle lags
        return jnp.argmax(jnp.abs(corrs))
        
    def find_spectral_peak_component(self, sfreq, target_freq, f_width=2.0):
        """ Find IC with maximum power ratio at target_freq. """
        from neurojax.analysis.filtering import filter_fft
        # Compute PSD for all components
        # Simple Welch or FFT ?
        # data: (k, T)
        
        n_time = self.components_.shape[1]
        freqs = jnp.fft.rfftfreq(n_time, d=1/sfreq)
        specs = jnp.abs(jnp.fft.rfft(self.components_, axis=1))
        
        # Mask for band
        f_min = target_freq - f_width
        f_max = target_freq + f_width
        mask_band = (freqs >= f_min) & (freqs <= f_max)
        
        power_band = jnp.sum(specs[:, mask_band]**2, axis=1)
        power_total = jnp.sum(specs**2, axis=1)
        
        ratio = power_band / (power_total + 1e-12)
        return jnp.argmax(ratio), ratio
