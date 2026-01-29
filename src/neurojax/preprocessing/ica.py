import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

class FastICA(eqx.Module):
    n_components: int = eqx.field(static=True)
    whiten_solver: str = eqx.field(static=True)
    max_iter: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    
    mixing_: jnp.ndarray | None = None
    components_: jnp.ndarray | None = None
    mean_: jnp.ndarray | None = None

    def __init__(self, n_components, whiten_solver="eigh", max_iter=200, tol=1e-4):
        self.n_components = n_components
        self.whiten_solver = whiten_solver
        self.max_iter = max_iter
        self.tol = tol
        self.mixing_ = None
        self.components_ = None
        self.mean_ = None

    def fit(self, X, key=None):
        if key is None: key = jax.random.PRNGKey(0)
        n_features, n_samples = X.shape
        mean = jnp.mean(X, axis=1, keepdims=True)
        X_centered = X - mean
        cov = jnp.cov(X_centered)
        d, E = jnp.linalg.eigh(cov)
        idx = jnp.argsort(d)[::-1]
        d = d[idx]
        E = E[:, idx]
        D_inv_sqrt = jnp.diag(1.0 / jnp.sqrt(d + 1e-18))
        K = jnp.dot(D_inv_sqrt, E.T)[:self.n_components, :]
        X_white = jnp.dot(K, X_centered) * jnp.sqrt(n_samples) 
        
        key, subkey = jax.random.split(key)
        W = jax.random.normal(subkey, (self.n_components, self.n_components))
        W = self._sym_decorrelation(W)
        
        def loop_body(val):
            i, W, converged = val
            wx = jnp.dot(W, X_white)
            g_wx = jnp.tanh(wx)
            g_prime_wx = 1.0 - g_wx**2
            term1 = jnp.dot(g_wx, X_white.T) / n_samples
            term2 = jnp.mean(g_prime_wx, axis=1, keepdims=True) * W
            W_new = term1 - term2
            W_new = self._sym_decorrelation(W_new)
            lim = jnp.max(jnp.abs(jnp.abs(jnp.diag(jnp.dot(W_new, W.T))) - 1.0))
            return i + 1, W_new, lim < self.tol
            
        init_val = (0, W, False)
        final_val = jax.lax.while_loop(lambda v: (v[0] < self.max_iter) & (~v[2]), loop_body, init_val)
        _, W_final, _ = final_val
        
        unmixing = jnp.dot(W_final, K)
        mixing = jnp.linalg.pinv(unmixing)
        components = jnp.dot(unmixing, X_centered)
        
        return eqx.tree_at(lambda t: (t.mixing_, t.components_, t.mean_), self, (mixing, components, mean), is_leaf=lambda x: x is None)

    def _sym_decorrelation(self, W):
        K = jnp.dot(W, W.T)
        s, u = jnp.linalg.eigh(K)
        return jnp.dot(jnp.dot(u, jnp.diag(1.0 / jnp.sqrt(s + 1e-18))), u.T) @ W

    def apply(self, X):
         X_centered = X - self.mean_
         unmixing = jnp.linalg.pinv(self.mixing_)
         return jnp.dot(unmixing, X_centered)
