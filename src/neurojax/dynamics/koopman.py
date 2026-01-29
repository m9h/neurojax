import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

class KoopmanEstimator(eqx.Module):
    """
    Koopman Operator Estimator using Exact Dynamic Mode Decomposition (DMD).
    """
    rank: int = 0  # 0 for full rank

    def __init__(self, rank: int = 0):
        self.rank = rank

    def fit(self, X: jax.Array, Y: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Fits the Koopman operator K such that Y Approx K X.
        Uses SVD-based Exact DMD.
        
        Args:
            X: Snapshot matrix at time t (n_features, n_samples). Note orientation!
            Y: Snapshot matrix at time t+1 (n_features, n_samples).
            
        Returns:
            K: Koopman operator (n_features, n_features) or projection.
            eigenvalues: Eigenvalues of K.
            modes: Koopman modes.
        """
        # Exact DMD Agorithm
        # 1. SVD of X
        U, S, Vh = jnp.linalg.svd(X, full_matrices=False)
        V = Vh.T.conj()
        
        # Truncate to rank
        r = self.rank if self.rank > 0 else len(S)
        # Handle case where rank > actual limited rank
        r = min(r, len(S))
        
        Ur = U[:, :r]
        Sr = jnp.diag(S[:r])
        Vr = V[:, :r]
        
        # 2. Compute Atilde (Projection of A onto POD modes)
        # A ~ Y * V * S^-1 * U'
        # Atilde = U' * A * U = U' * Y * V * S^-1
        
        Atilde = Ur.T.conj() @ Y @ Vr @ jnp.linalg.inv(Sr)
        
        # 3. Eigen decomposition of Atilde
        eigenvalues, W = jnp.linalg.eig(Atilde)
        
        # 4. Compute Koopman Modes
        # Phi = Y * V * S^-1 * W
        Phi = Y @ Vr @ jnp.linalg.inv(Sr) @ W
        
        # 5. Reconstruct high-dimensional operator K (optional, usually we keep modes)
        # K = Phi * diag(evals) * pinv(Phi) (conceptually)
        # Or simpler for exact DMD: A_approx = Ur * Atilde * Ur'
        
        K_matrix = Ur @ Atilde @ Ur.T.conj()
        
        return K_matrix, eigenvalues, Phi

    def predict(self, x0: jax.Array, t: int, K: jax.Array) -> jax.Array:
        """
        Predict state at step t given x0.
        x_t = K^t * x0
        """
        # Naive matrix power for now, can be optimized with eigenvalues
        Kt = jnp.linalg.matrix_power(K, t)
        return Kt @ x0
