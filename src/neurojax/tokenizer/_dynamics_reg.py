"""Koopman-based dynamics regularizer for tokenizer training.

Auxiliary loss encouraging token embeddings to be linearly predictable
in time (Koopman-friendly). Training-time only; does not affect inference.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from neurojax.dynamics.koopman import KoopmanEstimator


class KoopmanRegularizer(eqx.Module):
    """Penalizes token embeddings that lack temporal linear structure.

    Fits a low-rank Koopman operator to the embedding sequence and
    penalizes prediction error + spectral instability.

    Differentiability note: KoopmanEstimator.fit() uses jnp.linalg.svd
    and jnp.linalg.eig which support JAX autodiff, but can be unstable
    near degenerate singular values. Mitigated via rank truncation.
    """

    koopman: KoopmanEstimator
    lambda_k: float = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(self, rank: int = 16, lambda_k: float = 0.01):
        self.rank = rank
        self.lambda_k = lambda_k
        self.koopman = KoopmanEstimator(rank=rank)

    def __call__(self, embeddings: Float[Array, "T D"]) -> Float[Array, ""]:
        """Compute Koopman regularization loss on an embedding sequence.

        Args:
            embeddings: Token embeddings of shape (T, D), one per timestep.

        Returns:
            Scalar regularization loss.
        """
        # DMD expects (features, samples) orientation
        X = embeddings[:-1].T  # (D, T-1)
        Y = embeddings[1:].T  # (D, T-1)

        K, eigenvalues, _ = self.koopman.fit(X, Y)

        # Prediction error
        prediction_error = jnp.mean((Y - K @ X) ** 2)

        # Spectral penalty: encourage eigenvalues near unit circle
        spectral_penalty = jnp.mean((jnp.abs(eigenvalues) - 1.0) ** 2)

        return self.lambda_k * (prediction_error + 0.1 * spectral_penalty)
