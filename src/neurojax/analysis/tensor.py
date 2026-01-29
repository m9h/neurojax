
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple, List, Optional
import functools

def unfold(tensor: jax.Array, mode: int) -> jax.Array:
    """
    Unfolds (matricizes) a tensor along a specified mode.
    
    Args:
        tensor: input tensor of shape (d1, d2, ..., dN)
        mode: the mode along which to unfold (0-indexed)
        
    Returns:
        matrix: 2D array of shape (d_mode, product(other_dims))
    """
    return jnp.reshape(jnp.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold(matrix: jax.Array, mode: int, shape: Tuple[int, ...]) -> jax.Array:
    """
    Folds a matrix back into a tensor. Wrapper for reshape + moveaxis.
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim) # Shape after unfold is (dim_mode, rest)
    
    # Reshape to (dim_mode, d1, d2... skipping mode ...)
    tensor = jnp.reshape(matrix, full_shape)
    # Move dim 0 back to mode
    return jnp.moveaxis(tensor, 0, mode)

def mode_dot(tensor: jax.Array, matrix_or_vector: jax.Array, mode: int) -> jax.Array:
    """
    n-mode product of a tensor and a matrix (or vector).
    
    Args:
        tensor: input tensor of shape (..., I_n, ...)
        matrix_or_vector: matrix of shape (J, I_n) or vector of shape (I_n,)
        mode: the mode along which to apply the product
        
    Returns:
        tensor: tensor of shape (..., J, ...)
    """
    # X x_n U = U @ X_(n) then fold back
    # but tensordot is more efficient and cleaner in JAX
    if matrix_or_vector.ndim == 2:
        res = jnp.tensordot(tensor, matrix_or_vector, axes=(mode, 1))
        return jnp.moveaxis(res, -1, mode)
    else:
        return jnp.tensordot(tensor, matrix_or_vector, axes=(mode, 0))

def tucker_to_tensor(core: jax.Array, factors: List[jax.Array]) -> jax.Array:
    """
    Reconstructs tensor from Tucker decomposition (Core, [U1, U2, ...]).
    """
    tensor = core
    for i, factor in enumerate(factors):
        tensor = mode_dot(tensor, factor, i)
    return tensor

def hosvd(tensor: jax.Array, ranks: Optional[List[int]] = None) -> Tuple[jax.Array, List[jax.Array]]:
    """
    Higher-Order Singular Value Decomposition (HOSVD).
    
    Computes the Tucker decomposition via SVD on mode-unfoldings.
    
    Args:
        tensor: input tensor
        ranks: Optional list of target ranks for each mode. If None, uses full rank.
        
    Returns:
        core: The core tensor
        factors: List of factor matrices [U1, U2, ...]
    """
    factors = []
    n_modes = tensor.ndim
    
    if ranks is None:
        ranks = list(tensor.shape)
        
    for mode in range(n_modes):
        # 1. Unfold
        unfolded = unfold(tensor, mode)
        # 2. SVD
        # We only need U. shape (I_n, I_n) or (I_n, rank)
        # full_matrices=False gives (I_n, K) where K = min(I_n, prod_others).
        # We truncate to rank.
        U, _, _ = jnp.linalg.svd(unfolded, full_matrices=False)
        
        # Truncate
        r = ranks[mode]
        factors.append(U[:, :r])
        
    # 3. Compute Core
    # G = X x1 U1.T x2 U2.T ...
    # This is equivalent to projecting the tensor onto the factor basis
    core = tensor
    for i, factor in enumerate(factors):
        # mode_dot with factor.T
        # factor is (I, R), factor.T is (R, I)
        core = mode_dot(core, factor.T, i)
        
    return core, factors
