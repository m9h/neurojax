
import os
import sys
# Add src to path
sys.path.append(os.path.abspath("src"))

import jax.numpy as jnp
import jax
import numpy as np
from neurojax.analysis.tensor import unfold, mode_dot, hosvd, tucker_to_tensor, fold

def test_unfold_fold():
    print("Testing Unfold/Fold...")
    # Create random tensor (3, 4, 5)
    key = jax.random.PRNGKey(0)
    tensor = jax.random.normal(key, (3, 4, 5))
    
    # Unfold mode 0
    mat0 = unfold(tensor, 0)
    assert mat0.shape == (3, 20)
    # Fold back
    folded0 = fold(mat0, 0, tensor.shape)
    assert jnp.allclose(tensor, folded0), "Fold mode 0 failed"
    
    # Unfold mode 1
    mat1 = unfold(tensor, 1)
    assert mat1.shape == (4, 15) # 3*5
    folded1 = fold(mat1, 1, tensor.shape)
    assert jnp.allclose(tensor, folded1), "Fold mode 1 failed"
    
    print("Unfold/Fold Passed.")

def test_mode_dot():
    print("Testing Mode Dot...")
    tensor = jnp.ones((2, 3, 4))
    mat = jnp.ones((5, 3)) # J x I_n = 5 x 3. Mode 1 is dim 3.
    
    # Result should be (2, 5, 4)
    res = mode_dot(tensor, mat, 1)
    assert res.shape == (2, 5, 4)
    
    print("Mode Dot Passed.")

def test_hosvd_reconstruction():
    print("Testing HOSVD Reconstruction...")
    # Create rank-1 tensor
    # T = u o v o w
    u = jnp.array([1., 2., 3.]) # (3,)
    v = jnp.array([1., 0.])     # (2,)
    w = jnp.array([1., 1., 1., 1.]) # (4,)
    
    # T(i,j,k) = u[i]*v[j]*w[k]
    # Use einsum
    T = jnp.einsum('i,j,k->ijk', u, v, w)
    
    # HOSVD
    core, factors = hosvd(T) # Full rank
    
    # Reconstruct
    T_rec = tucker_to_tensor(core, factors)
    
    error = jnp.linalg.norm(T - T_rec)
    print(f"Reconstruction Error: {error}")
    assert error < 1e-5, "Reconstruction failed"
    
    # Check core size
    # For rank 1 tensor, core should be sparse or have 1 dominant element if rotated properly?
    # HOSVD core is all-orthogonal.
    print("Core shape:", core.shape)
    print("Factors shapes:", [f.shape for f in factors])
    
    print("HOSVD Passed.")

if __name__ == "__main__":
    test_unfold_fold()
    test_mode_dot()
    test_hosvd_reconstruction()
