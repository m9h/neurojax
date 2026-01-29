
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from neurojax.spatial.splines import SphericalSpline
from neurojax.analysis.rough import compute_signature, sliding_signature, augment_path

def verify_spatial():
    print("\n--- Verifying Spatial Splines ---")
    # 1. Create points on sphere (Random)
    key = jax.random.PRNGKey(0)
    pos = jax.random.normal(key, (32, 3))
    pos = pos / jnp.linalg.norm(pos, axis=1, keepdims=True)
    
    # 2. Define a known function: V(x,y,z) = z (Gradient field)
    # Laplacian of linear function on sphere?
    # Actually, spherical harmonic Y_1,0 is proportional to z.
    # eigenvalue is -1(2) = -2.
    # So Laplacian(z) should be -2z.
    
    values = pos[:, 2] # V = z
    
    # 3. Fit Spline
    spline = SphericalSpline(pos)
    coeffs = spline.fit(values)
    
    # 4. Interpolate back to same points (Self-consistency)
    recon = spline.interpolate(pos, coeffs)
    mse = jnp.mean((recon - values)**2)
    print(f"Interpolation MSE: {mse:.6f}")
    
    if mse > 1e-4:
        print("[FAILURE] Interpolation error too high")
    
    # 5. Check Laplacian
    # Laplacian(z) = -2z theoretically for spherical harmonic l=1
    lap = spline.laplacian(pos, coeffs)
    
    # Note: Spline approximation might differ from analytical slightly due to m=4 smoothing
    # But should be correlated.
    corr = jnp.corrcoef(lap, -2 * values)[0, 1]
    print(f"Laplacian Correlation with Theoretical (-2z): {corr:.4f}")
    if corr < 0.9:
        print("[WARNING] Laplacian correlation low. Check spline order.")
    else:
        print("[SUCCESS] Spatial Spline Verified.")

    # 6. PARE
    v_pare, c0 = spline.pare_correction(values)
    print(f"PARE c0 (should be ~0 for V=z): {c0:.4f}")

def verify_rough():
    print("\n--- Verifying Rough Paths ---")
    # 1. Create a simple path: Sine wave
    t = jnp.linspace(0, 2*jnp.pi, 100)
    x = jnp.sin(t)
    path = x[:, None] # (100, 1)
    
    # 2. Augment
    aug_path = augment_path(path)
    print(f"Augmented Shape: {aug_path.shape}")
    
    # 3. Signature
    try:
        sig = compute_signature(aug_path, depth=3)
        print(f"Signature Shape (Depth 3): {sig.shape}")
        # Sig dim for path in R^2 (Time, X) at depth 3
        # 1 + 2 + 4 + 8 = 15 terms? signax output format varies
        # signax usually returns flattened vector?
        print(f"Signature Vector: {sig[:5]}...")
        print("[SUCCESS] Signax Signature Computed.")
    except Exception as e:
        print(f"[FAILURE] Signax computation failed: {e}")
        
    # 4. Sliding Window
    try:
        # data (T=1000, Ch=1)
        long_data = jnp.sin(jnp.linspace(0, 20, 1000))[:, None]
        sigs = sliding_signature(long_data, window_size=100, stride=50, depth=3)
        print(f"Sliding Signature Shape: {sigs.shape}")
    except Exception as e:
        print(f"[FAILURE] Sliding Signature failed: {e}")

if __name__ == "__main__":
    verify_spatial()
    verify_rough()
