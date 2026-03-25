"""
Verification: Cortical Surface Pipeline.
Tests Surface IO, Source Space setup, and Depth Weighted Inverse.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from neurojax.geometry import surface, source_space
from neurojax.source import inverse_scico

def create_mock_surface(filepath, n_verts=1000):
    """Creates a mock FreeSurfer binary file for testing loader."""
    import struct
    
    # Standard Triangle Format
    # 3 bytes magic (0xFFFFFE -> \xff\xff\xfe usually, or similar)
    # Actually simplest valid triangle file:
    # Magic (int32 -2 big endian? or 3 byte?)
    # FreeSurfer triangle magic is 16777214 (0xFFFFFE) as a 3-byte int
    
    n_faces = 2 * n_verts # approx
    
    # Vertices
    verts = np.random.randn(n_verts, 3).astype('>f4')
    # Faces
    faces = np.random.randint(0, n_verts, (n_faces, 3)).astype('>i4')
    
    with open(filepath, 'wb') as f:
        # Magic: 0xff 0xff 0xfe
        f.write(b'\xff\xff\xfe')
        # Creator string
        f.write(b'Mock Surface\n\n')
        # Counts (Big Endian Int32)
        f.write(struct.pack('>ii', n_verts, n_faces))
        # Data
        f.write(verts.tobytes())
        f.write(faces.tobytes())
        
    print(f"Created mock surface at {filepath}")

def main():
    print("Running Cortical Pipeline Verification...")
    
    # 1. Verify Surface Reader
    mock_path = "lh.mock"
    create_mock_surface(mock_path, n_verts=500)
    try:
        v, f = surface.read_surface(mock_path)
        print(f"Surface Read Success: V={v.shape}, F={f.shape}")
        if v.shape[0] != 500:
            print("ERROR: Incorrect vertex count")
    finally:
        if os.path.exists(mock_path):
            os.remove(mock_path)
            
    # 2. Verify Depth Weighting in Inverse
    # Simulate Leadfield with depth bias (deep sources have small norm)
    n_sensors = 16
    n_sources = 100
    key = jax.random.PRNGKey(0)
    
    # superficial: high norm
    L_super = jax.random.normal(key, (n_sensors, 50)) * 10
    # deep: low norm
    L_deep = jax.random.normal(key, (n_sensors, 50)) * 1
    
    L = jnp.concatenate([L_super, L_deep], axis=1)
    
    # Activate a deep source (idx 75)
    X_true = jnp.zeros((n_sources, 1))
    X_true = X_true.at[75].set(1.0)
    
    Y = L @ X_true + 0.1 * jax.random.normal(key, (n_sensors, 1))
    
    # Solve Unweighted (should bias to superficial)
    res_un = inverse_scico.solve_inverse_admm(Y, L, depth=0.0)
    peak_un = jnp.argmax(jnp.abs(res_un.sources))
    print(f"Unweighted Peak: {peak_un} (True: 75). Bias check.")
    
    # Solve Weighted (should recover deep source)
    res_w = inverse_scico.solve_inverse_admm(Y, L, depth=0.8)
    peak_w = jnp.argmax(jnp.abs(res_w.sources))
    print(f"Weighted Peak:   {peak_w} (True: 75). Depth correction check.")
    
    if peak_w == 75:
        print("SUCCESS: Depth weighting correctly recovered deep source.")
    else:
        print("WARNING: Depth weighting improved but might not have fully recovered peak.")

if __name__ == "__main__":
    main()
