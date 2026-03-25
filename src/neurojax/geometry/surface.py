"""
FreeSurfer Surface IO.
Pure Python readers for FreeSurfer geometry formats (.pial, .white, .sphere) and curvature.
Avoids dependency on 'nibabel' or 'freesurfer' binaries for basic IO.
"""

import struct
import jax.numpy as jnp
import numpy as np
from typing import Tuple

def read_surface(filepath: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reads a FreeSurfer geometry file (triangle format).
    
    Args:
        filepath: Path to the surface file (e.g. lh.white)
        
    Returns:
        vertices: (N, 3) coordinates
        faces: (M, 3) triangle indices
    """
    with open(filepath, 'rb') as f:
        # 1. Magic Number (3 bytes)
        magic = f.read(3)
        magic_int = int.from_bytes(magic, 'big')
        
        # Check standard triangle format magic (dataset dependent, usually simple)
        # Often starts with \xff\xff\xfe for new format or \xff\xff\xff for old quad
        # Or standard hex is 0xFFFFFE for triangle
        
        # If magic is not the standard triangle one, we might need to handle Quad
        # For this implementation assuming standard triangular (FreeSurfer 6+)
        
        # 2. Creator String
        # Ends with two newlines
        while True:
            line = f.readline()
            if line == b'\n' or line == b'\r\n':
                break
            if len(line) == 0: break # EOF safety
            
        # 3. Read Two Integers (Vertices, Faces) - Big Endian
        v_bytes = f.read(4)
        f_bytes = f.read(4)
        num_vertices = struct.unpack('>i', v_bytes)[0]
        num_faces = struct.unpack('>i', f_bytes)[0]
        
        # 4. Read Vertices (Float32 Big Endian, 3 * num_vertices)
        # Use numpy for fast reading
        # 3 coords * 4 bytes = 12 bytes per vertex
        coords_data = f.read(num_vertices * 3 * 4)
        vertices = np.frombuffer(coords_data, dtype='>f4').reshape(num_vertices, 3).astype(np.float32)
        
        # 5. Read Faces (Int32 Big Endian, 3 * num_faces)
        # 3 indices * 4 bytes = 12 bytes per face
        faces_data = f.read(num_faces * 3 * 4)
        faces = np.frombuffer(faces_data, dtype='>i4').reshape(num_faces, 3).astype(np.int32)
        
    return jnp.array(vertices), jnp.array(faces)

def read_curv(filepath: str) -> jnp.ndarray:
    """
    Reads a FreeSurfer curvature file (.curv, .sulc).
    
    Returns:
        curv: (N,) curvature values
    """
    with open(filepath, 'rb') as f:
        # Magic number (3 bytes) usually \xff\xff\xff
        f.seek(0, 2) # Seek end
        filesize = f.tell()
        f.seek(0)
        
        # Header involves magic, num_vertices, num_faces(virtual), vals_per_vertex
        # Standard curv format (new):
        # - 3 bytes magic
        # - int32 num_vertices
        # - int32 num_faces
        # - int32 vals_per_vertex
        
        magic = f.read(3) # typically \xff\xff\xff
        v_bytes = f.read(4)
        num_vertices = struct.unpack('>i', v_bytes)[0]
        f.read(4) # num_faces (unused for curv)
        f.read(4) # vals_per_vertex (usually 1)
        
        # Data
        data = f.read(num_vertices * 4)
        curv = np.frombuffer(data, dtype='>f4').astype(np.float32)
        
    return jnp.array(curv)
