"""Graph utilities for cortical mesh source imaging.

Converts FreeSurfer triangular meshes into jraph GraphsTuples for
message-passing in the PI-GNN source imaging model. Provides
graph Laplacian, vertex feature computation, and orientation constraints.

Reuses patterns from neurojax.spatial.graph (EEGGraph) adapted for
cortical surface topology rather than electrode distance thresholds.
"""

import jax.numpy as jnp
import numpy as np
import jraph
from typing import Tuple, Optional


def adjacency_from_faces(faces: jnp.ndarray,
                         n_vertices: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract bidirectional edges from triangular faces.

    Each triangle (i, j, k) produces 6 directed edges:
      i→j, j→i, j→k, k→j, i→k, k→i
    Duplicates from shared edges are removed.

    Args:
        faces: (n_faces, 3) triangle vertex indices
        n_vertices: total number of vertices

    Returns:
        senders: (n_edges,) source node indices
        receivers: (n_edges,) target node indices
    """
    faces_np = np.asarray(faces)
    # All 6 directed edges per triangle
    edges = set()
    for tri in faces_np:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if i != j:
            edges.add((i, j))
            edges.add((j, i))
        if j != k:
            edges.add((j, k))
            edges.add((k, j))
        if i != k:
            edges.add((i, k))
            edges.add((k, i))

    if len(edges) == 0:
        return jnp.zeros(0, dtype=jnp.int32), jnp.zeros(0, dtype=jnp.int32)

    edges = np.array(sorted(edges), dtype=np.int32)
    return jnp.array(edges[:, 0]), jnp.array(edges[:, 1])


def mesh_to_graph(vertices: jnp.ndarray,
                  faces: jnp.ndarray,
                  node_features: Optional[jnp.ndarray] = None) -> jraph.GraphsTuple:
    """Convert a triangular mesh to a jraph GraphsTuple.

    Args:
        vertices: (n_vertices, 3) vertex coordinates
        faces: (n_faces, 3) triangle indices
        node_features: optional (n_vertices, d) features; defaults to positions

    Returns:
        jraph.GraphsTuple with mesh topology
    """
    n_vertices = vertices.shape[0]
    senders, receivers = adjacency_from_faces(faces, n_vertices)
    n_edges = len(senders)

    if node_features is None:
        node_features = vertices

    return jraph.GraphsTuple(
        nodes=node_features,
        edges=None,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([n_vertices]),
        n_edge=jnp.array([n_edges]),
        globals=None,
    )


def graph_laplacian(senders: jnp.ndarray,
                    receivers: jnp.ndarray,
                    n_vertices: int) -> jnp.ndarray:
    """Combinatorial graph Laplacian L = D - A.

    Args:
        senders: (n_edges,) source indices
        receivers: (n_edges,) target indices
        n_vertices: number of vertices

    Returns:
        (n_vertices, n_vertices) Laplacian matrix (symmetric, PSD)
    """
    # Build adjacency matrix
    senders_np = np.asarray(senders)
    receivers_np = np.asarray(receivers)
    A = np.zeros((n_vertices, n_vertices), dtype=np.float32)
    for s, r in zip(senders_np, receivers_np):
        A[int(s), int(r)] = 1.0

    # Symmetrise (should already be symmetric from adjacency_from_faces)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)

    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return jnp.array(L)


def compute_vertex_normals(vertices: jnp.ndarray,
                           faces: jnp.ndarray) -> jnp.ndarray:
    """Compute per-vertex normals by averaging face normals.

    Args:
        vertices: (n_vertices, 3)
        faces: (n_faces, 3)

    Returns:
        (n_vertices, 3) unit normals
    """
    verts = np.asarray(vertices)
    faces_np = np.asarray(faces)

    # Face normals
    v0 = verts[faces_np[:, 0]]
    v1 = verts[faces_np[:, 1]]
    v2 = verts[faces_np[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    # Accumulate face normals to vertices
    normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(normals, faces_np[:, i], face_normals)

    # Normalise
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normals = normals / norms

    return jnp.array(normals.astype(np.float32))


def compute_vertex_features(vertices: jnp.ndarray,
                            faces: jnp.ndarray,
                            normals: Optional[jnp.ndarray] = None,
                            curv: Optional[jnp.ndarray] = None,
                            myelin: Optional[jnp.ndarray] = None,
                            depth: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Concatenate multimodal vertex features.

    Features: normals (3) + curvature (1) + myelin (1) + depth (1)
    Missing features are omitted from the output.

    Args:
        vertices: (n_vertices, 3)
        faces: (n_faces, 3)
        normals: (n_vertices, 3) or None (computed from faces)
        curv: (n_vertices,) curvature values or None
        myelin: (n_vertices,) myelin map or None
        depth: (n_vertices,) sulcal depth or None

    Returns:
        (n_vertices, n_features) feature matrix
    """
    parts = []

    if normals is None:
        normals = compute_vertex_normals(vertices, faces)
    parts.append(normals)  # (n, 3)

    if curv is not None:
        parts.append(curv[:, None] if curv.ndim == 1 else curv)
    if myelin is not None:
        parts.append(myelin[:, None] if myelin.ndim == 1 else myelin)
    if depth is not None:
        parts.append(depth[:, None] if depth.ndim == 1 else depth)

    return jnp.concatenate(parts, axis=1)


def orientation_matrix(normals: jnp.ndarray,
                       mode: str = 'fixed') -> jnp.ndarray:
    """Compute orientation constraint matrix.

    Args:
        normals: (n_sources, 3) unit surface normals
        mode: 'fixed' (normal only), 'loose' (±30° from normal),
              'free' (unconstrained 3-component)

    Returns:
        fixed: (n_sources, 3) — unit normal vectors
        loose: (n_sources, 3, 3) — weighted projection allowing tangential
        free: (n_sources, 3, 3) — identity (no constraint)
    """
    n = normals.shape[0]
    normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True).clip(1e-10)

    if mode == 'fixed':
        return normals  # (n_sources, 3)

    elif mode == 'free':
        return jnp.tile(jnp.eye(3), (n, 1, 1))  # (n, 3, 3)

    elif mode == 'loose':
        # Lin et al. (2006): allow tangential with reduced weight
        # O_i = n_i n_i^T + loose * (I - n_i n_i^T)
        # where loose = sin(30°)² ≈ 0.25
        loose_weight = 0.25  # sin²(30°)
        nn = normals[:, :, None] * normals[:, None, :]  # (n, 3, 3)
        eye = jnp.tile(jnp.eye(3), (n, 1, 1))
        O = nn + loose_weight * (eye - nn)
        return O  # (n, 3, 3)

    else:
        raise ValueError(f"Unknown orientation mode: {mode}")
