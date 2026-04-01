"""Physics-Informed Graph Neural Network (PI-GNN) for MEG/EEG source imaging.

Combines:
  1. Physics-informed initialisation via truncated SVD pseudo-inverse
  2. Graph message-passing on cortical mesh topology (jraph)
  3. Multimodal vertex features (curvature, myelin, depth, orientation)
  4. Multi-objective loss: data fidelity + smoothness + orientation + sparsity

Architecture:
  Y (sensor data) → L⁺Y (physics init) → concat with vertex features →
  input projection → K rounds of graph convolution → output projection →
  orientation constraint → J (source estimates)

References:
  Bore et al. (2024) — physics-informed DL for source imaging
  Hecker et al. (2021, 2022) — ESINet / ConvDip
  Lin et al. (2006) — loose orientation constraint
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import jraph
import equinox as eqx
import optax
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Physics-informed initialisation
# ---------------------------------------------------------------------------

def estimate_tikhonov_reg(L: jnp.ndarray) -> float:
    """Estimate Tikhonov λ from the singular value spectrum of L.

    Uses the geometric mean of s_max and the noise floor (median of
    bottom-half singular values) as a robust, data-driven estimate.
    This is computed outside JIT so the value is concrete and reportable.

    Args:
        L: (n_sensors, n_sources) leadfield/gain matrix

    Returns:
        float: estimated regularisation parameter λ
    """
    s = jnp.linalg.svd(L, compute_uv=False)
    n = len(s)
    noise_floor = float(jnp.median(s[n // 2:]))
    s_max = float(s[0])
    s_min = float(s[-1])
    reg = float(np.sqrt(s_max * max(noise_floor, s_min * 0.01)))
    return reg


def tikhonov_inverse(Y: jnp.ndarray,
                     L: jnp.ndarray,
                     reg: float = 1e-3) -> jnp.ndarray:
    """Tikhonov-regularised pseudo-inverse via SVD.

    Applies continuous regularisation:

        J = V @ diag(s_i / (s_i² + λ²)) @ U^T @ Y

    This smoothly downweights small singular values rather than discarding
    them entirely, avoiding the cliff-edge sensitivity of truncated SVD.

    IMPORTANT: The reg parameter must always be provided explicitly.
    Use estimate_tikhonov_reg(L) to compute a data-driven value, then
    pass it here. This ensures the regularisation choice is transparent
    and reportable.

    Args:
        Y: (n_sensors, n_times) sensor data
        L: (n_sensors, n_sources) leadfield/gain matrix
        reg: Tikhonov parameter λ (must be concrete, not None)

    Returns:
        (n_sources, n_times) initial source estimate
    """
    U, s, Vt = jnp.linalg.svd(L, full_matrices=False)

    # Tikhonov filter factors: f_i = s_i / (s_i² + λ²)
    filter_factors = s / (s ** 2 + reg ** 2)

    # J = V @ diag(f) @ U^T @ Y
    J = Vt.T @ (filter_factors[:, None] * (U.T @ Y))
    return J


def truncated_svd_inverse(Y: jnp.ndarray,
                          L: jnp.ndarray,
                          rank: int = 20) -> jnp.ndarray:
    """Regularised pseudo-inverse (Tikhonov with auto λ).

    Backward-compatible wrapper. Computes λ from the singular value
    spectrum (ignoring the rank parameter) to avoid arbitrary threshold
    choices.

    Args:
        Y: (n_sensors, n_times) sensor data
        L: (n_sensors, n_sources) leadfield/gain matrix
        rank: ignored (kept for API compatibility)

    Returns:
        (n_sources, n_times) initial source estimate
    """
    reg = estimate_tikhonov_reg(L)
    return tikhonov_inverse(Y, L, reg=reg)


# ---------------------------------------------------------------------------
# Graph convolution layer
# ---------------------------------------------------------------------------

class GraphConvLayer(eqx.Module):
    """Message-passing layer on cortical mesh.

    Node update: h_i = σ(W_self · h_i + W_msg · mean(h_j, j ∈ N(i)) + b)
    """
    weight_self: jnp.ndarray
    weight_msg: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, hidden_dim: int, *, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key)
        scale = 1.0 / jnp.sqrt(hidden_dim)
        self.weight_self = jax.random.normal(k1, (hidden_dim, hidden_dim)) * scale
        self.weight_msg = jax.random.normal(k2, (hidden_dim, hidden_dim)) * scale
        self.bias = jnp.zeros(hidden_dim)

    def __call__(self, x: jnp.ndarray, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: (n_nodes, hidden_dim) node features
            graph: jraph.GraphsTuple with senders/receivers

        Returns:
            (n_nodes, hidden_dim) updated features
        """
        n_nodes = x.shape[0]
        senders = graph.senders
        receivers = graph.receivers

        # Gather sender features
        messages = x[senders]  # (n_edges, hidden)

        # Aggregate: sum messages per receiver, then normalise by degree
        agg = jnp.zeros_like(x)  # (n_nodes, hidden)
        agg = agg.at[receivers].add(messages)

        # Degree per node (for mean aggregation)
        degree = jnp.zeros(n_nodes)
        degree = degree.at[receivers].add(1.0)
        degree = jnp.maximum(degree, 1.0)[:, None]  # (n_nodes, 1)

        msg_mean = agg / degree

        # Update: self-transform + message-transform + bias
        out = x @ self.weight_self + msg_mean @ self.weight_msg + self.bias
        return jax.nn.relu(out)


# ---------------------------------------------------------------------------
# Full PI-GNN model
# ---------------------------------------------------------------------------

class SourceGNN(eqx.Module):
    """Physics-Informed Graph Neural Network for source imaging.

    Args:
        n_features: number of vertex features (normals + curvature + ...)
        n_times: number of time points in input data
        hidden_dim: hidden dimension for graph convolution layers
        n_layers: number of message-passing rounds
        orientation_mode: 'fixed', 'loose', or 'free'
        svd_rank: deprecated, ignored (kept for API compat)
        tikhonov_reg: Tikhonov λ for physics init. Must be a concrete
            float — use estimate_tikhonov_reg(L) to compute from data.
            Default 1e-3.
    """
    input_proj: eqx.nn.Linear
    layers: list
    output_proj: eqx.nn.Linear
    orientation_mode: str
    tikhonov_reg: float

    def __init__(self, n_features: int, n_times: int, hidden_dim: int = 64,
                 n_layers: int = 3, orientation_mode: str = 'fixed',
                 svd_rank: int = 20, tikhonov_reg: float = 1e-3,
                 *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, n_layers + 2)

        # Input: concatenation of J0 (n_times) + vertex features (n_features)
        self.input_proj = eqx.nn.Linear(
            n_features + n_times, hidden_dim, key=keys[0]
        )
        self.layers = [
            GraphConvLayer(hidden_dim, key=keys[i + 1])
            for i in range(n_layers)
        ]
        # Output: hidden → n_times (one scalar per time point per source)
        self.output_proj = eqx.nn.Linear(
            hidden_dim, n_times, key=keys[-1]
        )
        self.orientation_mode = orientation_mode
        self.tikhonov_reg = tikhonov_reg

    def __call__(self, Y: jnp.ndarray, L: jnp.ndarray,
                 graph: jraph.GraphsTuple,
                 vertex_features: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: sensor data → source estimates.

        Args:
            Y: (n_sensors, n_times) sensor measurements
            L: (n_sensors, n_sources) leadfield matrix
            graph: cortical mesh as jraph.GraphsTuple
            vertex_features: (n_sources, n_features) multimodal features

        Returns:
            (n_sources, n_times) estimated source activity
        """
        # 1. Physics-informed init (Tikhonov, auto-regularised)
        J0 = tikhonov_inverse(Y, L, reg=self.tikhonov_reg)  # (n_src, n_times)

        # 2. Concatenate with vertex features
        x = jnp.concatenate([J0, vertex_features], axis=1)  # (n_src, n_times + n_feat)

        # 3. Project to hidden dim
        x = jax.vmap(self.input_proj)(x)  # (n_src, hidden)

        # 4. Graph message-passing
        for layer in self.layers:
            x = layer(x, graph)

        # 5. Output projection
        J = jax.vmap(self.output_proj)(x)  # (n_src, n_times)

        return J

    def physics_loss(self, Y: jnp.ndarray, L: jnp.ndarray,
                     J: jnp.ndarray) -> jnp.ndarray:
        """Data fidelity: ‖Y - L·J‖² / ‖Y‖²."""
        residual = Y - L @ J
        return jnp.sum(residual ** 2) / jnp.maximum(jnp.sum(Y ** 2), 1e-10)

    def smoothness_loss(self, J: jnp.ndarray,
                        laplacian: jnp.ndarray) -> jnp.ndarray:
        """Graph Laplacian smoothness: tr(J^T L J) / n_sources."""
        n_sources = J.shape[0]
        return jnp.trace(J.T @ laplacian @ J) / n_sources

    def orientation_loss(self, J: jnp.ndarray,
                         normals: jnp.ndarray) -> jnp.ndarray:
        """Penalise tangential (off-normal) components.

        For fixed orientation, this is zero (J is scalar per source).
        For free/loose, penalise ‖J_tangential‖².
        """
        # Project J onto normals: J_normal = (J · n) * n
        # J_tangential = J - J_normal
        # For 1-component sources (n_src, n_times), interpret as
        # magnitude along normal — no tangential component
        if J.ndim == 2 and self.orientation_mode == 'fixed':
            return jnp.float32(0.0)

        # For 3-component: J is (n_src, 3, n_times)
        # Not yet implemented — placeholder
        return jnp.float32(0.0)

    def sparsity_loss(self, J: jnp.ndarray) -> jnp.ndarray:
        """L1 sparsity penalty."""
        return jnp.mean(jnp.abs(J))

    def total_loss(self, Y: jnp.ndarray, L: jnp.ndarray,
                   graph: jraph.GraphsTuple,
                   vertex_features: jnp.ndarray,
                   normals: jnp.ndarray,
                   laplacian: jnp.ndarray,
                   lambda_data: float = 1.0,
                   lambda_smooth: float = 0.1,
                   lambda_orient: float = 0.01,
                   lambda_sparse: float = 0.001) -> jnp.ndarray:
        """Combined multi-objective loss."""
        J = self(Y, L, graph, vertex_features)
        loss = (lambda_data * self.physics_loss(Y, L, J)
                + lambda_smooth * self.smoothness_loss(J, laplacian)
                + lambda_orient * self.orientation_loss(J, normals)
                + lambda_sparse * self.sparsity_loss(J))
        return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_source_gnn(model: SourceGNN,
                     Y: jnp.ndarray,
                     L: jnp.ndarray,
                     graph: jraph.GraphsTuple,
                     vertex_features: jnp.ndarray,
                     normals: jnp.ndarray,
                     laplacian: jnp.ndarray,
                     n_steps: int = 200,
                     lr: float = 1e-3,
                     lambda_data: float = 1.0,
                     lambda_smooth: float = 0.01,
                     lambda_orient: float = 0.0,
                     lambda_sparse: float = 0.001
                     ) -> Tuple[SourceGNN, List[float]]:
    """Train the PI-GNN with optax Adam.

    Args:
        model: SourceGNN instance
        Y, L, graph, vertex_features, normals, laplacian: problem data
        n_steps: number of optimisation steps
        lr: learning rate

    Returns:
        (trained_model, loss_history)
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: m.total_loss(
                Y, L, graph, vertex_features, normals, laplacian,
                lambda_data, lambda_smooth, lambda_orient, lambda_sparse
            )
        )(model)
        updates, opt_state_new = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model_new = eqx.apply_updates(model, updates)
        return model_new, opt_state_new, loss

    losses = []
    for _ in range(n_steps):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))

    return model, losses
