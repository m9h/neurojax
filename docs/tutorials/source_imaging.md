# Source Imaging Methods

This tutorial compares neurojax's source imaging methods on synthetic
data, from classical minimum-norm through biophysical priors to the
physics-informed graph neural network (PI-GNN).

## The inverse problem

MEG/EEG source imaging recovers neural current density **J** from sensor
measurements **Y = LJ + n**, where **L** is the leadfield (forward model)
and **n** is sensor noise. The problem is ill-posed because there are far
more sources than sensors.

## Setup: synthetic forward problem

```python
import numpy as np
import jax
import jax.numpy as jnp

# Spatially-structured gain matrix (mimicking physics)
rng = np.random.RandomState(42)
n_src, n_sen, n_times = 50, 32, 30
positions = rng.randn(n_src, 3).astype(np.float32) * 50  # source positions (mm)
sensor_pos = rng.randn(n_sen, 3).astype(np.float32) * 80

diff = sensor_pos[:, None, :] - positions[None, :, :]
dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1.0)
L = jnp.array((1.0 / (dist ** 2) / np.max(1.0 / (dist ** 2)) * 0.01).astype(np.float32))

# Single active source
J_true = np.zeros((n_src, n_times), dtype=np.float32)
J_true[20, :] = np.sin(2 * np.pi * 10 * np.linspace(0, 0.5, n_times)).astype(np.float32)
Y = jnp.array(L @ J_true + rng.randn(n_sen, n_times).astype(np.float32) * 1e-5)
noise_cov = jnp.eye(n_sen) * 1e-10
```

## Method 1: Tikhonov pseudo-inverse

The simplest approach — regularised least-squares with automatic
lambda selection from the singular value spectrum:

```python
from neurojax.source.source_gnn import tikhonov_inverse, estimate_tikhonov_reg

# Always estimate and report the regularisation parameter
reg = estimate_tikhonov_reg(L)
print(f"Tikhonov lambda = {reg:.4f}")  # transparent, reproducible

J_tik = tikhonov_inverse(Y, L, reg=reg)
power = jnp.sum(J_tik ** 2, axis=1)
peak = int(jnp.argmax(power))
print(f"Peak source: {peak} (true: 20)")
```

## Method 2: LAURA (biophysical 1/d^3 prior)

LAURA encodes dipole field physics — nearby sources covary with
strength proportional to 1/distance^3:

```python
from neurojax.source.laura import laura

J_laura = laura(Y, L, jnp.array(positions), noise_cov)
power = jnp.sum(J_laura ** 2, axis=1)
print(f"LAURA peak: {int(jnp.argmax(power))}")
```

## Method 3: VARETA (adaptive resolution)

VARETA uses data-driven variable resolution — more smoothing in
quiet areas, less near active sources:

```python
from neurojax.source.vareta import vareta

J_vareta, _, _ = vareta(Y, L, noise_cov)
power = jnp.sum(J_vareta ** 2, axis=1)
print(f"VARETA peak: {int(jnp.argmax(power))}")
```

## Method 4: PI-GNN (physics-informed graph neural network)

The PI-GNN combines physics-informed initialisation (Tikhonov) with
graph message-passing on cortical mesh topology. Multimodal vertex
features (normals, curvature, myelin) are concatenated with the
initial estimate and refined through learned graph convolutions.

```python
from neurojax.source.graph_utils import (
    mesh_to_graph, compute_vertex_features,
    adjacency_from_faces, graph_laplacian
)
from neurojax.source.source_gnn import SourceGNN, train_source_gnn

# Build mesh graph (in practice, from FreeSurfer cortical surface)
vertices = positions
faces = np.random.randint(0, n_src, (90, 3)).astype(np.int32)
graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))
normals = jnp.array(vertices / np.linalg.norm(vertices, axis=1, keepdims=True))
features = compute_vertex_features(jnp.array(vertices), jnp.array(faces),
                                    normals=normals)
senders, receivers = adjacency_from_faces(jnp.array(faces), n_src)
lap = graph_laplacian(senders, receivers, n_src)

# Explicit regularisation (always reported)
reg = estimate_tikhonov_reg(L)

model = SourceGNN(
    n_features=features.shape[1],
    n_times=n_times,
    hidden_dim=32,
    n_layers=2,
    orientation_mode='fixed',
    tikhonov_reg=reg,
    key=jax.random.PRNGKey(0)
)

# Train: multi-objective loss (data fidelity + smoothness + sparsity)
model, losses = train_source_gnn(
    model, Y, L, graph, features, normals, lap, n_steps=100
)

J_gnn = model(Y, L, graph, features)
power = jnp.sum(J_gnn ** 2, axis=1)
print(f"PI-GNN peak: {int(jnp.argmax(power))}")
print(f"Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
```

## Comparing methods

The PI-GNN loss function balances four objectives:

- **Data fidelity**: ||Y - LJ||^2 (sensor-space residual)
- **Graph smoothness**: tr(J^T L_G J) (Laplacian penalty on cortical mesh)
- **Orientation constraint**: penalise tangential (off-normal) components
- **Sparsity**: ||J||_1 (focal activations)

Because the entire pipeline is differentiable in JAX, all
hyperparameters (loss weights, regularisation, network weights) can
be optimised via gradient descent.

## Orientation modes

Source orientation relative to the cortical surface:

```python
from neurojax.source.graph_utils import orientation_matrix

# Fixed: 1 scalar per source (normal-to-cortex only)
O_fixed = orientation_matrix(normals, mode='fixed')     # (n_src, 3)

# Loose: +/-30 degrees from normal (Lin et al. 2006)
O_loose = orientation_matrix(normals, mode='loose')     # (n_src, 3, 3)

# Free: 3 unconstrained components
O_free = orientation_matrix(normals, mode='free')       # (n_src, 3, 3)
```

## Next steps

- See `tests/test_source_gnn.py` for 26 comprehensive tests
- See `tests/test_phantom_benchmark.py` for validation on Brainstorm CTF phantom
- See the [head modeling tutorial](head_modeling.md) for forward model construction
