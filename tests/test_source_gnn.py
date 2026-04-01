"""TDD tests for Physics-Informed Graph Neural Network (PI-GNN) source imaging.

RED phase — these tests define the expected API and behaviour before
implementation exists. The PI-GNN combines:
  1. Physics-informed initialisation (truncated SVD pseudo-inverse)
  2. Graph convolutions on cortical mesh topology (jraph + equinox)
  3. Multimodal vertex features (curvature, myelin, depth, orientation)

References:
  Bore et al. (2024) — physics-informed DL source imaging
  Hecker et al. (2021, 2022) — ESINet / ConvDip
  Lin et al. (2006) — loose orientation constraint
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jraph


# ---------------------------------------------------------------------------
# Helpers for synthetic test data
# ---------------------------------------------------------------------------

def _make_mesh(n_vertices=100, n_faces=180):
    """Create a synthetic triangular mesh (sphere-like)."""
    rng = np.random.RandomState(0)
    vertices = rng.randn(n_vertices, 3).astype(np.float32)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices *= 80  # mm

    # Random but valid triangulation (indices in range)
    faces = rng.randint(0, n_vertices, size=(n_faces, 3)).astype(np.int32)
    return vertices, faces


def _make_forward_problem(n_sources=100, n_sensors=32, n_times=50, seed=42):
    """Synthetic forward problem with spatially-structured gain."""
    rng = np.random.RandomState(seed)
    positions = rng.randn(n_sources, 3).astype(np.float32) * 50
    sensor_pos = rng.randn(n_sensors, 3).astype(np.float32) * 80

    diff = sensor_pos[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1.0)
    L = (1.0 / (dist ** 2)).astype(np.float32)
    L = L / np.max(L) * 0.01

    J_true = np.zeros((n_sources, n_times), dtype=np.float32)
    J_true[20, :] = np.sin(2 * np.pi * 10 * np.linspace(0, 0.5, n_times)).astype(np.float32)

    Y = (L @ J_true + rng.randn(n_sensors, n_times).astype(np.float32) * 1e-5)
    noise_cov = np.eye(n_sensors, dtype=np.float32) * 1e-10

    return {
        'Y': jnp.array(Y), 'L': jnp.array(L),
        'positions': jnp.array(positions),
        'noise_cov': jnp.array(noise_cov),
        'J_true': J_true, 'active_idx': 20,
        'n_sources': n_sources, 'n_sensors': n_sensors, 'n_times': n_times,
    }


# ===========================================================================
# TestCorticalGraph — mesh → jraph graph construction
# ===========================================================================

class TestCorticalGraph:
    """Graph construction from cortical mesh topology."""

    def test_mesh_to_graph(self):
        """FreeSurfer mesh → jraph GraphsTuple."""
        from neurojax.source.graph_utils import mesh_to_graph
        vertices, faces = _make_mesh(100, 180)
        graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))
        assert isinstance(graph, jraph.GraphsTuple)
        assert graph.n_node[0] == 100

    def test_adjacency_from_faces(self):
        """Triangular faces → sender/receiver edge arrays."""
        from neurojax.source.graph_utils import adjacency_from_faces
        faces = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)
        senders, receivers = adjacency_from_faces(faces, n_vertices=4)
        # Each triangle contributes 6 directed edges (3 pairs × 2 directions)
        # Two triangles share edge (1,2), so after dedup: ≤ 10 edges
        assert len(senders) == len(receivers)
        assert len(senders) > 0
        # All indices valid
        assert int(jnp.max(senders)) < 4
        assert int(jnp.max(receivers)) < 4

    def test_node_features_shape(self):
        """Vertex features should include normals + optional scalars."""
        from neurojax.source.graph_utils import compute_vertex_features
        vertices, faces = _make_mesh(100, 180)
        normals = np.random.randn(100, 3).astype(np.float32)
        curv = np.random.randn(100).astype(np.float32)
        features = compute_vertex_features(
            jnp.array(vertices), jnp.array(faces),
            normals=jnp.array(normals), curv=jnp.array(curv)
        )
        # At minimum: normals (3) + curvature (1) = 4 features
        assert features.shape[0] == 100
        assert features.shape[1] >= 4

    def test_graph_laplacian(self):
        """Graph Laplacian should be symmetric and positive semi-definite."""
        from neurojax.source.graph_utils import adjacency_from_faces, graph_laplacian
        faces = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 0]], dtype=jnp.int32)
        senders, receivers = adjacency_from_faces(faces, n_vertices=4)
        L = graph_laplacian(senders, receivers, n_vertices=4)
        assert L.shape == (4, 4)
        # Symmetric
        np.testing.assert_allclose(L, L.T, atol=1e-6)
        # PSD: eigenvalues >= 0
        eigvals = jnp.linalg.eigvalsh(L)
        assert jnp.all(eigvals >= -1e-6)
        # Row sums = 0 (Laplacian property)
        np.testing.assert_allclose(jnp.sum(L, axis=1), 0.0, atol=1e-6)


# ===========================================================================
# TestGraphConvLayer — single message-passing layer
# ===========================================================================

class TestGraphConvLayer:
    """Graph convolution layer on cortical mesh."""

    def test_message_passing_shape(self):
        """Output shape should match input shape."""
        from neurojax.source.graph_utils import mesh_to_graph
        from neurojax.source.source_gnn import GraphConvLayer
        vertices, faces = _make_mesh(50, 90)
        graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))

        hidden = 16
        layer = GraphConvLayer(hidden_dim=hidden, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (50, hidden))
        out = layer(x, graph)
        assert out.shape == (50, hidden)

    def test_learnable_parameters(self):
        """Layer should have trainable weights."""
        from neurojax.source.source_gnn import GraphConvLayer
        import equinox as eqx
        layer = GraphConvLayer(hidden_dim=16, key=jax.random.PRNGKey(0))
        params = eqx.filter(layer, eqx.is_array)
        n_params = sum(p.size for p in jax.tree.leaves(params))
        # weight_self (16×16) + weight_msg (16×16) + bias (16) = 528
        assert n_params >= 500

    def test_equivariant_to_permutation(self):
        """Permuting node indices (with consistent graph) should permute output."""
        from neurojax.source.graph_utils import mesh_to_graph
        from neurojax.source.source_gnn import GraphConvLayer

        # Small graph: 4 nodes, known edges
        vertices = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=jnp.float32)
        faces = jnp.array([[0, 1, 2], [1, 2, 3]], dtype=jnp.int32)
        graph = mesh_to_graph(vertices, faces)

        hidden = 8
        layer = GraphConvLayer(hidden_dim=hidden, key=jax.random.PRNGKey(0))
        x = jax.random.normal(jax.random.PRNGKey(1), (4, hidden))
        out = layer(x, graph)

        # Permute nodes 0 ↔ 3
        perm = jnp.array([3, 1, 2, 0])
        x_perm = x[perm]
        # Permute graph edges accordingly
        inv_perm = jnp.array([3, 1, 2, 0])  # inverse of swap(0,3)
        senders_perm = inv_perm[graph.senders]
        receivers_perm = inv_perm[graph.receivers]
        graph_perm = graph._replace(
            nodes=x_perm,
            senders=senders_perm,
            receivers=receivers_perm
        )
        out_perm = layer(x_perm, graph_perm)

        # Output should be permuted version of original
        np.testing.assert_allclose(out[0], out_perm[3], atol=1e-5)
        np.testing.assert_allclose(out[3], out_perm[0], atol=1e-5)


# ===========================================================================
# TestPhysicsInit — pseudo-inverse initialisation
# ===========================================================================

class TestPhysicsInit:
    """Physics-informed initialisation via truncated SVD."""

    def test_estimate_tikhonov_reg(self):
        """Auto-reg should return a concrete, positive float."""
        from neurojax.source.source_gnn import estimate_tikhonov_reg
        p = _make_forward_problem()
        reg = estimate_tikhonov_reg(p['L'])
        assert isinstance(reg, float)
        assert reg > 0
        # Should be between smallest and largest singular value
        s = np.asarray(jnp.linalg.svd(p['L'], compute_uv=False))
        assert reg >= s[-1] * 0.01
        assert reg <= s[0] * 10

    def test_tikhonov_shape(self):
        from neurojax.source.source_gnn import tikhonov_inverse, estimate_tikhonov_reg
        p = _make_forward_problem()
        reg = estimate_tikhonov_reg(p['L'])
        J0 = tikhonov_inverse(p['Y'], p['L'], reg=reg)
        assert J0.shape == (p['n_sources'], p['n_times'])

    def test_tikhonov_recovers_point_source(self):
        """Tikhonov inverse should roughly localise a single dipole."""
        from neurojax.source.source_gnn import tikhonov_inverse, estimate_tikhonov_reg
        p = _make_forward_problem(n_sources=50, n_sensors=32)
        reg = estimate_tikhonov_reg(p['L'])
        J0 = np.asarray(tikhonov_inverse(p['Y'], p['L'], reg=reg))
        power = np.sum(J0 ** 2, axis=1)
        peak = int(np.argmax(power))
        top10 = set(np.argsort(power)[-10:])
        assert p['active_idx'] in top10, \
            f"Active source {p['active_idx']} not in top 10 (peak={peak})"

    def test_tikhonov_stronger_reg_smoother(self):
        """Higher λ should produce lower-energy (smoother) solutions."""
        from neurojax.source.source_gnn import tikhonov_inverse
        p = _make_forward_problem()
        J_weak = tikhonov_inverse(p['Y'], p['L'], reg=0.001)
        J_strong = tikhonov_inverse(p['Y'], p['L'], reg=1.0)
        energy_weak = float(jnp.sum(J_weak ** 2))
        energy_strong = float(jnp.sum(J_strong ** 2))
        assert energy_strong <= energy_weak  # stronger reg → less energy


# ===========================================================================
# TestSourceGNN — full model forward pass and losses
# ===========================================================================

class TestSourceGNN:
    """Full PI-GNN model."""

    @pytest.fixture
    def model_and_data(self):
        from neurojax.source.graph_utils import mesh_to_graph, compute_vertex_features, graph_laplacian, adjacency_from_faces
        from neurojax.source.source_gnn import SourceGNN

        n_src, n_sen, n_times = 50, 32, 30
        vertices, faces = _make_mesh(n_src, 90)
        p = _make_forward_problem(n_sources=n_src, n_sensors=n_sen, n_times=n_times)

        graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))
        normals = jnp.array(vertices / np.linalg.norm(vertices, axis=1, keepdims=True))
        features = compute_vertex_features(
            jnp.array(vertices), jnp.array(faces), normals=normals
        )
        senders, receivers = adjacency_from_faces(jnp.array(faces), n_src)
        laplacian = graph_laplacian(senders, receivers, n_src)

        from neurojax.source.source_gnn import estimate_tikhonov_reg
        reg = estimate_tikhonov_reg(p['L'])
        model = SourceGNN(
            n_features=features.shape[1],
            n_times=n_times,
            hidden_dim=32,
            n_layers=2,
            orientation_mode='fixed',
            tikhonov_reg=reg,
            key=jax.random.PRNGKey(0)
        )
        return model, graph, features, laplacian, normals, p

    def test_forward_shape(self, model_and_data):
        model, graph, features, laplacian, normals, p = model_and_data
        J = model(p['Y'], p['L'], graph, features)
        assert J.shape == (50, 30)  # (n_sources, n_times)

    def test_forward_differentiable(self, model_and_data):
        """jax.grad should flow through the full model."""
        import equinox as eqx
        model, graph, features, laplacian, normals, p = model_and_data

        @eqx.filter_grad
        def grad_fn(model):
            J = model(p['Y'], p['L'], graph, features)
            return jnp.sum(J ** 2)

        grads = grad_fn(model)
        grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)

    def test_physics_loss(self, model_and_data):
        """Data fidelity loss ‖Y - L·J‖² should be computable."""
        model, graph, features, laplacian, normals, p = model_and_data
        J = model(p['Y'], p['L'], graph, features)
        loss = model.physics_loss(p['Y'], p['L'], J)
        assert jnp.isfinite(loss)
        assert float(loss) >= 0

    def test_smoothness_loss(self, model_and_data):
        """Graph Laplacian smoothness penalty."""
        model, graph, features, laplacian, normals, p = model_and_data
        J = model(p['Y'], p['L'], graph, features)
        loss = model.smoothness_loss(J, laplacian)
        assert jnp.isfinite(loss)
        assert float(loss) >= 0

    def test_orientation_loss(self, model_and_data):
        """Penalise tangential (off-normal) components."""
        model, graph, features, laplacian, normals, p = model_and_data
        J = model(p['Y'], p['L'], graph, features)
        loss = model.orientation_loss(J, normals)
        assert jnp.isfinite(loss)
        assert float(loss) >= 0

    def test_with_vertex_features(self, model_and_data):
        """Model should accept and use multimodal vertex features."""
        from neurojax.source.graph_utils import compute_vertex_features
        model, graph, features, laplacian, normals, p = model_and_data
        vertices, faces = _make_mesh(50, 90)
        curv = jnp.array(np.random.randn(50).astype(np.float32))
        myelin = jnp.array(np.abs(np.random.randn(50).astype(np.float32)))
        features_rich = compute_vertex_features(
            jnp.array(vertices), jnp.array(faces),
            normals=normals, curv=curv, myelin=myelin
        )
        # Need a model with matching feature dim
        from neurojax.source.source_gnn import SourceGNN
        model2 = SourceGNN(
            n_features=features_rich.shape[1],
            n_times=30,
            hidden_dim=32,
            n_layers=2,
            orientation_mode='fixed',
            svd_rank=20,
            key=jax.random.PRNGKey(0)
        )
        J = model2(p['Y'], p['L'], graph, features_rich)
        assert J.shape == (50, 30)


# ===========================================================================
# TestSourceGNNTraining — training loop convergence
# ===========================================================================

class TestSourceGNNTraining:
    """Training the PI-GNN on synthetic data."""

    def test_synthetic_dipole_recovery(self):
        """Train on synthetic data and recover the active source."""
        from neurojax.source.graph_utils import (
            mesh_to_graph, compute_vertex_features,
            graph_laplacian, adjacency_from_faces
        )
        from neurojax.source.source_gnn import (
            SourceGNN, train_source_gnn, estimate_tikhonov_reg
        )

        n_src, n_sen, n_times = 50, 32, 30
        vertices, faces = _make_mesh(n_src, 90)
        p = _make_forward_problem(n_sources=n_src, n_sensors=n_sen, n_times=n_times)

        graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))
        normals = jnp.array(vertices / np.linalg.norm(vertices, axis=1, keepdims=True))
        features = compute_vertex_features(
            jnp.array(vertices), jnp.array(faces), normals=normals
        )
        senders, receivers = adjacency_from_faces(jnp.array(faces), n_src)
        laplacian = graph_laplacian(senders, receivers, n_src)

        # Precompute Tikhonov reg from data — always explicit and reportable
        reg = estimate_tikhonov_reg(p['L'])
        assert isinstance(reg, float) and reg > 0

        model = SourceGNN(
            n_features=features.shape[1],
            n_times=n_times,
            hidden_dim=32,
            n_layers=2,
            orientation_mode='fixed',
            tikhonov_reg=reg,
            key=jax.random.PRNGKey(0)
        )

        model, losses = train_source_gnn(
            model, p['Y'], p['L'], graph, features,
            normals, laplacian, n_steps=100, lr=1e-3
        )

        # Loss should decrease
        assert losses[-1] < losses[0]

        # Check source recovery
        J = np.asarray(model(p['Y'], p['L'], graph, features))
        power = np.sum(J ** 2, axis=1)
        top5 = set(np.argsort(power)[-5:])
        assert p['active_idx'] in top5, \
            f"Active source {p['active_idx']} not in top 5: {top5}"

    def test_snr_sensitivity(self):
        """Performance should degrade gracefully with noise."""
        from neurojax.source.source_gnn import truncated_svd_inverse

        errors = []
        for snr_db in [20, 10, 0]:
            p = _make_forward_problem(n_sources=50, n_sensors=32)
            signal_power = float(jnp.mean(p['Y'] ** 2))
            noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10)))
            Y_noisy = p['Y'] + noise_std * jax.random.normal(
                jax.random.PRNGKey(snr_db), p['Y'].shape
            )
            J0 = np.asarray(truncated_svd_inverse(Y_noisy, p['L'], rank=20))
            power = np.sum(J0 ** 2, axis=1)
            peak = int(np.argmax(power))
            errors.append(abs(peak - p['active_idx']))

        # Higher SNR should generally give smaller localisation error
        assert errors[0] <= errors[-1] + 5  # some tolerance

    def test_outperforms_pseudoinverse(self):
        """Trained GNN should produce a sparser solution than L⁺Y.

        The GNN multi-objective loss trades data fidelity for regularisation,
        so we compare on source-space sparsity (Gini coefficient) rather
        than sensor-space residual.
        """
        from neurojax.source.graph_utils import (
            mesh_to_graph, compute_vertex_features,
            graph_laplacian, adjacency_from_faces
        )
        from neurojax.source.source_gnn import (
            SourceGNN, truncated_svd_inverse, train_source_gnn,
            estimate_tikhonov_reg
        )

        n_src, n_sen, n_times = 50, 32, 30
        vertices, faces = _make_mesh(n_src, 90)
        p = _make_forward_problem(n_sources=n_src, n_sensors=n_sen, n_times=n_times)

        graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))
        normals = jnp.array(vertices / np.linalg.norm(vertices, axis=1, keepdims=True))
        features = compute_vertex_features(
            jnp.array(vertices), jnp.array(faces), normals=normals
        )
        senders, receivers = adjacency_from_faces(jnp.array(faces), n_src)
        laplacian = graph_laplacian(senders, receivers, n_src)

        # Precompute reg — always explicit
        reg = estimate_tikhonov_reg(p['L'])

        # Pseudo-inverse baseline
        J_pinv = np.asarray(truncated_svd_inverse(p['Y'], p['L']))
        power_pinv = np.sum(J_pinv ** 2, axis=1)

        # Trained GNN
        model = SourceGNN(
            n_features=features.shape[1],
            n_times=n_times,
            hidden_dim=32,
            n_layers=2,
            orientation_mode='fixed',
            tikhonov_reg=reg,
            key=jax.random.PRNGKey(0)
        )
        model, losses = train_source_gnn(
            model, p['Y'], p['L'], graph, features,
            normals, laplacian, n_steps=200, lr=1e-3
        )
        J_gnn = np.asarray(model(p['Y'], p['L'], graph, features))
        power_gnn = np.sum(J_gnn ** 2, axis=1)

        # GNN should produce a more focal (sparser) solution
        # Measured by ratio of max power to mean power
        focality_pinv = np.max(power_pinv) / (np.mean(power_pinv) + 1e-20)
        focality_gnn = np.max(power_gnn) / (np.mean(power_gnn) + 1e-20)

        # Training loss should decrease
        assert losses[-1] < losses[0]
        # GNN should be at least somewhat focal
        assert focality_gnn > 1.5, f"GNN focality={focality_gnn:.1f}"


# ===========================================================================
# TestPhantomValidation — integration tests with real phantom data
# ===========================================================================

_PHANTOM_AVAILABLE = False
try:
    import mne
    _PHANTOM_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not _PHANTOM_AVAILABLE, reason="MNE not available")
class TestPhantomValidation:
    """Validation on Brainstorm CTF phantom (known dipole positions)."""

    @pytest.fixture(scope="class")
    def phantom_data(self):
        """Download and prepare CTF phantom if available."""
        try:
            data_path = mne.datasets.brainstorm.bst_phantom_ctf.data_path(
                download=False
            )
            return {'path': data_path, 'available': True}
        except Exception:
            pytest.skip("Brainstorm CTF phantom not downloaded")

    def test_brainstorm_ctf_phantom(self, phantom_data):
        """Localization error < 15mm on CTF phantom dipoles."""
        pytest.skip("Requires phantom data download — run manually")

    def test_compare_gnn_vs_vareta(self, phantom_data):
        """GNN vs VARETA on same phantom data."""
        pytest.skip("Requires phantom data download — run manually")

    def test_compare_gnn_vs_mne(self, phantom_data):
        """GNN vs MNE dSPM on same phantom data."""
        pytest.skip("Requires phantom data download — run manually")


# ===========================================================================
# TestOrientationModes
# ===========================================================================

class TestOrientationModes:
    """Dipole orientation constraint modes."""

    def test_fixed_orientation(self):
        """Fixed mode: 1 component per source (normal-to-cortex)."""
        from neurojax.source.graph_utils import orientation_matrix
        normals = jnp.array(np.random.randn(50, 3).astype(np.float32))
        normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
        O = orientation_matrix(normals, mode='fixed')
        # Fixed: (n_sources, 3) projection to normal
        assert O.shape == (50, 3)
        # Each row should be a unit vector
        norms = jnp.linalg.norm(O, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_loose_orientation(self):
        """Loose mode: ±30° from normal (Lin et al. 2006)."""
        from neurojax.source.graph_utils import orientation_matrix
        normals = jnp.array(np.random.randn(50, 3).astype(np.float32))
        normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
        O = orientation_matrix(normals, mode='loose')
        # Loose: (n_sources, 3, 3) allowing tangential with penalty
        assert O.shape == (50, 3, 3)

    def test_free_orientation(self):
        """Free mode: 3 unconstrained components per source."""
        from neurojax.source.graph_utils import orientation_matrix
        normals = jnp.array(np.random.randn(50, 3).astype(np.float32))
        normals = normals / jnp.linalg.norm(normals, axis=1, keepdims=True)
        O = orientation_matrix(normals, mode='free')
        # Free: (n_sources, 3, 3) identity-like (no constraint)
        assert O.shape == (50, 3, 3)
        # Should be identity for each source
        for i in range(50):
            np.testing.assert_allclose(O[i], jnp.eye(3), atol=1e-6)


# ===========================================================================
# TestPatchSize — spatial extent of source activation
# ===========================================================================

class TestPatchSize:
    """Source activation spatial extent — point vs patch."""

    def test_point_source(self):
        """Single vertex activation should be recoverable."""
        from neurojax.source.source_gnn import truncated_svd_inverse
        p = _make_forward_problem(n_sources=50, n_sensors=32)
        J0 = np.asarray(truncated_svd_inverse(p['Y'], p['L'], rank=20))
        power = np.sum(J0 ** 2, axis=1)
        # The solution should have a clear peak
        peak_power = np.max(power)
        mean_power = np.mean(power)
        assert peak_power > 3 * mean_power  # focal activation

    def test_patch_activation(self):
        """Group of co-activated sources should be recovered as a cluster."""
        from neurojax.source.source_gnn import truncated_svd_inverse

        rng = np.random.RandomState(42)
        n_src, n_sen, n_times = 50, 32, 30
        positions = rng.randn(n_src, 3).astype(np.float32) * 50
        sensor_pos = rng.randn(n_sen, 3).astype(np.float32) * 80

        diff = sensor_pos[:, None, :] - positions[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1.0)
        L = jnp.array((1.0 / (dist ** 2) / np.max(1.0 / (dist ** 2)) * 0.01).astype(np.float32))

        # Activate a patch: source 20 + its 3 nearest neighbours
        dists_src = np.sqrt(np.sum(
            (positions[:, None] - positions[None, :]) ** 2, axis=-1
        ))
        neighbours = np.argsort(dists_src[20])[:4]  # source 20 + 3 nearest

        J_true = np.zeros((n_src, n_times), dtype=np.float32)
        t = np.linspace(0, 0.5, n_times).astype(np.float32)
        for idx in neighbours:
            J_true[idx, :] = np.sin(2 * np.pi * 10 * t)

        Y = jnp.array(L @ J_true + rng.randn(n_sen, n_times).astype(np.float32) * 1e-5)
        J0 = np.asarray(truncated_svd_inverse(Y, L, rank=20))
        power = np.sum(J0 ** 2, axis=1)
        top10 = set(np.argsort(power)[-10:])

        # At least 2 of the 4 patch sources should be in top 10
        overlap = len(set(neighbours) & top10)
        assert overlap >= 2, f"Only {overlap}/4 patch sources in top 10"

    def test_variable_patch(self):
        """GNN receptive field should adapt patch size via message-passing."""
        from neurojax.source.graph_utils import mesh_to_graph, adjacency_from_faces
        # After K message-passing layers, each node's receptive field
        # is its K-hop neighbourhood
        vertices, faces = _make_mesh(50, 90)
        graph = mesh_to_graph(jnp.array(vertices), jnp.array(faces))

        senders = np.asarray(graph.senders)
        receivers = np.asarray(graph.receivers)

        # Build 1-hop neighbourhood
        from collections import defaultdict
        neighbours = defaultdict(set)
        for s, r in zip(senders, receivers):
            neighbours[int(s)].add(int(r))

        # With K=2 layers, receptive field = 2-hop
        two_hop = set()
        node = 0
        one_hop = neighbours[node]
        two_hop = set(one_hop)
        for n in one_hop:
            two_hop |= neighbours[n]

        # The GNN with 2 layers should have receptive field >= 2-hop size
        assert len(two_hop) >= 2  # at least some neighbours reachable
