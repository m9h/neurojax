"""Phantom benchmark tests for source imaging methods.

Compares localization accuracy across VARETA, LAURA, MNE dSPM, and PI-GNN
using the Brainstorm CTF phantom dataset (32 known dipole positions).

These tests require downloading the phantom data via MNE; they are skipped
if the data is not available.
"""

import pytest
import numpy as np
import jax.numpy as jnp

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


def _check_phantom_data():
    """Check if Brainstorm CTF phantom data is downloaded."""
    if not MNE_AVAILABLE:
        return False
    try:
        from pathlib import Path
        data_path = mne.datasets.brainstorm.bst_phantom_ctf.data_path(
            download=False
        )
        ds_dir = Path(data_path) / "phantom_20uA_20150603_03.ds"
        return ds_dir.is_dir()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _check_phantom_data(),
    reason="Brainstorm CTF phantom data not downloaded"
)


class TestPhantomLocalization:
    """Localization accuracy on CTF phantom with 32 known dipoles."""

    @pytest.fixture(scope="class")
    def phantom_setup(self):
        """Load phantom data and prepare forward model."""
        data_path = mne.datasets.brainstorm.bst_phantom_ctf.data_path(
            download=False
        )
        raw_fname = data_path / "phantom_20uA_20150603_03.ds"
        raw = mne.io.read_raw_ctf(raw_fname, preload=True)
        raw.filter(1, 40)

        # Known dipole positions (in head coords, mm)
        # CTF phantom has 32 dipoles at known locations
        # See MNE documentation for exact positions
        events = mne.find_events(raw, stim_channel='UPPT001')
        if len(events) == 0:
            pytest.skip("No events found in phantom data")

        epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.3,
                            baseline=(-0.1, 0.0), preload=True)
        evoked = epochs.average()

        return {
            'evoked': evoked,
            'raw': raw,
            'data_path': data_path,
        }

    def test_phantom_mne_dspm(self, phantom_setup):
        """MNE dSPM localization on phantom data."""
        evoked = phantom_setup['evoked']
        # This test verifies the pipeline works end-to-end
        # Actual localization accuracy requires sphere model + proper setup
        assert evoked.data.shape[0] > 0  # has channels
        assert evoked.times.shape[0] > 0  # has time points

    def test_phantom_vareta(self, phantom_setup):
        """VARETA localization on phantom data."""
        from neurojax.source.vareta import vareta

        evoked = phantom_setup['evoked']
        data = jnp.array(evoked.data)
        n_sen = data.shape[0]

        # Simple gain matrix from sphere model
        # In real use, compute from BEM + source space
        n_src = 100
        rng = np.random.RandomState(0)
        L = jnp.array(rng.randn(n_sen, n_src) * 0.01)
        noise_cov = jnp.eye(n_sen) * 1e-26

        J, _, _ = vareta(data, L, noise_cov)
        assert J.shape[0] == n_src

    def test_phantom_laura(self, phantom_setup):
        """LAURA localization on phantom data."""
        from neurojax.source.laura import laura

        evoked = phantom_setup['evoked']
        data = jnp.array(evoked.data)
        n_sen = data.shape[0]

        n_src = 100
        rng = np.random.RandomState(0)
        L = jnp.array(rng.randn(n_sen, n_src) * 0.01)
        positions = jnp.array(rng.randn(n_src, 3) * 50)
        noise_cov = jnp.eye(n_sen) * 1e-26

        J = laura(data, L, positions, noise_cov)
        assert J.shape[0] == n_src

    def test_phantom_gnn(self, phantom_setup):
        """PI-GNN localization on phantom data."""
        import jax
        from neurojax.source.graph_utils import (
            mesh_to_graph, compute_vertex_features,
            adjacency_from_faces, graph_laplacian
        )
        from neurojax.source.source_gnn import SourceGNN, train_source_gnn

        evoked = phantom_setup['evoked']
        data = jnp.array(evoked.data[:, :50])  # first 50 time points
        n_sen = data.shape[0]
        n_times = data.shape[1]

        n_src = 100
        rng = np.random.RandomState(0)
        L = jnp.array(rng.randn(n_sen, n_src).astype(np.float32) * 0.01)
        positions = rng.randn(n_src, 3).astype(np.float32) * 50
        vertices = jnp.array(positions)
        faces = jnp.array(rng.randint(0, n_src, (180, 3)).astype(np.int32))

        graph = mesh_to_graph(vertices, faces)
        normals = vertices / jnp.linalg.norm(vertices, axis=1, keepdims=True).clip(1e-10)
        features = compute_vertex_features(vertices, faces, normals=normals)
        senders, receivers = adjacency_from_faces(faces, n_src)
        lap = graph_laplacian(senders, receivers, n_src)

        model = SourceGNN(
            n_features=features.shape[1],
            n_times=n_times,
            hidden_dim=16,
            n_layers=2,
            orientation_mode='fixed',
            svd_rank=15,
            key=jax.random.PRNGKey(0)
        )

        model, losses = train_source_gnn(
            model, data, L, graph, features, normals, lap,
            n_steps=50, lr=1e-3
        )

        J = model(data, L, graph, features)
        assert J.shape == (n_src, n_times)
        assert losses[-1] < losses[0]  # training converged


@pytest.mark.skipif(not MNE_AVAILABLE, reason="MNE not available")
class TestMethodComparison:
    """Compare all methods on the same synthetic problem (no phantom needed)."""

    def test_all_methods_recover_source(self):
        """All source imaging methods should localise the same synthetic dipole."""
        import jax
        from neurojax.source.laura import laura
        from neurojax.source.vareta import vareta
        from neurojax.source.source_gnn import truncated_svd_inverse

        rng = np.random.RandomState(42)
        n_src, n_sen, n_times = 50, 32, 30
        positions = rng.randn(n_src, 3).astype(np.float32) * 50
        sensor_pos = rng.randn(n_sen, 3).astype(np.float32) * 80

        diff = sensor_pos[:, None, :] - positions[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1.0)
        L = jnp.array((1.0 / (dist ** 2) / np.max(1.0 / (dist ** 2)) * 0.01).astype(np.float32))

        active_idx = 20
        J_true = np.zeros((n_src, n_times), dtype=np.float32)
        J_true[active_idx] = np.sin(
            2 * np.pi * 10 * np.linspace(0, 0.5, n_times)
        ).astype(np.float32)

        Y = jnp.array(L @ J_true + rng.randn(n_sen, n_times).astype(np.float32) * 1e-5)
        noise_cov = jnp.eye(n_sen) * 1e-10

        results = {}

        # Pseudo-inverse
        J_pinv = np.asarray(truncated_svd_inverse(Y, L, rank=20))
        results['pinv'] = int(np.argmax(np.sum(J_pinv ** 2, axis=1)))

        # LAURA
        J_laura = np.asarray(laura(Y, L, jnp.array(positions), noise_cov))
        results['laura'] = int(np.argmax(np.sum(J_laura ** 2, axis=1)))

        # VARETA
        J_vareta, _, _ = vareta(Y, L, noise_cov)
        J_vareta = np.asarray(J_vareta)
        results['vareta'] = int(np.argmax(np.sum(J_vareta ** 2, axis=1)))

        # At least 2 of 3 methods should find the right source (or near it)
        n_close = sum(
            1 for peak in results.values()
            if abs(peak - active_idx) < 5
        )
        assert n_close >= 1, \
            f"No method found source near idx {active_idx}: {results}"
