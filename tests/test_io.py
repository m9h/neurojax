"""Tests for neurojax.io.bridge and neurojax.io.loader modules."""

import pytest
import numpy as np
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

class TestIOImportability:
    def test_bridge_importable(self):
        from neurojax.io.bridge import mne_to_jax, jax_to_mne
        assert callable(mne_to_jax)
        assert callable(jax_to_mne)

    def test_loader_importable(self):
        from neurojax.io.loader import load_data
        assert callable(load_data)


# ---------------------------------------------------------------------------
# Bridge tests -- using real MNE objects (lightweight synthetic)
# ---------------------------------------------------------------------------

mne = pytest.importorskip("mne")


class TestMneToJax:
    """Test mne_to_jax conversion."""

    @pytest.fixture()
    def raw_fixture(self):
        """Create a minimal MNE RawArray for testing."""
        n_channels, n_times = 3, 500
        sfreq = 256.0
        data = np.random.default_rng(42).standard_normal((n_channels, n_times)).astype(np.float64)
        info = mne.create_info(
            ch_names=[f"EEG{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types="eeg",
        )
        return mne.io.RawArray(data, info)

    def test_returns_tuple(self, raw_fixture):
        from neurojax.io.bridge import mne_to_jax

        result = mne_to_jax(raw_fixture)
        assert isinstance(result, tuple) and len(result) == 2

    def test_jax_array_type(self, raw_fixture):
        from neurojax.io.bridge import mne_to_jax

        arr, _ = mne_to_jax(raw_fixture)
        assert isinstance(arr, jnp.ndarray)

    def test_sfreq_preserved(self, raw_fixture):
        from neurojax.io.bridge import mne_to_jax

        _, sfreq = mne_to_jax(raw_fixture)
        assert sfreq == 256.0

    def test_shape_preserved(self, raw_fixture):
        from neurojax.io.bridge import mne_to_jax

        arr, _ = mne_to_jax(raw_fixture)
        assert arr.shape == (3, 500)

    def test_output_float32(self, raw_fixture):
        from neurojax.io.bridge import mne_to_jax

        arr, _ = mne_to_jax(raw_fixture)
        assert arr.dtype == jnp.float32

    def test_values_close(self, raw_fixture):
        from neurojax.io.bridge import mne_to_jax

        original_data = raw_fixture.get_data().astype(np.float32)
        arr, _ = mne_to_jax(raw_fixture)
        np.testing.assert_allclose(np.array(arr), original_data, rtol=1e-6)


class TestJaxToMne:
    """Test jax_to_mne conversion."""

    @pytest.fixture()
    def raw_template(self):
        n_ch, n_t = 4, 300
        data = np.zeros((n_ch, n_t), dtype=np.float64)
        info = mne.create_info(
            ch_names=[f"CH{i}" for i in range(n_ch)],
            sfreq=128.0,
            ch_types="eeg",
        )
        return mne.io.RawArray(data, info)

    def test_returns_raw_instance(self, raw_template):
        from neurojax.io.bridge import jax_to_mne

        new_data = jnp.ones((4, 300), dtype=jnp.float32)
        result = jax_to_mne(new_data, raw_template)
        assert isinstance(result, mne.io.BaseRaw)

    def test_data_overwritten(self, raw_template):
        from neurojax.io.bridge import jax_to_mne

        new_data = jnp.ones((4, 300), dtype=jnp.float32) * 42.0
        result = jax_to_mne(new_data, raw_template)
        np.testing.assert_allclose(result.get_data(), 42.0)

    def test_template_not_mutated(self, raw_template):
        from neurojax.io.bridge import jax_to_mne

        new_data = jnp.ones((4, 300), dtype=jnp.float32) * 99.0
        _ = jax_to_mne(new_data, raw_template)
        # Template should still be zeros because jax_to_mne calls .copy()
        np.testing.assert_allclose(raw_template.get_data(), 0.0)


class TestRoundtrip:
    """mne -> jax -> mne roundtrip preserves data."""

    def test_roundtrip(self):
        from neurojax.io.bridge import mne_to_jax, jax_to_mne

        n_ch, n_t = 5, 1000
        sfreq = 512.0
        rng = np.random.default_rng(7)
        original = rng.standard_normal((n_ch, n_t)).astype(np.float64)
        info = mne.create_info(
            ch_names=[f"E{i}" for i in range(n_ch)],
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(original, info)

        jax_data, sf = mne_to_jax(raw)
        assert sf == sfreq

        raw2 = jax_to_mne(jax_data, raw)
        # float32 roundtrip may lose precision
        np.testing.assert_allclose(
            raw2.get_data(), original, atol=1e-6, rtol=1e-5,
        )


class TestLoaderErrors:
    """Loader should raise when given a non-existent path."""

    def test_load_nonexistent_file(self):
        from neurojax.io.loader import load_data

        with pytest.raises(Exception):
            load_data("/nonexistent/path/file.fif")
