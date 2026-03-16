"""Tests for the neurojax.tokenizer subpackage."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from neurojax.tokenizer._encoder import RNNEncoder
from neurojax.tokenizer._quantizer import TemperatureQuantizer, VQQuantizer, FSQQuantizer
from neurojax.tokenizer._decoder import Conv1dDecoder
from neurojax.tokenizer._tokenizer import EphysTokenizer, TokenizerOutput
from neurojax.tokenizer._baselines import mu_law_tokenize, mu_law_detokenize, quantile_tokenize
from neurojax.tokenizer._metrics import pve, pve_per_channel, token_utilization
from neurojax.tokenizer._vocab import refactor_vocabulary
from neurojax.tokenizer._train import fit


# ── RNNEncoder ──────────────────────────────────────────────────────────────

class TestRNNEncoder:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        encoder = RNNEncoder(hidden_size=32, n_layers=1, key=key)
        x = jax.random.normal(key, (100,))
        out = encoder(x)
        assert out.shape == (100, 32)

    def test_multi_layer(self):
        key = jax.random.PRNGKey(1)
        encoder = RNNEncoder(hidden_size=16, n_layers=3, key=key)
        x = jax.random.normal(key, (50,))
        out = encoder(x)
        assert out.shape == (50, 16)

    def test_vmap_channels(self):
        key = jax.random.PRNGKey(2)
        encoder = RNNEncoder(hidden_size=32, n_layers=1, key=key)
        x = jax.random.normal(key, (3, 100))  # 3 channels, 100 timesteps
        out = jax.vmap(encoder)(x)
        assert out.shape == (3, 100, 32)

    def test_scan_matches_loop(self):
        key = jax.random.PRNGKey(3)
        encoder = RNNEncoder(hidden_size=16, n_layers=1, key=key)
        x = jax.random.normal(key, (20,))

        # Scan result
        scan_out = encoder(x)

        # Manual loop
        cell = encoder.cells[0]
        h = jnp.zeros(16)
        outputs = []
        for t in range(20):
            h = cell(x[t:t+1], h)
            outputs.append(h)
        loop_out = jnp.stack(outputs)

        assert jnp.allclose(scan_out, loop_out, atol=1e-5)


# ── TemperatureQuantizer ────────────────────────────────────────────────────

class TestTemperatureQuantizer:
    def test_soft_at_temp_1(self):
        key = jax.random.PRNGKey(10)
        q = TemperatureQuantizer(hidden_size=32, n_tokens=16, key=key)
        h = jax.random.normal(key, (32,))
        out = q(h, jnp.array(1.0))
        assert out.shape == (16,)
        assert jnp.allclose(jnp.sum(out), 1.0, atol=1e-5)
        assert jnp.all(out >= 0)

    def test_hard_at_temp_0(self):
        key = jax.random.PRNGKey(11)
        q = TemperatureQuantizer(hidden_size=32, n_tokens=16, key=key)
        h = jax.random.normal(key, (32,))
        out = q(h, jnp.array(0.0))
        assert out.shape == (16,)
        # Should be one-hot
        assert jnp.sum(out == 1.0) == 1
        assert jnp.sum(out == 0.0) == 15

    def test_ste_gradient(self):
        key = jax.random.PRNGKey(12)
        q = TemperatureQuantizer(hidden_size=32, n_tokens=16, key=key)
        h = jax.random.normal(key, (32,))

        # STE should allow gradients through quantizer params even at hard temp
        # Use a weighted sum so the gradient isn't trivially zero
        target = jax.random.normal(jax.random.PRNGKey(99), (16,))

        @eqx.filter_grad
        def grad_fn(q):
            out = q(h, jnp.array(0.0))
            return jnp.sum(out * target)

        grads = grad_fn(q)
        # Check that linear layer weights received nonzero gradients
        assert not jnp.allclose(grads.linear.weight, jnp.zeros_like(grads.linear.weight))


# ── VQQuantizer ─────────────────────────────────────────────────────────────

class TestVQQuantizer:
    def test_one_hot_output(self):
        key = jax.random.PRNGKey(20)
        q = VQQuantizer(hidden_size=32, n_tokens=16, key=key)
        h = jax.random.normal(key, (32,))
        out = q(h, jnp.array(1.0))
        assert out.shape == (16,)
        assert jnp.sum(out) == 1.0

    def test_nearest_neighbor(self):
        key = jax.random.PRNGKey(21)
        q = VQQuantizer(hidden_size=8, n_tokens=4, key=key)
        # Use a codebook entry as input — should map to itself
        h = q.codebook[2]
        out = q(h, jnp.array(0.0))
        assert jnp.argmax(out) == 2


# ── FSQQuantizer ────────────────────────────────────────────────────────────

class TestFSQQuantizer:
    def test_output_shape(self):
        key = jax.random.PRNGKey(25)
        q = FSQQuantizer(hidden_size=32, levels=(8, 6, 5), key=key)
        assert q.n_tokens == 240
        h = jax.random.normal(key, (32,))
        out = q(h, jnp.array(0.0))
        assert out.shape == (240,)
        assert jnp.sum(out) == 1.0


# ── Conv1dDecoder ───────────────────────────────────────────────────────────

class TestConv1dDecoder:
    def test_length_preservation(self):
        key = jax.random.PRNGKey(30)
        dec = Conv1dDecoder(n_tokens=16, token_dim=10, key=key)
        tw = jax.random.normal(key, (16, 50))
        out = dec(tw)
        assert out.shape == (50,)

    def test_different_token_dims(self):
        key = jax.random.PRNGKey(31)
        for td in [3, 7, 15]:
            dec = Conv1dDecoder(n_tokens=8, token_dim=td, key=key)
            tw = jax.random.normal(key, (8, 100))
            out = dec(tw)
            assert out.shape == (100,)


# ── EphysTokenizer (composed) ──────────────────────────────────────────────

class TestEphysTokenizer:
    def test_forward_shape(self):
        key = jax.random.PRNGKey(40)
        model = EphysTokenizer(n_tokens=16, hidden_size=32, token_dim=5, key=key)
        x = jax.random.normal(key, (2, 50, 3))
        out = model(x)
        assert isinstance(out, TokenizerOutput)
        assert out.reconstruction.shape == (2, 50, 3)
        assert out.token_weights.shape == (2, 3, 50, 16)
        assert out.token_ids.shape == (2, 3, 50)
        assert out.loss.shape == ()

    def test_forward_vq(self):
        key = jax.random.PRNGKey(41)
        model = EphysTokenizer(n_tokens=16, hidden_size=32, quantizer_type="vq", key=key)
        x = jax.random.normal(key, (2, 50, 3))
        out = model(x)
        assert out.reconstruction.shape == (2, 50, 3)

    def test_forward_fsq(self):
        key = jax.random.PRNGKey(42)
        model = EphysTokenizer(hidden_size=32, quantizer_type="fsq",
                               fsq_levels=(4, 3, 2), key=key)
        assert model.n_tokens == 24
        x = jax.random.normal(key, (2, 50, 3))
        out = model(x)
        assert out.reconstruction.shape == (2, 50, 3)

    def test_loss_decreases(self):
        key = jax.random.PRNGKey(43)
        model = EphysTokenizer(n_tokens=16, hidden_size=32, token_dim=5, key=key)
        x = jax.random.normal(key, (4, 50, 2))

        out_before = model(x)
        loss_before = out_before.loss

        # One gradient step
        @eqx.filter_value_and_grad
        def loss_fn(m):
            return m(x, jnp.array(1.0)).loss

        import optax
        opt = optax.adam(1e-3)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        loss, grads = loss_fn(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        out_after = model(x)
        loss_after = out_after.loss
        assert loss_after < loss_before

    def test_tokenize_inference(self):
        key = jax.random.PRNGKey(44)
        model = EphysTokenizer(n_tokens=16, hidden_size=32, key=key)
        x = jax.random.normal(key, (2, 50, 3))
        ids = model.tokenize(x)
        assert ids.shape == (2, 3, 50)
        assert ids.dtype == jnp.int32

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(45)
        model = EphysTokenizer(n_tokens=16, hidden_size=32, key=key)
        x = jax.random.normal(key, (2, 50, 3))
        # Already uses @eqx.filter_jit, so this should succeed
        out = model(x)
        assert out.loss.shape == ()


# ── Baselines ───────────────────────────────────────────────────────────────

class TestBaselines:
    def test_mu_law_range(self):
        x = jax.random.normal(jax.random.PRNGKey(50), (1000,))
        tokens = mu_law_tokenize(x, n_tokens=256)
        assert tokens.min() >= 0
        assert tokens.max() <= 255

    def test_mu_law_roundtrip(self):
        x = jax.random.normal(jax.random.PRNGKey(51), (1000,))
        tokens = mu_law_tokenize(x, n_tokens=256)
        x_hat = mu_law_detokenize(tokens, n_tokens=256)
        # Rough reconstruction (quantization error bounded by 1/n_tokens)
        # Detokenized is in [-1,1], so compare normalized
        x_norm = x / jnp.maximum(jnp.max(jnp.abs(x)), 1e-10)
        assert jnp.mean(jnp.abs(x_norm - x_hat)) < 0.05

    def test_quantile_equal_freq(self):
        key = jax.random.PRNGKey(52)
        x = jax.random.normal(key, (10000,))
        tokens, boundaries = quantile_tokenize(x, n_tokens=10)
        # Each bin should have roughly equal count
        for i in range(10):
            count = jnp.sum(tokens == i)
            assert 800 < int(count) < 1200  # 1000 +/- 200


# ── Vocabulary Refactoring ──────────────────────────────────────────────────

class TestVocabRefactoring:
    def test_output_invariant(self):
        key = jax.random.PRNGKey(60)
        model = EphysTokenizer(n_tokens=16, hidden_size=32, token_dim=5, key=key)
        x = jax.random.normal(key, (2, 50, 3))

        out_before = model(x, temperature=jnp.array(1.0))
        model_new, label_map = refactor_vocabulary(model, out_before.token_weights)
        out_after = model_new(x, temperature=jnp.array(1.0))

        # Reconstruction should be identical (permutation doesn't change output)
        assert jnp.allclose(out_before.reconstruction, out_after.reconstruction, atol=1e-4)


# ── Metrics ─────────────────────────────────────────────────────────────────

class TestPVE:
    def test_perfect(self):
        x = jax.random.normal(jax.random.PRNGKey(70), (100,))
        assert jnp.isclose(pve(x, x), 100.0)

    def test_zero(self):
        x = jax.random.normal(jax.random.PRNGKey(71), (100,))
        assert jnp.isclose(pve(x, jnp.zeros_like(x)), 0.0)

    def test_per_channel(self):
        x = jax.random.normal(jax.random.PRNGKey(72), (2, 50, 3))
        ch_pve = pve_per_channel(x, x)
        assert ch_pve.shape == (3,)
        assert jnp.allclose(ch_pve, 100.0)


class TestTokenUtilization:
    def test_full_utilization(self):
        ids = jnp.arange(10)
        assert jnp.isclose(token_utilization(ids, 10), 1.0)

    def test_partial_utilization(self):
        ids = jnp.zeros(100, dtype=jnp.int32)  # Only token 0 used
        assert jnp.isclose(token_utilization(ids, 10), 0.1)


# ── Training ────────────────────────────────────────────────────────────────

class TestTraining:
    def test_smoke_2_epochs(self):
        key = jax.random.PRNGKey(80)
        model = EphysTokenizer(n_tokens=8, hidden_size=16, token_dim=3, key=key)
        data = jax.random.normal(key, (4, 30, 2))

        model, history = fit(
            model, data,
            n_epochs=2, batch_size=4, lr=1e-3, key=key,
            refactor_vocab=False,
        )
        assert len(history["loss"]) == 2
        assert history["loss"][1] < history["loss"][0]


# ── Extensions ──────────────────────────────────────────────────────────────

class TestExtensions:
    def test_signature_tokenizer(self):
        from neurojax.tokenizer._extensions import SignatureTokenizer
        import signax

        # Compute sig_dim for (window_size, C+1) path at depth 3
        # sig_dim depends on signax implementation; use a probe
        key = jax.random.PRNGKey(90)
        probe = jax.random.normal(key, (100, 4))  # 3 channels + 1 time
        sig_dim = signax.signature(probe, 3).shape[0]

        tok = SignatureTokenizer(
            sig_dim=sig_dim, embed_dim=16, n_tokens=32,
            depth=3, window_size=100, stride=50, key=key,
        )
        data = jax.random.normal(key, (200, 3))
        ids, z = tok(data)
        n_windows = (200 - 100) // 50 + 1
        assert ids.shape == (n_windows,)
        assert z.shape == (n_windows, 16)

    def test_riemannian_tokenizer(self):
        from neurojax.tokenizer._extensions import RiemannianTokenizer

        key = jax.random.PRNGKey(91)
        tok = RiemannianTokenizer(
            n_channels=3, embed_dim=16, n_tokens=32,
            window_size=50, stride=25, key=key,
        )
        data = jax.random.normal(key, (3, 200))
        ids, z = tok(data)
        n_windows = (200 - 50) // 25 + 1
        assert ids.shape == (n_windows,)

    def test_tfr_tokenizer(self):
        from neurojax.tokenizer._extensions import TFRTokenizer

        key = jax.random.PRNGKey(92)
        tok = TFRTokenizer(
            patch_freq=4, patch_time=8, embed_dim=16, n_tokens=32, key=key,
        )
        # Need enough time samples for superlet at lowest freq
        data = jax.random.normal(key, (2, 500))
        sfreq = 250.0
        freqs = tuple(jnp.linspace(8, 30, 8).tolist())
        ids, z = tok(data, sfreq, freqs)
        assert ids.shape[0] == 2  # n_channels
        assert ids.ndim == 2


# ── Dynamics Regularizer ────────────────────────────────────────────────────

class TestKoopmanRegularizer:
    def test_forward(self):
        from neurojax.tokenizer._dynamics_reg import KoopmanRegularizer

        reg = KoopmanRegularizer(rank=4, lambda_k=0.01)
        embeddings = jax.random.normal(jax.random.PRNGKey(100), (50, 8))
        loss = reg(embeddings)
        assert loss.shape == ()
        assert jnp.isfinite(loss)


# ── Consumer ────────────────────────────────────────────────────────────────

class TestTokenConsumer:
    def test_forward_shape(self):
        from neurojax.tokenizer._consumer import TokenConsumer

        key = jax.random.PRNGKey(110)
        consumer = TokenConsumer(vocab_size=32, embed_dim=16, n_layers=1,
                                 n_heads=2, max_seq_len=64, key=key)
        ids = jax.random.randint(key, (30,), 0, 32)
        logits = consumer(ids)
        assert logits.shape == (30, 32)

    def test_loss(self):
        from neurojax.tokenizer._consumer import TokenConsumer

        key = jax.random.PRNGKey(111)
        consumer = TokenConsumer(vocab_size=32, embed_dim=16, n_layers=1,
                                 n_heads=2, key=key)
        ids = jax.random.randint(key, (30,), 0, 32)
        loss = consumer.loss(ids)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
