# © NeuroJAX developers
#
# License: BSD (3-clause)

import click
import jax
import jax.numpy as jnp
from pathlib import Path
import time
import equinox as eqx

from neurojax.io.loader import load_data
from neurojax.io.bridge import mne_to_jax, jax_to_mne
from neurojax.preprocessing.filter import filter_data
from neurojax.preprocessing.ica import FastICA
from neurojax.analysis.dimensionality import PPCA
from neurojax.analysis.spectral import SpecParam
from neurojax.analysis.timefreq import morlet_transform

@click.group()
def main():
    """NeuroJAX: JAX-accelerated Electrophysiology Analysis."""
    pass

@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file path.")
@click.option("--highpass", "-hp", type=float, help="Highpass filter cutoff (Hz).")
@click.option("--lowpass", "-lp", type=float, help="Lowpass filter cutoff (Hz).")
@click.option("--ica", is_flag=True, help="Run FastICA using Consensus Dimensionality.")
@click.option("--n-components", type=int, default=None, help="Force N components (overrides consensus).")
@click.option("--spectral", is_flag=True, help="Run SpecParam (FOOOF) analysis.")
@click.option("--wavelet", is_flag=True, help="Compute Morlet Wavelet TFR.")
@click.option("--device", default="cpu", help="JAX device (cpu, gpu, tpu).")
def process(input_file, output, highpass, lowpass, ica, n_components, spectral, wavelet, device):
    """
    Process an EEG/MEG file with RYTHMIC pipeline.
    """
    click.echo(f"Running on {jax.devices()}")

    # 1. Load Data
    click.echo(f"Loading {input_file}...")
    start_time = time.time()
    raw = load_data(input_file, preload=True)
    click.echo(f"Loaded {raw.info['nchan']} channels, {raw.times[-1]:.2f}s duration.")
    
    # 2. Bridge to JAX
    data, sfreq = mne_to_jax(raw)
    
    # 3. Filtering
    if highpass or lowpass:
        import scipy.signal
        nyq = 0.5 * sfreq
        if highpass and lowpass:
            b, a = scipy.signal.iirfilter(4, [highpass / nyq, lowpass / nyq], btype='band')
        elif highpass:
            b, a = scipy.signal.iirfilter(4, highpass / nyq, btype='highpass')
        elif lowpass:
            b, a = scipy.signal.iirfilter(4, lowpass / nyq, btype='lowpass')
        data = filter_data(data, jnp.array(b), jnp.array(a))
        click.echo("Filtering applied.")

    # 4. ICA with Consensus Dim Est
    if ica:
        if n_components is None:
            click.echo("Estimating dimensionality (AIC/BIC Consensus)...")
            n_components = PPCA.estimate_dimensionality(data, method='consensus')
            click.echo(f"Consensus Intrinsic Dimensionality: {n_components}")
        
        click.echo(f"Running FastICA (n={n_components})...")
        model = FastICA(n_components=n_components)
        
        @eqx.filter_jit
        def fit_fn(m, d): return m.fit(d)
        
        model = fit_fn(model, data)
        click.echo("ICA Converged.")

    # 5. Spectral Parameterization (SpecParam)
    if spectral:
        click.echo("Running SpecParam (FOOOF-like)...")
        # Compute PSD using Welch (scipy) then fit
        import scipy.signal
        f, Pxx = scipy.signal.welch(data, fs=sfreq, nperseg=int(sfreq*2))
        Pxx = jnp.array(Pxx)
        f = jnp.array(f)
        
        # Fit first channel for demo
        sp_model = SpecParam.fit(f, jnp.log10(Pxx[0]), n_peaks=3)
        off, knee, exp = sp_model.aperiodic_params
        click.echo(f"Ch 0 Aperiodic: Offset={off:.2f}, Knee={knee:.2f}, Exp={exp:.2f}")

    # 6. Wavelet
    if wavelet:
        click.echo("Computing Morlet Transform...")
        freqs = jnp.logspace(jnp.log10(2), jnp.log10(50), 20)
        tfr = morlet_transform(data[:1], sfreq, freqs) # Just first channel
        click.echo(f"TFR computed: {tfr.shape}")

    # 7. Save
    if output:
        if ica and not (spectral or wavelet):
             processed_raw = jax_to_mne(data, raw)
             processed_raw.save(output, overwrite=True)
             click.echo(f"Saved to {output}")
        else:
             click.echo("Result saving for Spectral/Wavelet not fully implemented in CLI demo.")
             
    click.echo(f"Done in {time.time() - start_time:.2f}s")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--n-tokens", default=64, help="Number of tokens in codebook.")
@click.option("--hidden-size", default=64, help="GRU hidden state dimension.")
@click.option("--token-dim", default=10, help="Decoder kernel length.")
@click.option("--n-layers", default=1, help="Number of GRU layers.")
@click.option(
    "--quantizer",
    default="temperature",
    type=click.Choice(["temperature", "vq", "fsq"]),
    help="Quantization backend.",
)
@click.option("--n-epochs", default=100, help="Training epochs.")
@click.option("--batch-size", default=32, help="Batch size.")
@click.option("--lr", default=1e-3, type=float, help="Learning rate.")
@click.option("--segment-length", default=200, type=int, help="Segment length in samples.")
@click.option("--output", "-o", required=True, help="Output model path (.eqx).")
@click.option("--seed", default=0, type=int, help="Random seed.")
def tokenize(input_file, n_tokens, hidden_size, token_dim, n_layers, quantizer,
             n_epochs, batch_size, lr, segment_length, output, seed):
    """Train an electrophysiology tokenizer on MEG/EEG data."""
    from neurojax.tokenizer import EphysTokenizer, fit, pve

    click.echo(f"Running on {jax.devices()}")
    start_time = time.time()

    # Load and convert
    click.echo(f"Loading {input_file}...")
    raw = load_data(input_file, preload=True)
    data, sfreq = mne_to_jax(raw)
    click.echo(f"Loaded {data.shape[0]} channels, {data.shape[1]} samples @ {sfreq} Hz")

    # Segment into (N, L, C) format: data is (C, T) -> segments of (L, C)
    C, T = data.shape
    n_segments = T // segment_length
    if n_segments == 0:
        click.echo("Error: data shorter than segment-length.", err=True)
        return
    data_segmented = data[:, :n_segments * segment_length]
    data_segmented = data_segmented.reshape(C, n_segments, segment_length)
    data_segmented = jnp.transpose(data_segmented, (1, 2, 0))  # (N, L, C)
    click.echo(f"Segmented into {n_segments} segments of length {segment_length}")

    # Build model
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = EphysTokenizer(
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        token_dim=token_dim,
        n_layers=n_layers,
        quantizer_type=quantizer,
        key=model_key,
    )
    click.echo(f"Built EphysTokenizer: {quantizer} quantizer, {n_tokens} tokens")

    # Train
    click.echo(f"Training for {n_epochs} epochs...")
    model, history = fit(
        model, data_segmented,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr, key=key,
    )
    click.echo(f"Final loss: {history['loss'][-1]:.6f}")

    # Evaluate
    out = model(data_segmented[:min(batch_size, n_segments)], temperature=jnp.array(0.01))
    score = pve(data_segmented[:min(batch_size, n_segments)], out.reconstruction)
    click.echo(f"PVE: {float(score):.1f}%")

    # Save
    eqx.tree_serialise_leaves(output, model)
    click.echo(f"Saved to {output}")
    click.echo(f"Done in {time.time() - start_time:.2f}s")


@main.command("evaluate-tokenizer")
@click.argument("model_file", type=click.Path(exists=True))
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--n-tokens", default=64, help="Number of tokens (must match model).")
@click.option("--hidden-size", default=64, help="Hidden size (must match model).")
@click.option("--token-dim", default=10, help="Token dim (must match model).")
@click.option("--quantizer", default="temperature",
              type=click.Choice(["temperature", "vq", "fsq"]))
@click.option("--segment-length", default=200, type=int)
@click.option("--seed", default=0, type=int)
def evaluate_tokenizer_cmd(model_file, input_file, n_tokens, hidden_size, token_dim,
                           quantizer, segment_length, seed):
    """Evaluate a trained tokenizer on data."""
    from neurojax.tokenizer import EphysTokenizer, pve, pve_per_channel, token_utilization

    key = jax.random.PRNGKey(seed)

    # Rebuild model skeleton and load weights
    model = EphysTokenizer(
        n_tokens=n_tokens, hidden_size=hidden_size, token_dim=token_dim,
        quantizer_type=quantizer, key=key,
    )
    model = eqx.tree_deserialise_leaves(model_file, model)

    # Load data
    raw = load_data(input_file, preload=True)
    data, sfreq = mne_to_jax(raw)
    C, T = data.shape
    n_segments = T // segment_length
    data_seg = data[:, :n_segments * segment_length].reshape(C, n_segments, segment_length)
    data_seg = jnp.transpose(data_seg, (1, 2, 0))

    # Evaluate
    out = model(data_seg, temperature=jnp.array(0.01))
    click.echo(f"PVE (global): {float(pve(data_seg, out.reconstruction)):.1f}%")
    ch_pve = pve_per_channel(data_seg, out.reconstruction)
    click.echo(f"PVE (per-ch): min={float(jnp.min(ch_pve)):.1f}%, max={float(jnp.max(ch_pve)):.1f}%")
    util = token_utilization(out.token_ids, n_tokens)
    click.echo(f"Token utilization: {float(util)*100:.1f}%")
