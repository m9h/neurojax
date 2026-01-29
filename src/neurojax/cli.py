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
