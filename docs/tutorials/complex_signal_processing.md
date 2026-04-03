# Complex Signal Processing at Every Stage

The analytic signal --- computed via a simple Hilbert transform ---
provides instantaneous amplitude and phase. These are more fundamental
representations than the raw real signal for virtually all neuroscience
questions. This tutorial demonstrates why working in complex space from
the start benefits every processing stage.

## The analytic signal

Given a real signal x(t), the analytic signal is:

    z(t) = x(t) + i * H[x(t)]

where H is the Hilbert transform. From z(t) you get:

- **|z(t)|** = instantaneous amplitude (envelope)
- **angle(z(t))** = instantaneous phase
- **d(phase)/dt** = instantaneous frequency

All three are computable from a single FFT operation.

```python
from neurojax.analysis.analytic import hilbert, envelope, instantaneous_phase

# Convert real MEG channels to analytic signal
z = hilbert(data)          # (n_channels, n_times) complex
amp = jnp.abs(z)           # amplitude envelope
phase = jnp.angle(z)       # instantaneous phase
```

## Stage 1: Preprocessing

### Envelope-based artifact detection

Raw amplitude thresholding is noisy. The Hilbert envelope is smooth
and gives a principled artifact detector:

```python
from neurojax.analysis.analytic import envelope_zscore, detect_artifacts

# Z-score the amplitude envelope per channel
z_scores = envelope_zscore(data)

# Artifact if any channel envelope exceeds 4 std
bad_samples = detect_artifacts(data, threshold=4.0)
clean_data = data[:, ~bad_samples]
```

**Why it's better:** The envelope integrates over the carrier oscillation,
so a brief muscle artifact that oscillates at 50 Hz but has high envelope
is detected cleanly, while a genuine large alpha burst is not.

## Stage 2: Source imaging

### Phase-based connectivity (PLV)

After source reconstruction, phase locking value measures functional
connectivity without being confounded by amplitude:

```python
from neurojax.analysis.analytic import plv_matrix, imaginary_plv

# Phase locking between all source pairs
plv = plv_matrix(source_timecourses)   # (n_sources, n_sources)

# Imaginary PLV: robust to volume conduction
# (zero-lag leakage has zero imaginary part)
iplv = imaginary_plv(source_timecourses)
```

**Why it matters:** PLV separates "are these regions phase-locked?"
from "are they co-activated?" Envelope correlation answers the latter;
PLV answers the former. Both require the analytic signal.

### Envelope correlation (resting-state MEG)

The standard measure for MEG resting-state networks (Brookes et al.
2011, Hipp et al. 2012):

```python
from neurojax.analysis.analytic import envelope_correlation

# After source reconstruction + parcellation
fc = envelope_correlation(parcellated_sources)  # (n_parcels, n_parcels)
```

## Stage 3: Dynamics

### Phase-amplitude coupling (PAC)

Theta-gamma coupling, a hallmark of memory encoding, requires
simultaneous access to phase (from theta) and amplitude (from gamma):

```python
from neurojax.analysis.analytic import phase_amplitude_coupling, narrowband_analytic

# Extract theta phase and gamma amplitude
z_theta = narrowband_analytic(data, sfreq=1000, fmin=4, fmax=8)
z_gamma = narrowband_analytic(data, sfreq=1000, fmin=30, fmax=80)

# Modulation index: how much does theta phase modulate gamma amplitude?
mi = phase_amplitude_coupling(jnp.real(z_theta), jnp.real(z_gamma))
```

### Instantaneous frequency for non-stationary dynamics

```python
from neurojax.analysis.analytic import instantaneous_frequency

# Alpha band instantaneous frequency
z_alpha = narrowband_analytic(data, sfreq=1000, fmin=8, fmax=13)
freq = instantaneous_frequency(jnp.real(z_alpha), sfreq=1000)
# freq tracks moment-to-moment alpha frequency (typically 9-11 Hz)
```

## Stage 4: Statistics

### Circular statistics on phase

Phase is a circular variable --- standard mean/variance are wrong.
Use circular statistics:

```python
from neurojax.analysis.analytic import circular_mean, circular_variance, rayleigh_z

phase_at_stimulus = instantaneous_phase(data)[:, stimulus_onset]

# Average phase at stimulus onset across trials
mean_phase = circular_mean(phase_at_stimulus, axis=-1)

# Test for non-uniform phase distribution (phase locking to stimulus)
z_stat = rayleigh_z(phase_at_stimulus, axis=-1)
# Large Z → significant phase reset at stimulus onset
```

## Narrowband analytic: filter + Hilbert in one step

Separate bandpass → Hilbert is two FFTs. Combined is one:

```python
from neurojax.analysis.analytic import narrowband_analytic

# Single FFT: bandpass [8, 13] Hz + analytic signal
z_alpha = narrowband_analytic(data, sfreq=1000, fmin=8, fmax=13)

# Now you have amplitude AND phase in the alpha band
alpha_power = jnp.abs(z_alpha) ** 2
alpha_phase = jnp.angle(z_alpha)
```

## Connection to existing neurojax modules

| Module | Uses complex signal | How |
|--------|-------------------|-----|
| `source/beamformer.py` | DICS | Cross-spectral density matrix (complex) |
| `source/champagne.py` | Imaginary coherence | Complex cross-spectrum |
| `analysis/complex_ica.py` | Complex ICA | Analytic signal decomposition |
| `analysis/multitaper.py` | Coherence | Cross-power spectral density |
| `analysis/state_spectra.py` | State coherence | CPSD per HMM state |
| `data/loading.py` | Amplitude envelope | Hilbert for MEG preparation |
| `analysis/analytic.py` | **All of the above** | Unified complex processing |

## Why JAX makes this natural

JAX natively supports complex arithmetic. The Hilbert transform is a
single `jnp.fft.fft` → multiply → `jnp.fft.ifft`. All operations
(PLV, PAC, envelope correlation) are differentiable, so you can
optimise parameters through phase-based loss functions:

```python
def phase_locking_loss(params):
    sources = inverse_model(data, params)
    plv = plv_matrix(sources)
    return -jnp.mean(plv)  # maximise phase locking

grad = jax.grad(phase_locking_loss)(params)
```
