# MEG Preprocessing: Filtering, ASR, and Artifact Removal

MEG recordings are always contaminated by artifacts -- environmental noise,
cardiac signals, eye blinks, muscle activity, and sensor jumps.  Robust
preprocessing is the foundation of any trustworthy analysis.  This tutorial
walks through the NeuroJAX preprocessing toolkit: IIR filtering, Artifact
Subspace Reconstruction (ASR), ICA-based artifact rejection, and quality
metrics.

Every step runs as pure JAX, so the entire preprocessing chain is
differentiable and GPU-acceleratable via `jax.jit` and `jax.vmap`.

```{contents}
:depth: 3
:local:
```

## Prerequisites

```bash
uv sync
```

```python
import jax
import jax.numpy as jnp

print(jax.devices())  # should list at least one device
```

---

## 1. Generating Synthetic MEG Data with Artifacts

For testing and validation it is useful to work with synthetic data where
the ground truth is known.  We create multi-channel data with controlled
spectral content and then inject known artifacts.

```python
def make_sine(freq, sfreq, duration, n_channels=1, key=None, noise_level=0.0):
    """Generate a multi-channel sine wave with optional noise.

    Returns array of shape (n_channels, n_times).
    """
    n_times = int(sfreq * duration)
    t = jnp.linspace(0, duration, n_times)
    signal = jnp.sin(2 * jnp.pi * freq * t)
    data = jnp.tile(signal, (n_channels, 1))
    if noise_level > 0 and key is not None:
        data = data + noise_level * jax.random.normal(key, data.shape)
    return data
```

Generate a 4-channel recording at 1000 Hz with a 10 Hz signal and background
noise:

```python
sfreq = 1000.0
duration = 1.0
n_channels = 4
key = jax.random.PRNGKey(0)

# Clean signal: 10 Hz sine + light noise
clean_data = make_sine(10.0, sfreq, duration, n_channels=n_channels,
                       key=key, noise_level=0.1)
print(f"Clean data shape: {clean_data.shape}")  # (4, 1000)
```

Now inject a large artifact -- simulating an eye blink or sensor jump -- on
channel 0 between samples 200 and 400:

```python
artifact_amplitude = 50.0
corrupted_data = clean_data.at[0, 200:400].set(
    clean_data[0, 200:400] + artifact_amplitude * jnp.ones(200)
)

rms_clean = float(jnp.sqrt(jnp.mean(clean_data[0] ** 2)))
rms_corrupted = float(jnp.sqrt(jnp.mean(corrupted_data[0] ** 2)))
print(f"Channel 0 RMS -- clean: {rms_clean:.3f}, corrupted: {rms_corrupted:.3f}")
```

---

## 2. Bandpass Filtering with `filter_data()`

NeuroJAX implements IIR filtering using a Direct Form II Transposed structure
built on `jax.lax.scan`.  Unlike `scipy.signal.lfilter`, this implementation
is fully differentiable and JIT-compilable.

### 2.1 The IIR Filter Equation

A causal IIR filter processes input $x[n]$ to produce output $y[n]$ via the
difference equation:

$$
a_0\, y[n] = \sum_{k=0}^{M} b_k\, x[n-k] - \sum_{k=1}^{N} a_k\, y[n-k]
$$

The numerator coefficients $\mathbf{b}$ define the feedforward (FIR) part,
while the denominator coefficients $\mathbf{a}$ define the feedback (IIR)
part.  When $\mathbf{a} = [1]$, the filter is purely FIR.

### 2.2 Identity Filter (passthrough)

The simplest filter has $b = [1]$, $a = [1]$ -- the output equals the input:

```python
from neurojax.preprocessing.filter import filter_data

key = jax.random.PRNGKey(0)
data = jax.random.normal(key, (3, 200))
b = jnp.array([1.0])
a = jnp.array([1.0])

out = filter_data(data, b, a)
print(f"Identity filter -- allclose: {bool(jnp.allclose(out, data, atol=1e-6))}")
```

### 2.3 Moving-Average Low-Pass Filter

A simple $N$-tap moving average acts as a low-pass filter.  It attenuates
high-frequency components while passing low frequencies:

```python
# Create a signal with 5 Hz and 200 Hz components
low_freq = make_sine(5.0, sfreq, duration)     # passes through
high_freq = make_sine(200.0, sfreq, duration)   # should be attenuated
data = low_freq + high_freq

# 5-tap moving average
n_taps = 5
b = jnp.ones(n_taps) / n_taps
a = jnp.array([1.0])

filtered = filter_data(data, b, a)

# Verify high-frequency attenuation via FFT
fft_in = jnp.abs(jnp.fft.rfft(data[0])) ** 2
fft_out = jnp.abs(jnp.fft.rfft(filtered[0])) ** 2
hf_idx = int(200 * data.shape[1] / sfreq)
ratio = float(fft_out[hf_idx] / (fft_in[hf_idx] + 1e-12))
print(f"High-frequency power ratio after filtering: {ratio:.4f}")
# Should be < 0.5, indicating substantial attenuation
```

```{note}
The `filter_data()` function operates along the last axis and supports
both 1-D inputs (single channel) and batched 2-D inputs
`(n_channels, n_times)`.  Internally, `jax.vmap` parallelizes across
channels.
```

### 2.4 Shape Preservation

The filter always preserves the input shape:

```python
data_4ch = jnp.ones((4, 100))
b = jnp.array([0.5, 0.5])
a = jnp.array([1.0])
out = filter_data(data_4ch, b, a)
print(f"Input shape: {data_4ch.shape}, Output shape: {out.shape}")  # both (4, 100)
```

```{tip}
For designing Butterworth, Chebyshev, or other standard filter
coefficients, use `scipy.signal.butter()` or `scipy.signal.cheby1()` to
compute `b` and `a`, then pass them to `filter_data()`.  The filtering
itself runs in JAX; only the coefficient design needs SciPy.
```

---

## 3. ASR Calibration with `calibrate_asr()`

Artifact Subspace Reconstruction (ASR) is a principled method for removing
high-variance artifacts while preserving the underlying neural signal.  The
algorithm works in two phases:

1. **Calibration**: Learn the statistics of clean data in a PCA-rotated
   coordinate system
2. **Application**: Identify and suppress components whose variance exceeds
   a threshold relative to the calibration statistics

### 3.1 How ASR Works

During calibration, ASR computes the eigendecomposition of the data
covariance matrix:

$$
\mathbf{C} = \frac{1}{N-1}\, \mathbf{X}_{\text{clean}}\, \mathbf{X}_{\text{clean}}^T = \mathbf{V}\, \boldsymbol{\Lambda}\, \mathbf{V}^T
$$

The eigenvectors $\mathbf{V}$ define a **mixing matrix** that rotates sensor
data into statistically independent principal components.  The square roots
of the eigenvalues give the expected standard deviation of each component.

During application, each sliding window of data is projected into this
component space.  Any component whose RMS amplitude exceeds
$\text{cutoff} \times \sigma_{\text{calibration}}$ is zeroed out, and the
data is reconstructed from the surviving components.

### 3.2 Calibrating on Clean Reference Data

```python
from neurojax.preprocessing.asr import ASRState, calibrate_asr, apply_asr

key = jax.random.PRNGKey(1)
n_channels, n_times = 4, 500

# Simulate clean reference data
clean_ref = jax.random.normal(key, (n_channels, n_times)) * 0.1

# Calibrate ASR with a cutoff of 3 standard deviations
state = calibrate_asr(clean_ref, cutoff=3.0)
```

The returned `ASRState` is a named tuple containing the calibration
parameters:

```python
print(f"Mixing matrix shape:    {state.mixing_matrix.shape}")    # (4, 4)
print(f"Component stdevs shape: {state.component_stdevs.shape}") # (4,)
print(f"Cutoff:                 {state.cutoff}")                  # 3.0
```

### 3.3 Verifying Calibration Properties

The mixing matrix consists of orthogonal eigenvectors:

```python
M = state.mixing_matrix
orthogonality = M.T @ M
print("M^T @ M (should be ~I):")
print(jnp.round(orthogonality, 4))
assert jnp.allclose(orthogonality, jnp.eye(n_channels), atol=1e-5)
```

The component standard deviations should all be positive:

```python
print(f"Component stdevs: {state.component_stdevs}")
assert jnp.all(state.component_stdevs > 0)
```

```{note}
The `cutoff` parameter controls sensitivity.  Lower values (e.g., 3.0)
are more aggressive and will reject more components, while higher values
(e.g., 10.0) are more lenient.  For MEG data, values between 3 and 5 are
typical.  Start with `cutoff=5.0` (the default) and decrease if artifacts
remain visible.
```

---

## 4. Applying ASR with `apply_asr()`

Once calibrated, ASR processes new (potentially corrupted) data using sliding
windows with overlap-add reconstruction.

### 4.1 Artifact Removal

```python
k1, k2 = jax.random.split(jax.random.PRNGKey(5))
n_channels, n_times = 4, 600

# Generate clean data and calibrate
clean = jax.random.normal(k1, (n_channels, n_times)) * 0.1
state = calibrate_asr(clean, cutoff=3.0)

# Inject a large artifact on channel 0 in the middle segment
corrupted = clean.at[0, 200:400].set(
    clean[0, 200:400] + 50.0 * jnp.ones(200)
)

# Apply ASR
cleaned = apply_asr(corrupted, state, window_size=100, step_size=50)
print(f"Output shape: {cleaned.shape}")  # same as input: (4, 600)
```

### 4.2 Measuring Artifact Reduction

Compare the RMS amplitude before and after ASR:

```python
rms_corrupted = float(jnp.sqrt(jnp.mean(corrupted[0] ** 2)))
rms_cleaned = float(jnp.sqrt(jnp.mean(cleaned[0] ** 2)))
print(f"Channel 0 RMS -- corrupted: {rms_corrupted:.3f}, "
      f"cleaned: {rms_cleaned:.3f}")
print(f"Reduction: {(1 - rms_cleaned/rms_corrupted)*100:.1f}%")
assert rms_cleaned < rms_corrupted, "ASR should reduce artifact RMS"
```

### 4.3 Preserving Clean Data

A critical property of ASR: when the data looks like the calibration data
(i.e., is artifact-free), ASR should not distort it:

```python
key = jax.random.PRNGKey(6)
n_channels, n_times = 3, 400

# All-clean data
data = jax.random.normal(key, (n_channels, n_times)) * 0.1
state = calibrate_asr(data, cutoff=10.0)  # lenient cutoff
cleaned = apply_asr(data, state, window_size=100, step_size=50)

# Correlation between original and cleaned should be high
corr = float(jnp.corrcoef(data.ravel(), cleaned.ravel())[0, 1])
print(f"Correlation (clean data preserved): {corr:.4f}")
assert corr > 0.8, f"Clean data should be mostly preserved, got corr={corr}"
```

```{tip}
The `window_size` and `step_size` parameters control the temporal
resolution of ASR.  Smaller windows detect shorter artifacts but may
introduce edge effects.  A `step_size` of `window_size // 2` (50%
overlap) with Hanning window weighting provides smooth reconstruction.
For MEG at 1000 Hz, `window_size=500` (0.5 s) and `step_size=250` is a
reasonable starting point.
```

---

## 5. ICA-Based Artifact Rejection Pipeline

While ASR is fully automated, ICA provides a complementary approach where
individual components can be inspected and classified.  A typical ICA
artifact rejection pipeline:

1. Filter the data to the band of interest
2. Run ICA to decompose into independent components
3. Identify artifact components (by topography, time course, or correlation
   with reference channels)
4. Zero out artifact components and reconstruct

```python
from neurojax.preprocessing.ica import FastICA

# Step 1: Generate multi-channel data with a known artifact
key = jax.random.PRNGKey(14)
n_channels, n_times = 4, 2000
t = jnp.linspace(0, 1, n_times)

# Neural sources
s_alpha = jnp.sin(2 * jnp.pi * 10 * t)    # 10 Hz alpha
s_beta = jnp.sin(2 * jnp.pi * 25 * t)     # 25 Hz beta

# Artifact source (simulated eye blink: brief, high-amplitude pulse)
s_artifact = jnp.zeros(n_times)
s_artifact = s_artifact.at[500:600].set(5.0)
s_artifact = s_artifact.at[1200:1300].set(5.0)

sources = jnp.stack([s_alpha, s_beta, s_artifact])

# Mixing matrix (artifact projects mainly to frontal channels 0, 1)
A = jnp.array([
    [1.0, 0.3, 2.0],   # channel 0: alpha + little beta + strong artifact
    [0.5, 0.8, 1.5],   # channel 1: some alpha + beta + artifact
    [0.8, 1.0, 0.1],   # channel 2: alpha + beta, minimal artifact
    [0.3, 0.9, 0.05],  # channel 3: little alpha + beta, almost no artifact
])
X = A @ sources + 0.05 * jax.random.normal(key, (n_channels, n_times))

# Step 2: Run ICA
ica = FastICA(n_components=3, max_iter=500, tol=1e-5)
model = ica.fit(X, key=jax.random.PRNGKey(42))

# Step 3: Examine components
components = model.components_
for i in range(3):
    peak = float(jnp.max(jnp.abs(components[i])))
    rms = float(jnp.sqrt(jnp.mean(components[i] ** 2)))
    print(f"Component {i}: peak={peak:.3f}, rms={rms:.3f}")
```

```{note}
In practice, artifact components are identified by their spatial
topography (using `model.mixing_` columns), their time course shape, and
their correlation with reference channels (EOG, ECG).  NeuroJAX also
provides Riemannian artifact detection in
`neurojax.preprocessing.artifact.detect_artifacts_riemann` for automated
epoch rejection.
```

---

## 6. Quality Metrics Before/After Preprocessing

Quantitative metrics help you assess whether preprocessing improved the data
without removing too much signal.

### 6.1 RMS Amplitude

```python
def compute_rms(data):
    """Per-channel RMS amplitude."""
    return jnp.sqrt(jnp.mean(data ** 2, axis=1))

rms_before = compute_rms(corrupted)
rms_after = compute_rms(cleaned)
for ch in range(n_channels):
    print(f"Channel {ch}: RMS before={float(rms_before[ch]):.4f}, "
          f"after={float(rms_after[ch]):.4f}")
```

### 6.2 Signal-to-Noise Ratio

If you have a ground-truth clean signal (as in simulations), compute the SNR
improvement:

```python
def snr_db(signal, noise):
    """Signal-to-noise ratio in dB."""
    power_signal = jnp.mean(signal ** 2)
    power_noise = jnp.mean(noise ** 2)
    return 10 * jnp.log10(power_signal / (power_noise + 1e-12))

noise_before = corrupted - clean
noise_after = cleaned - clean
snr_before = float(snr_db(clean, noise_before))
snr_after = float(snr_db(clean, noise_after))
print(f"SNR: before={snr_before:.1f} dB, after={snr_after:.1f} dB")
```

### 6.3 Correlation with Ground Truth

```python
for ch in range(n_channels):
    corr_before = float(jnp.corrcoef(clean[ch], corrupted[ch])[0, 1])
    corr_after = float(jnp.corrcoef(clean[ch], cleaned[ch])[0, 1])
    print(f"Channel {ch}: corr before={corr_before:.4f}, "
          f"after={corr_after:.4f}")
```

---

## 7. Full Preprocessing Chain Example

Here is a complete preprocessing pipeline that chains filtering, ASR, and
ICA together:

```python
import jax
import jax.numpy as jnp
from neurojax.preprocessing.filter import filter_data
from neurojax.preprocessing.asr import calibrate_asr, apply_asr
from neurojax.preprocessing.ica import FastICA

# ── 1. Simulate raw MEG data ──────────────────────────────────────────────
key = jax.random.PRNGKey(0)
sfreq = 1000.0
n_channels = 4
n_times = 2000
t = jnp.linspace(0, 2, n_times)

# Neural signal: 10 Hz oscillation
neural = 0.5 * jnp.sin(2 * jnp.pi * 10 * t)
data = jnp.tile(neural, (n_channels, 1))
data = data + 0.05 * jax.random.normal(key, (n_channels, n_times))

# Inject artifacts
# -- High-frequency noise burst on channel 0 (samples 400-600)
hf_artifact = 10.0 * jnp.sin(2 * jnp.pi * 200 * t[400:600])
data = data.at[0, 400:600].add(hf_artifact)
# -- DC shift on channel 2 (samples 1000-1200)
data = data.at[2, 1000:1200].add(20.0)

print(f"Raw data shape: {data.shape}")

# ── 2. Bandpass filter (remove very low and very high frequencies) ────────
# Design a simple 5-tap moving average as a low-pass
n_taps = 5
b_lp = jnp.ones(n_taps) / n_taps
a_lp = jnp.array([1.0])
data_filtered = filter_data(data, b_lp, a_lp)
print("Filtering complete.")

# ── 3. ASR calibration and application ────────────────────────────────────
# Use the first 200 samples as "clean" reference for calibration
# (In practice, manually select an artifact-free segment)
clean_segment = data_filtered[:, :200]
asr_state = calibrate_asr(clean_segment, cutoff=3.0)

data_asr = apply_asr(data_filtered, asr_state,
                      window_size=100, step_size=50)
print("ASR complete.")

# ── 4. ICA for residual artifact removal ──────────────────────────────────
ica = FastICA(n_components=3, max_iter=500, tol=1e-5)
model = ica.fit(data_asr, key=jax.random.PRNGKey(42))
components = model.components_
print(f"ICA extracted {components.shape[0]} components.")

# ── 5. Quality check ─────────────────────────────────────────────────────
rms_raw = jnp.sqrt(jnp.mean(data ** 2, axis=1))
rms_processed = jnp.sqrt(jnp.mean(data_asr ** 2, axis=1))
for ch in range(n_channels):
    print(f"Channel {ch}: RMS raw={float(rms_raw[ch]):.4f} -> "
          f"processed={float(rms_processed[ch]):.4f}")

print("\nPreprocessing pipeline complete.")
```

```{tip}
In a real workflow you would also include:

- **Resampling** via `neurojax.preprocessing.resample.resample_minimal()`
  to downsample from the acquisition rate to your analysis rate
- **Bad channel interpolation** via
  `neurojax.preprocessing.interpolate.spherical_spline_interpolate()`
- **Riemannian artifact detection** via
  `neurojax.preprocessing.artifact.detect_artifacts_riemann()` for
  automated epoch rejection based on covariance-space outlier detection

All of these are JAX-native and composable with the pipeline above.
```

---

## Summary

| Step | Module | Key function |
|---|---|---|
| IIR filtering | `preprocessing.filter` | `filter_data(data, b, a)` |
| ASR calibration | `preprocessing.asr` | `calibrate_asr(clean_data, cutoff)` |
| ASR application | `preprocessing.asr` | `apply_asr(data, state, window_size, step_size)` |
| FastICA | `preprocessing.ica` | `FastICA(n_components).fit(X, key)` |
| Resampling | `preprocessing.resample` | `resample_minimal(data, orig_sfreq, target_sfreq)` |
| Interpolation | `preprocessing.interpolate` | `spherical_spline_interpolate(data, bad_idx, coords)` |
| Artifact detection | `preprocessing.artifact` | `detect_artifacts_riemann(covs, n_std)` |
