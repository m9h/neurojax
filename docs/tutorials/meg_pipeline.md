# End-to-End MEG Analysis Pipeline

This tutorial walks through a complete MEG analysis workflow using NeuroJAX,
from raw data loading through source reconstruction and statistical inference.
Every computational step runs on JAX, giving you automatic GPU/TPU acceleration
and full differentiability of the signal-processing chain.

```{contents}
:depth: 3
:local:
```

## Prerequisites

Install NeuroJAX with its documentation dependencies:

```bash
uv sync --extra doc
```

NeuroJAX depends on [MNE-Python](https://mne.tools) for file I/O and sensor
layouts, and on JAX / Equinox / Lineax for the numerical backend.  Make sure
a JAX backend is available (CPU works out of the box; for GPU see the
[JAX installation guide](https://github.com/google/jax#installation)).

```python
import jax
print(jax.devices())   # should list at least one device
```

---

## 1. Why JAX for MEG?

MEG source analysis involves repeated linear algebra on large matrices:
covariance estimation, eigendecompositions for ICA and dimensionality
reduction, beamformer weight computation, and massive permutation tests for
statistical inference.  These operations map naturally onto JAX primitives:

| Operation | NeuroJAX module | JAX advantage |
|---|---|---|
| IIR / FIR filtering | `preprocessing.filter` | `jax.lax.scan` fused loop |
| Artifact Subspace Reconstruction | `preprocessing.asr` | `vmap` over windows |
| FastICA / PICA | `analysis.ica` | `while_loop` + `jit` |
| Dimensionality estimation | `analysis.dimensionality` | Vectorised AIC/BIC |
| LCMV / DICS beamforming | `source.beamformer` | `jit`-compiled linear algebra |
| MNE / dSPM / sLORETA / eLORETA | `source.minimum_norm` | Differentiable inverse |
| Morlet time-frequency | `analysis.timefreq` | `conv_general_dilated` |
| SpecParam (FOOOF) | `analysis.spectral` | Equinox + Optax gradient descent |
| GLM + permutation testing | `glm` | `vmap` over 5 000+ permutations |
| Functional connectivity | `analysis.funcnet` | MI matrices, graph measures |

A typical 5 000-permutation GLM test on 300 sensors that takes minutes in NumPy
completes in seconds on a single GPU through `jax.vmap`.

---

## 2. Data Loading

NeuroJAX uses MNE-Python as its I/O layer, then bridges sensor data into JAX
arrays.  Any format that MNE can read (`.fif`, `.ds`, `.sqd`, BIDS, etc.)
works seamlessly.

### 2.1 Loading from a file

```python
from neurojax.io.loader import load_data
from neurojax.io.bridge import mne_to_jax, jax_to_mne

# Load an Elekta/Neuromag .fif file (or CTF .ds, KIT .sqd, ...)
raw = load_data("/data/meg/sub-01/meg/sub-01_task-faces_meg.fif",
                preload=True)
print(f"{raw.info['nchan']} channels, {raw.times[-1]:.1f} s, "
      f"sfreq = {raw.info['sfreq']} Hz")
```

### 2.2 Loading from a BIDS dataset

For BIDS-formatted datasets you can use `mne_bids` before handing data to
NeuroJAX:

```python
import mne_bids

bids_path = mne_bids.BIDSPath(
    subject="01", task="faces", datatype="meg",
    root="/data/bids/ds000117",
)
raw = mne_bids.read_raw_bids(bids_path)
raw.load_data()
```

### 2.3 Bridging to JAX

The `mne_to_jax` function extracts the data matrix and sampling frequency as
JAX arrays, casting to `float32` for efficient GPU computation:

```python
data, sfreq = mne_to_jax(raw)   # data: (n_channels, n_times), sfreq: float
print(f"JAX array shape: {data.shape}, dtype: {data.dtype}")
```

The inverse bridge `jax_to_mne` reconstitutes an MNE `Raw` object from a JAX
array, preserving the original `Info` structure:

```python
cleaned_raw = jax_to_mne(cleaned_data, template=raw)
cleaned_raw.save("sub-01_cleaned-raw.fif", overwrite=True)
```

---

## 3. Preprocessing

### 3.1 Bandpass filtering

NeuroJAX implements IIR filtering entirely in JAX via a Direct-Form II
Transposed structure inside `jax.lax.scan`.  This makes the filter
differentiable and GPU-compatible.

```python
import jax.numpy as jnp
import scipy.signal
from neurojax.preprocessing.filter import filter_data

# Design a 4th-order Butterworth bandpass 1--100 Hz
nyq = 0.5 * float(sfreq)
b, a = scipy.signal.iirfilter(4, [1.0 / nyq, 100.0 / nyq], btype="band")

# Apply on JAX — runs on GPU if available
data_filt = filter_data(data, jnp.array(b), jnp.array(a))
```

The transfer function of a 4th-order Butterworth bandpass is:

$$
H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}}
$$

where the coefficients $b_k, a_k$ are produced by `scipy.signal.iirfilter`.

```{note}
For HCP-style minimal preprocessing (highpass + resample + bad-channel
interpolation) see {py:func}`neurojax.pipeline.hcp_minimal.hcp_minimal_preproc`.
```

### 3.2 Artifact Subspace Reconstruction (ASR)

ASR is a principled method for removing transient high-variance artifacts
(eye blinks, muscle bursts, head movements) without manual annotation.
NeuroJAX implements ASR natively in JAX, using `vmap` over sliding windows
and `lax.scan` for overlap-add reconstruction.

The algorithm:

1. **Calibrate** on a clean segment --- learn the PCA mixing matrix $\mathbf{V}$
   and per-component standard deviations $\sigma_k$ from eigendecomposition
   of the calibration covariance.
2. **Apply** to each sliding window --- project into PCA space, reject
   components whose RMS exceeds $\kappa \cdot \sigma_k$ (default $\kappa = 5$),
   and reconstruct.

```python
from neurojax.preprocessing.asr import calibrate_asr, apply_asr

# Identify a clean segment (e.g., first 30 s) for calibration
n_calib = int(30.0 * sfreq)
calib_data = data_filt[:, :n_calib]

# Calibrate
asr_state = calibrate_asr(calib_data, cutoff=5.0)

# Apply ASR with 50 % overlap Hanning windows
window_samples = int(0.5 * sfreq)   # 500 ms windows
step_samples   = window_samples // 2
data_asr = apply_asr(data_filt, asr_state,
                     window_size=window_samples,
                     step_size=step_samples)
```

```{tip}
Use `cutoff=5.0` (the default) for a moderate cleaning pass.  A stricter
value like `3.0` removes more variance but risks attenuating genuine brain
signal.  Inspect the removed component (original minus cleaned) to verify.
```

### 3.3 ICA-based artifact removal

After ASR, residual stereotyped artifacts (cardiac, ocular) can be identified
and removed with ICA.  NeuroJAX provides two ICA implementations:

- **`FastICA`** (Hyvarinen 1999): A fast fixed-point ICA as an Equinox module.
  Ideal when you know the number of components.
- **`PICA`** (MELODIC-style): Probabilistic ICA with automatic dimensionality
  estimation via AIC/BIC consensus and optional GMM thresholding.

#### 3.3.1 Automatic dimensionality estimation

Before running ICA, we must decide how many components to extract.  The
`PPCA` module provides a consensus estimator that averages AIC and BIC
optima from the Laplace-approximated model evidence of Probabilistic PCA
{cite:p}`minka2000automatic,beckmann2004probabilistic`:

$$
\log p(\mathbf{X} \mid k) \approx
    \ell(\hat{\theta}_k)
    - \tfrac{1}{2} m_k \log N
$$

where $m_k = dk - k(k+1)/2$ counts the free parameters for $k$ signal
components in a $d$-channel dataset of $N$ samples.

```python
from neurojax.analysis.dimensionality import PPCA

# Consensus of AIC and BIC
n_components = PPCA.estimate_dimensionality(data_asr, method="consensus")
print(f"Estimated intrinsic dimensionality: {n_components}")

# Inspect individual criteria
scores = PPCA.get_consensus_evidence(data_asr)
# scores["aic"], scores["bic"] are arrays indexed by k
```

#### 3.3.2 Running PICA

```python
from neurojax.analysis.ica import PICA

pica = PICA(n_components=n_components)
pica = pica.fit(data_asr)

# Independent component time courses: (n_components, n_times)
ic_timecourses = pica.components_
# Mixing matrix (spatial maps): (n_channels, n_components)
ic_maps = pica.mixing_
```

#### 3.3.3 Identifying artifact components

PICA offers several strategies for flagging artifact ICs:

```python
# Find the IC whose spatial map best matches an EOG template
eog_template = raw.copy().pick("eog").get_data().mean(axis=0)
eog_ic = pica.find_temporally_correlated_component(
    jnp.array(eog_template)
)
print(f"EOG artifact is IC #{eog_ic}")

# Find the IC with maximum power near 50 Hz (line noise)
line_ic, power_ratios = pica.find_spectral_peak_component(
    float(sfreq), target_freq=50.0, f_width=2.0
)
print(f"Line-noise artifact is IC #{line_ic}")
```

#### 3.3.4 Removing artifact ICs and back-projecting

```python
# Zero out artifact components and reconstruct
artifact_ics = jnp.array([int(eog_ic), int(line_ic)])
clean_sources = ic_timecourses.at[artifact_ics].set(0.0)
data_clean = ic_maps @ clean_sources + pica.mean_
```

#### 3.3.5 Alternative: FastICA as an Equinox module

If you already know the number of components (or have estimated it with PPCA),
you can use the `FastICA` Equinox module directly, which integrates cleanly
with `eqx.filter_jit`:

```python
import equinox as eqx
from neurojax.preprocessing.ica import FastICA

model = FastICA(n_components=n_components)

@eqx.filter_jit
def fit_ica(m, d):
    return m.fit(d)

model = fit_ica(model, data_asr)
# model.components_: (n_components, n_times)
# model.mixing_:     (n_channels, n_components)
```

---

## 4. Source Reconstruction

NeuroJAX provides two families of source-reconstruction algorithms, all
`jax.jit`-compiled:

- **Beamformers** (`neurojax.source.beamformer`): LCMV, Vector LCMV, SAM, DICS, eigenspace LCMV
- **Minimum-norm estimates** (`neurojax.source.minimum_norm`): MNE, wMNE, dSPM, sLORETA, eLORETA

### 4.1 Setting up the forward model

Source reconstruction requires a forward model (leadfield matrix) mapping
source currents to sensor measurements.  This is typically computed with MNE:

```python
import mne

# Set up a single-sphere head model (fast, good for sensor-level demos)
sphere = mne.make_sphere_model(r0="auto", head_radius="auto",
                               info=raw.info)

# Volumetric source space at 10 mm resolution
src = mne.setup_volume_source_space(
    subject=None, pos=10.0, sphere=sphere, bem=sphere
)

# Compute forward solution
fwd = mne.make_forward_solution(
    raw.info, trans=None, src=src, bem=sphere,
    eeg=False, meg=True, verbose=False,
)
print(f"Forward model: {fwd['nsource']} source locations")

# Extract the gain (leadfield) matrix as a JAX array
G = jnp.array(fwd["sol"]["data"])   # (n_sensors, n_sources * n_orient)
```

```{note}
For a cortically-constrained surface source space with a BEM model,
replace the sphere model with a FreeSurfer-derived BEM and a surface
source space.  The NeuroJAX inverse functions accept any gain matrix
shape.
```

### 4.2 Data and noise covariance

Both beamformers and MNE-family inverses require a data covariance matrix.
MNE variants additionally need a noise covariance estimate:

```python
# Data covariance from the cleaned, filtered continuous data
X = data_clean   # (n_channels, n_times)
X_centered = X - jnp.mean(X, axis=1, keepdims=True)
n_samples = X.shape[1]
data_cov = X_centered @ X_centered.T / n_samples

# Noise covariance from an empty-room recording or pre-stimulus baseline
# Here we simulate with a scaled identity (diagonal noise model)
noise_cov = 1e-26 * jnp.eye(X.shape[0])
```

### 4.3 LCMV beamformer

The scalar LCMV beamformer {cite:p}`vanveen1997localization` computes spatial
filter weights that pass signal from a target location while minimizing total
output power:

$$
\mathbf{w}_i
= \frac{\mathbf{C}^{-1}\,\mathbf{g}_i}
       {\mathbf{g}_i^\top\,\mathbf{C}^{-1}\,\mathbf{g}_i}
$$

where $\mathbf{C}$ is the data covariance and $\mathbf{g}_i$ the leadfield
column for source $i$.

```python
from neurojax.source.beamformer import (
    make_lcmv_filter,
    apply_lcmv,
    neural_activity_index,
    unit_noise_gain,
    lcmv_power_map,
)

# Compute LCMV weights (jit-compiled)
W = make_lcmv_filter(data_cov, G, reg=0.05)

# Source-space power map
power = lcmv_power_map(data_cov, G, reg=0.05)

# Neural Activity Index normalization removes depth bias
nai = neural_activity_index(W, noise_cov)
power_nai = power / (nai ** 2)

# Apply beamformer to get source time courses
source_tc = apply_lcmv(data_clean, W)   # (n_sources, n_times)
```

### 4.4 SAM pseudo-Z mapping

Synthetic Aperture Magnetometry (SAM) compares beamformer power between
active and control windows {cite:p}`robinson1999functional`:

$$
Z_i = \frac{P_i^{\text{active}} - P_i^{\text{control}}}
           {P_i^{\text{control}}}
$$

```python
from neurojax.source.beamformer import sam_pseudo_z

# Define active/control windows (e.g., stimulus vs baseline)
active_seg = data_clean[:, int(1.0 * sfreq):int(2.0 * sfreq)]
control_seg = data_clean[:, int(-1.0 * sfreq):]

# Covariance per condition
cov_act = (active_seg @ active_seg.T) / active_seg.shape[1]
cov_ctl = (control_seg @ control_seg.T) / control_seg.shape[1]

pseudo_z = sam_pseudo_z(cov_act, cov_ctl, G, reg=0.05)
```

### 4.5 DICS (frequency-domain beamformer)

DICS {cite:p}`gross2001dynamic` operates on the cross-spectral density (CSD)
at a frequency of interest --- ideal for localizing oscillatory sources:

```python
from neurojax.source.beamformer import dics_power, dics_coherence
from neurojax.analysis.multitaper import multitaper_csd

# Compute CSD at the alpha peak (10 Hz) using multitaper
csd_10hz = multitaper_csd(data_clean, sfreq, fmin=9.0, fmax=11.0)

# Source power at 10 Hz
alpha_power = dics_power(csd_10hz, G, reg=0.05)

# Coherence with a seed source (e.g., primary visual cortex)
seed_idx = 42   # index of the seed in the source space
coh_map = dics_coherence(csd_10hz, G, seed_idx=seed_idx, reg=0.05)
```

### 4.6 Minimum-norm estimates (MNE / dSPM / sLORETA / eLORETA)

The MNE family solves the underdetermined inverse problem with a quadratic
source prior.  All variants share the structure:

$$
\hat{\mathbf{x}} = \mathbf{R}\,\mathbf{G}^\top
\bigl(\mathbf{G}\,\mathbf{R}\,\mathbf{G}^\top + \lambda\,\mathbf{C}\bigr)^{-1}
\mathbf{y}
$$

and differ in the normalization applied to the resulting source estimates.
NeuroJAX computes all variants with a single `jax.jit`-compiled function and
adds depth weighting automatically:

```python
from neurojax.source.minimum_norm import (
    make_inverse_operator,
    apply_inverse,
    resolution_matrix,
    resolution_metrics,
    compare_inverse_methods,
)

# Compute inverse operator (dSPM)
W_inv, noise_norm = make_inverse_operator(
    G, noise_cov, depth=0.8, reg=0.1, method="dSPM"
)

# Apply to sensor data -> (n_sources, n_times) z-scored map
source_dspm = apply_inverse(W_inv, noise_norm, data_clean, method="dSPM")
```

#### 4.6.1 Comparing inverse methods with resolution metrics

NeuroJAX can compute the full resolution matrix $\mathbf{M} = \mathbf{W G}$
and derive Point Spread Functions (PSF), Cross-Talk Functions (CTF), and
summary quality metrics for each inverse variant:

```python
# Compare all four methods
comparison = compare_inverse_methods(
    G, noise_cov,
    methods=("MNE", "dSPM", "sLORETA", "eLORETA"),
    depth=0.8, reg=0.1,
)

for method, metrics in comparison.items():
    amp = float(jnp.mean(metrics["relative_amplitude"]))
    spread = float(jnp.mean(metrics["spatial_spread"]))
    print(f"{method:10s}  amplitude recovery = {amp:.3f}  "
          f"spatial spread = {spread:.4f}")
```

---

## 5. Sensor-Space Analysis

### 5.1 Time-frequency decomposition (Morlet wavelets)

Morlet wavelets provide time-frequency representations with adaptive
resolution: narrow windows at high frequencies, wide at low.  NeuroJAX
implements the convolution via `jax.lax.conv_general_dilated` for GPU
efficiency:

```python
from neurojax.analysis.timefreq import morlet_transform

# Define frequencies of interest (must be a tuple for JIT shape tracing)
freqs = tuple(jnp.logspace(jnp.log10(4), jnp.log10(80), 30).tolist())

# Compute TFR for the first 10 channels
tfr = morlet_transform(
    data_clean[:10],
    sfreq=float(sfreq),
    freqs=freqs,
    n_cycles_min=3.0,
    n_cycles_max=7.0,
)
# tfr shape: (10, 30, n_times) — complex-valued
power = jnp.abs(tfr) ** 2
```

### 5.2 Spectral parameterization (SpecParam / FOOOF)

SpecParam decomposes the power spectrum into aperiodic (1/f) and periodic
(oscillatory) components.  The model for log-power at frequency $f$ is:

$$
P(f) = \underbrace{b - \log_{10}(k + f^{\chi})}_{\text{aperiodic}}
     + \underbrace{\sum_{n=1}^{N} a_n \exp\!\Bigl(
         -\frac{(f - c_n)^2}{2\,w_n^2}\Bigr)}_{\text{periodic peaks}}
$$

NeuroJAX fits this model using Equinox + Optax gradient descent, making
the entire fit differentiable:

```python
import scipy.signal
from neurojax.analysis.spectral import SpecParam

# Compute PSD via Welch
f_welch, Pxx = scipy.signal.welch(
    data_clean, fs=float(sfreq), nperseg=int(sfreq * 2)
)
f_welch = jnp.array(f_welch)
Pxx = jnp.array(Pxx)

# Fit channel 0 with up to 3 oscillatory peaks
sp = SpecParam.fit(f_welch, jnp.log10(Pxx[0]), n_peaks=3, steps=1000, lr=0.05)

offset, knee, exponent = sp.aperiodic_params
print(f"Aperiodic: offset={offset:.2f}, knee={knee:.2f}, "
      f"exponent={exponent:.2f}")
print(f"Peak params:\n{sp.peak_params}")

# Evaluate the fitted model
model_fit = sp.get_model(f_welch)
```

### 5.3 Multitaper spectral estimation

For robust spectral estimates with reduced leakage, NeuroJAX provides
DPSS (Slepian) multitaper methods following {cite:p}`thomson1982spectrum`:

```python
from neurojax.analysis.multitaper import dpss_tapers, multitaper_psd

# Compute DPSS tapers
tapers = dpss_tapers(n_samples=int(2 * sfreq), bandwidth=4.0)

# Multitaper PSD
freqs_mt, psd_mt = multitaper_psd(data_clean, sfreq, bandwidth=4.0)
```

---

## 6. Statistical Inference with the GLM

NeuroJAX provides a GPU-accelerated General Linear Model with
`vmap`-parallelized permutation testing.  The model is implemented as an
Equinox module using Lineax for numerically stable QR-based solves and
Distrax for likelihood-based model comparison.

### 6.1 Setting up the design matrix

```python
import numpy as np
import mne

# Create epochs from events
events = mne.find_events(raw, stim_channel="STI101")
event_id = {"faces": 1, "scrambled": 2}
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.6,
                    baseline=(None, 0), preload=True)

# Build a design matrix (n_times x n_regressors)
n_times = epochs.get_data().shape[2]
times = epochs.times

# Simple boxcar regressors per condition
design = np.zeros((len(epochs), 2))
design[epochs.events[:, 2] == 1, 0] = 1.0   # faces
design[epochs.events[:, 2] == 2, 1] = 1.0   # scrambled
design_jax = jnp.array(design)
```

### 6.2 Fitting the model

```python
from neurojax.glm import GeneralLinearModel, run_permutation_test

# Average across time for a sensor-level amplitude analysis
# data shape: (n_epochs, n_sensors)
epoch_data = jnp.array(epochs.get_data().mean(axis=2))

glm = GeneralLinearModel(design_jax, epoch_data)
fitted = glm.fit()

# Contrast: faces > scrambled
contrast = jnp.array([1.0, -1.0])
t_stats = fitted.get_t_stats(contrast)
print(f"Max t-stat: {float(jnp.max(t_stats)):.2f}")
```

### 6.3 Permutation testing with max-t correction

The key acceleration: NeuroJAX `vmap`s over thousands of permutations
simultaneously.  Each permutation shuffles the data rows to break the
design-outcome relationship, preserving the spatial correlation structure:

```python
import jax

key = jax.random.PRNGKey(42)

t_stats, p_values = run_permutation_test(
    glm, contrast, key, n_perms=5000
)

# Significant sensors (family-wise error corrected)
sig_mask = p_values < 0.05
n_sig = int(jnp.sum(sig_mask))
print(f"Significant sensors (p < 0.05, max-t corrected): {n_sig}")
```

### 6.4 Model comparison via log-likelihood

```python
log_lik = fitted.log_likelihood()
print(f"Model log-likelihood: {float(log_lik):.1f}")
```

---

## 7. Functional Connectivity

NeuroJAX includes graph-theoretic network analysis tools in
`neurojax.analysis.funcnet`, covering nonlinear coupling measures and
standard small-world / centrality metrics.

### 7.1 Mutual information connectivity matrix

```python
from neurojax.analysis.funcnet import (
    mutual_information_matrix,
    threshold_matrix,
    degree,
    clustering_coefficient,
    global_clustering,
    characteristic_path_length,
    small_world_index,
    betweenness_centrality,
)

# Compute MI matrix over source-reconstructed time series
# source_tc: (n_sources, n_times) -> transpose to (n_times, n_sources)
mi_matrix = mutual_information_matrix(source_tc[:50].T, n_bins=32)
```

### 7.2 Network measures

```python
# Threshold to 10 % density
A = threshold_matrix(mi_matrix, density=0.1)

# Node degree
deg = degree(A)

# Clustering and path length
C = global_clustering(A)
L = characteristic_path_length(A)
sigma = small_world_index(A, n_random=20)

print(f"Clustering: {C:.3f}, Path length: {L:.2f}, "
      f"Small-world index: {sigma:.2f}")
```

### 7.3 Lagged cross-correlation

```python
from neurojax.analysis.funcnet import optimal_lag

# Find the lag at which two source signals are maximally correlated
lag, corr = optimal_lag(source_tc[0], source_tc[10], max_lag=50)
print(f"Optimal lag: {lag} samples ({lag / sfreq * 1000:.1f} ms), "
      f"r = {corr:.3f}")
```

---

## 8. Reporting

NeuroJAX can generate self-contained HTML reports with embedded figures,
following the style of FSL's browser-based reports:

```python
from neurojax.reporting.html import (
    HTMLReport,
    plot_quality_metrics,
    plot_dimensionality,
    plot_spectral_analysis,
)

report = HTMLReport(title="Sub-01 MEG Analysis")

# Section 1: Data quality
quality_figs = plot_quality_metrics(epochs)
report.add_section(
    "Data Quality",
    "Global Field Power and sensor noise topography.",
    quality_figs,
)

# Section 2: Dimensionality
dim_figs = plot_dimensionality(epoch_data)
report.add_section(
    "Dimensionality",
    f"Intrinsic dimensionality estimated at {n_components} (AIC/BIC consensus).",
    dim_figs,
)

# Section 3: Spectral decomposition
spectral_figs = plot_spectral_analysis(f_welch, jnp.log10(Pxx.mean(axis=0)))
report.add_section(
    "Spectral Parameterization",
    "Aperiodic + periodic decomposition (SpecParam).",
    spectral_figs,
)

report.save("sub-01_meg_report.html")
```

---

## 9. Putting It All Together

Below is a condensed script that chains every step from this tutorial into a
single end-to-end pipeline.  Replace the file path with your own data.

```python
"""End-to-end MEG analysis with NeuroJAX."""

import jax
import jax.numpy as jnp
import scipy.signal
import mne

from neurojax.io.loader import load_data
from neurojax.io.bridge import mne_to_jax
from neurojax.preprocessing.filter import filter_data
from neurojax.preprocessing.asr import calibrate_asr, apply_asr
from neurojax.analysis.ica import PICA
from neurojax.analysis.dimensionality import PPCA
from neurojax.analysis.spectral import SpecParam
from neurojax.analysis.timefreq import morlet_transform
from neurojax.source.beamformer import make_lcmv_filter, apply_lcmv, lcmv_power_map
from neurojax.source.minimum_norm import make_inverse_operator, apply_inverse
from neurojax.glm import GeneralLinearModel, run_permutation_test
from neurojax.reporting.html import HTMLReport

# --- 1. Load ---------------------------------------------------------------
raw = load_data("sub-01_task-faces_meg.fif", preload=True)
data, sfreq = mne_to_jax(raw)

# --- 2. Filter --------------------------------------------------------------
nyq = 0.5 * float(sfreq)
b, a = scipy.signal.iirfilter(4, [1.0 / nyq, 100.0 / nyq], btype="band")
data = filter_data(data, jnp.array(b), jnp.array(a))

# --- 3. ASR ------------------------------------------------------------------
asr_state = calibrate_asr(data[:, :int(30 * sfreq)], cutoff=5.0)
data = apply_asr(data, asr_state,
                 window_size=int(0.5 * sfreq),
                 step_size=int(0.25 * sfreq))

# --- 4. ICA artifact removal ------------------------------------------------
n_comp = PPCA.estimate_dimensionality(data, method="consensus")
pica = PICA(n_components=n_comp).fit(data)
eog_ic = pica.find_spectral_peak_component(float(sfreq), 1.5, f_width=1.0)[0]
data = pica.mixing_ @ pica.components_.at[int(eog_ic)].set(0.0) + pica.mean_

# --- 5. Source reconstruction ------------------------------------------------
sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=raw.info)
src = mne.setup_volume_source_space(subject=None, pos=10.0,
                                     sphere=sphere, bem=sphere)
fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere,
                                 eeg=False, meg=True, verbose=False)
G = jnp.array(fwd["sol"]["data"])
data_cov = (data @ data.T) / data.shape[1]
noise_cov = 1e-26 * jnp.eye(data.shape[0])

# Beamformer
power = lcmv_power_map(data_cov, G, reg=0.05)

# dSPM
W_inv, nn = make_inverse_operator(G, noise_cov, depth=0.8, reg=0.1,
                                   method="dSPM")
source = apply_inverse(W_inv, nn, data, method="dSPM")

# --- 6. Time-frequency -------------------------------------------------------
freqs = tuple(jnp.logspace(jnp.log10(4), jnp.log10(80), 30).tolist())
tfr = morlet_transform(data[:5], float(sfreq), freqs)

# --- 7. Spectral parameterization -------------------------------------------
f, Pxx = scipy.signal.welch(data, fs=float(sfreq), nperseg=int(sfreq * 2))
sp = SpecParam.fit(jnp.array(f), jnp.log10(jnp.array(Pxx[0])), n_peaks=3)

# --- 8. GLM + permutation test ----------------------------------------------
events = mne.find_events(raw, stim_channel="STI101")
epochs = mne.Epochs(raw, events, {"faces": 1, "scrambled": 2},
                    tmin=-0.2, tmax=0.6, baseline=(None, 0), preload=True)
epoch_data = jnp.array(epochs.get_data().mean(axis=2))
design = jnp.zeros((len(epochs), 2))
design = design.at[epochs.events[:, 2] == 1, 0].set(1.0)
design = design.at[epochs.events[:, 2] == 2, 1].set(1.0)
glm = GeneralLinearModel(design, epoch_data)
t_stats, p_vals = run_permutation_test(glm, jnp.array([1.0, -1.0]),
                                        jax.random.PRNGKey(0), n_perms=5000)

# --- 9. Report ---------------------------------------------------------------
report = HTMLReport(title="Sub-01 MEG Pipeline")
report.save("sub-01_pipeline_report.html")
print("Pipeline complete.")
```

---

## 10. CLI Usage

NeuroJAX also exposes the core preprocessing steps as a command-line tool:

```bash
# Bandpass filter + ICA with automatic dimensionality estimation
neurojax process sub-01_task-faces_meg.fif \
    --highpass 1.0 --lowpass 100.0 \
    --ica \
    --output sub-01_cleaned.fif

# Add spectral parameterization
neurojax process sub-01_task-faces_meg.fif \
    --spectral --wavelet

# Force a specific number of ICA components
neurojax process sub-01_task-faces_meg.fif \
    --ica --n-components 20 \
    --device gpu
```

---

## References

```{bibliography}
:filter: docname in docnames
```
