# Source Imaging: LCMV Beamforming for MEG

This tutorial covers the MEG inverse problem using beamforming techniques
implemented in NeuroJAX. Starting from a synthetic forward model, you will
learn how to construct LCMV spatial filters, estimate source activity, and
build power maps -- all with JIT-compiled, GPU-accelerable JAX code. We also
cover CHAMPAGNE sparse Bayesian learning for handling correlated sources and
compare these approaches to minimum-norm alternatives.

```{contents}
:depth: 3
:local:
```

## Prerequisites

```bash
uv sync --extra doc
```

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")  # or "gpu" if available
print(jax.devices())
```

---

## 1. The MEG Inverse Problem

MEG sensors measure magnetic fields produced by post-synaptic currents in
cortical pyramidal neurons. The **forward model** describes how a set of
current dipole sources at known cortical locations project to the sensor array:

$$
\mathbf{Y} = \mathbf{G} \mathbf{X} + \boldsymbol{\varepsilon}
$$

where:
- $\mathbf{Y} \in \mathbb{R}^{M \times T}$ is the sensor data ($M$ sensors, $T$ time samples),
- $\mathbf{G} \in \mathbb{R}^{M \times N}$ is the **leadfield** (gain) matrix ($N$ source locations),
- $\mathbf{X} \in \mathbb{R}^{N \times T}$ is the unknown source time-course matrix,
- $\boldsymbol{\varepsilon}$ is sensor noise.

The inverse problem -- recovering $\mathbf{X}$ from $\mathbf{Y}$ -- is
ill-posed because $N \gg M$ in realistic cortical models. Different source
imaging methods impose different constraints to regularize the solution.
Beamformers take a **spatial filtering** approach: they design a separate
linear filter for each source location that passes signal from that location
while suppressing interference from all other sources.

---

## 2. Setting Up a Synthetic Forward Model

Before working with real MEG data, it helps to build a small synthetic forward
model where we know the ground truth. This lets us verify that the beamformer
correctly recovers the active source.

```python
# Reproducible random state
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

# Dimensions
n_sensors = 10   # MEG channels (gradiometers/magnetometers)
n_sources = 5    # Candidate source locations on cortex
n_time = 200     # Time samples

# Random leadfield matrix (in practice, computed from BEM + source space)
gain = jax.random.normal(k1, (n_sensors, n_sources))

# Ground truth: only source index 2 is active
sources_true = jnp.zeros((n_sources, n_time))
sources_true = sources_true.at[2].set(jax.random.normal(k2, (n_time,)))

# Simulated sensor data with additive noise
noise = 0.1 * jax.random.normal(k3, (n_sensors, n_time))
data = gain @ sources_true + noise
```

The **data covariance matrix** $\mathbf{C} = \frac{1}{T} \mathbf{Y}\mathbf{Y}^\top$
captures the second-order statistics of the sensor recordings. It is the
central quantity for beamformer design:

```python
cov = (data @ data.T) / n_time
print(f"Data covariance shape: {cov.shape}")
# Data covariance shape: (10, 10)
```

```{admonition} Practical tip
:class: tip
In real MEG pipelines, compute the data covariance from a sufficiently long
segment of task or resting-state data *after* preprocessing (filtering,
artifact rejection, ICA). Poor covariance estimates degrade beamformer
performance. A rule of thumb is to use at least $3M$ time samples, where $M$
is the number of sensors.
```

---

## 3. LCMV Beamformer Weight Computation

The Linearly Constrained Minimum Variance (LCMV) beamformer (Van Veen et al.,
1997) designs a spatial filter $\mathbf{w}_i$ for each source $i$ that:

1. **Passes signal** from source $i$ with unit gain: $\mathbf{w}_i^\top \mathbf{g}_i = 1$
2. **Minimizes output variance** (i.e., suppresses interference): $\min \mathbf{w}_i^\top \mathbf{C} \mathbf{w}_i$

### The Mathematics

Using the method of Lagrange multipliers, the solution for the full weight
matrix is:

$$
\mathbf{W} = \left(\mathbf{G}^\top \mathbf{C}^{-1} \mathbf{G}\right)^{-1} \mathbf{G}^\top \mathbf{C}^{-1}
$$

Each row $\mathbf{w}_i$ of $\mathbf{W}$ is the spatial filter for source $i$.
The **unit-gain constraint** is satisfied by construction:

$$
\mathbf{W} \mathbf{G} = \left(\mathbf{G}^\top \mathbf{C}^{-1} \mathbf{G}\right)^{-1} \mathbf{G}^\top \mathbf{C}^{-1} \mathbf{G} = \mathbf{I}_{N \times N}
$$

This means that when the filter for source $i$ is applied to data generated
purely by source $j$, the output is zero for $i \neq j$ and equal to the
source amplitude for $i = j$.

### Tikhonov Regularization

In practice, the data covariance may be rank-deficient (e.g., after
dimensionality reduction or with short data segments). The `make_lcmv_filter()`
function adds Tikhonov regularization:

$$
\mathbf{C}_{\text{reg}} = \mathbf{C} + \alpha \cdot \frac{\text{tr}(\mathbf{C})}{M} \cdot \mathbf{I}
$$

where $\alpha$ is the `reg` parameter (default 0.05). This ensures the
covariance is invertible and improves numerical stability.

### NeuroJAX Implementation

```python
from neurojax.source.beamformer import make_lcmv_filter

# Compute LCMV spatial filter weights
weights = make_lcmv_filter(cov, gain, reg=0.05)
print(f"Weight matrix shape: {weights.shape}")
# Weight matrix shape: (5, 10) -- one spatial filter per source
```

```{admonition} Function signature
:class: note
`make_lcmv_filter(cov, gain, reg=0.05) -> jnp.ndarray`

- `cov`: `(n_sensors, n_sensors)` data covariance matrix
- `gain`: `(n_sensors, n_sources)` leadfield matrix
- `reg`: `float` -- Tikhonov regularization parameter (default 0.05)
- **Returns**: `(n_sources, n_sensors)` spatial filter weight matrix

The function is decorated with `@jax.jit`, so the first call triggers XLA
compilation and subsequent calls with arrays of the same shape are near-instant.
```

### Verifying the Unit-Gain Constraint

We can verify that $\mathbf{W} \mathbf{G} \approx \mathbf{I}$:

```python
product = weights @ gain
identity = jnp.eye(n_sources)

# Should be very close to the identity matrix
print("W @ G (should be ~I):")
print(jnp.round(product, decimals=4))
np.testing.assert_allclose(product, identity, atol=1e-4)
```

---

## 4. Applying the Spatial Filter

Once the weights are computed, reconstructing source activity is a simple
matrix multiplication:

$$
\hat{\mathbf{X}} = \mathbf{W} \mathbf{Y}
$$

```python
from neurojax.source.beamformer import apply_lcmv

# Reconstruct source time courses
sources_hat = apply_lcmv(data, weights)
print(f"Estimated sources shape: {sources_hat.shape}")
# Estimated sources shape: (5, 200)
```

```{admonition} Function signature
:class: note
`apply_lcmv(data, weights) -> jnp.ndarray`

- `data`: `(n_sensors, n_time)` sensor recordings
- `weights`: `(n_sources, n_sensors)` spatial filter from `make_lcmv_filter()`
- **Returns**: `(n_sources, n_time)` estimated source time courses
```

### Checking Source Recovery

We know that source 2 was the only active source. Let us verify that the
beamformer output power is highest for that source:

```python
# Power = mean(amplitude^2) for each source
power = jnp.mean(sources_hat ** 2, axis=1)
peak_source = int(jnp.argmax(power))

print(f"Source power: {jnp.round(power, 4)}")
print(f"Peak source index: {peak_source}")
assert peak_source == 2, "Beamformer should peak at the active source"
```

---

## 5. Power Mapping and the Neural Activity Index

For source localization, we often want a **power map** -- the estimated signal
power at each candidate source location -- rather than full time courses.

### Direct Power Computation

The `lcmv_power_map()` function computes source power directly from the
covariance, without materializing the full time-course matrix:

$$
P_i = \mathbf{w}_i^\top \mathbf{C} \mathbf{w}_i
$$

```python
from neurojax.source.beamformer import lcmv_power_map

power_map = lcmv_power_map(cov, gain, reg=0.05)
print(f"Power map shape: {power_map.shape}")
# Power map shape: (5,)
print(f"Power values: {jnp.round(power_map, 4)}")
```

### Neural Activity Index (NAI)

Raw beamformer power is biased toward deeper sources because the leadfield
magnitude decreases with depth. The **Neural Activity Index** normalizes each
source by the noise power projected through its spatial filter, removing this
depth bias:

$$
\text{NAI}_i = \frac{\mathbf{w}_i^\top \mathbf{C}_{\text{data}} \mathbf{w}_i}{\mathbf{w}_i^\top \mathbf{C}_{\text{noise}} \mathbf{w}_i}
$$

```python
from neurojax.source.beamformer import neural_activity_index, unit_noise_gain

# Assume a simple noise covariance (identity scaled by noise variance)
noise_cov = 0.01 * jnp.eye(n_sensors)

# NAI normalization factors
nai_factors = neural_activity_index(weights, noise_cov)
print(f"NAI factors shape: {nai_factors.shape}")
# NAI factors shape: (5,)

# Alternatively, get unit-noise-gain normalized weights directly
weights_normalized = unit_noise_gain(weights, noise_cov)
print(f"Normalized weights shape: {weights_normalized.shape}")
```

```{admonition} Function signatures
:class: note
`neural_activity_index(weights, noise_cov) -> jnp.ndarray`
- Returns `(n_sources,)` normalization factors.

`unit_noise_gain(weights, noise_cov) -> jnp.ndarray`
- Returns `(n_sources, n_sensors)` rescaled weights where the noise power
  through each filter equals 1.
```

```{admonition} When to use NAI
:class: tip
Always apply NAI normalization when comparing beamformer power across sources
at different depths. Without it, superficial sources will appear systematically
stronger than deep sources, even when they have the same neural activity level.
```

---

## 6. Identifying Active Sources from Power Maps

With the power map in hand, we can localize neural activity by finding sources
whose power significantly exceeds the noise floor:

```python
# Z-score the power map
mean_power = jnp.mean(power_map)
std_power = jnp.std(power_map)
z_scores = (power_map - mean_power) / std_power

# Threshold at z > 2
active_mask = z_scores > 2.0
active_indices = jnp.where(active_mask, size=n_sources)[0]

print(f"Z-scores: {jnp.round(z_scores, 2)}")
print(f"Active source indices (z > 2): {active_indices}")
```

In a realistic analysis with thousands of source locations, you would project
this power map onto the cortical surface mesh for visualization, typically
using MNE-Python's `stc.plot()` after constructing a `SourceEstimate` object.

---

## 7. CHAMPAGNE: Sparse Bayesian Learning

The LCMV beamformer assumes that sources are uncorrelated. When sources are
correlated (e.g., bilateral auditory responses), LCMV performance degrades
because the spatial filter may cancel the correlated component. **CHAMPAGNE**
(Wipf et al., 2008) addresses this using a sparse Bayesian learning (SBL)
framework.

### The Generative Model

CHAMPAGNE models the data as:

$$
\mathbf{Y} = \mathbf{G}\mathbf{X} + \boldsymbol{\varepsilon}, \quad
\mathbf{X} \sim \mathcal{N}(0, \boldsymbol{\Gamma}), \quad
\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \boldsymbol{\Sigma}_\text{noise})
$$

where $\boldsymbol{\Gamma} = \text{diag}(\gamma_1, \ldots, \gamma_N)$ is a
diagonal source power prior. The algorithm iteratively updates $\gamma_i$ to
maximize the model evidence (marginal likelihood), driving inactive sources'
$\gamma_i$ toward zero.

### Update Rule (Convex Bounding)

The model covariance is:

$$
\boldsymbol{\Sigma}_y = \mathbf{G}\boldsymbol{\Gamma}\mathbf{G}^\top + \boldsymbol{\Sigma}_\text{noise}
$$

At each iteration, the source powers are updated:

$$
\gamma_i^{\text{new}} = \gamma_i \sqrt{\frac{[\mathbf{G}^\top \boldsymbol{\Sigma}_y^{-1} \mathbf{C}_\text{data} \boldsymbol{\Sigma}_y^{-1} \mathbf{G}]_{ii}}{[\mathbf{G}^\top \boldsymbol{\Sigma}_y^{-1} \mathbf{G}]_{ii}}}
$$

This is implemented using `jax.lax.while_loop` for efficient convergence
checking without Python-level control flow overhead.

### Using CHAMPAGNE

```python
from neurojax.source.champagne import champagne_solver

# Run CHAMPAGNE with 20 EM iterations
gamma, weights_champ = champagne_solver(cov, gain, max_iter=20)

print(f"Source powers (gamma): {jnp.round(gamma, 4)}")
print(f"CHAMPAGNE weights shape: {weights_champ.shape}")
# weights_champ shape: (5, 10)

# The active source should have the highest gamma
peak = int(jnp.argmax(gamma))
print(f"Peak source: {peak}")
assert peak == 2
```

```{admonition} Function signature
:class: note
`champagne_solver(cov, gain, noise_cov=None, max_iter=20, tol=1e-4) -> (gamma, weights)`

- `cov`: `(n_sensors, n_sensors)` data covariance
- `gain`: `(n_sensors, n_sources)` leadfield
- `noise_cov`: `(n_sensors, n_sensors)` noise covariance (default: identity)
- `max_iter`: maximum EM iterations
- `tol`: convergence tolerance on $\max|\gamma^{(t)} - \gamma^{(t-1)}|$
- **Returns**: `gamma` `(n_sources,)` source power estimates, `weights` `(n_sources, n_sensors)` spatial filter
```

### Providing a Noise Covariance

When you have a pre-stimulus baseline or empty-room recording, compute the
noise covariance and pass it explicitly for better source estimates:

```python
noise_cov = 0.01 * jnp.eye(n_sensors)
gamma, weights_champ = champagne_solver(
    cov, gain, noise_cov=noise_cov, max_iter=20
)
assert jnp.all(jnp.isfinite(gamma))
assert jnp.all(jnp.isfinite(weights_champ))
```

### Imaginary Coherence for Connectivity

After source reconstruction, you can assess functional connectivity using
**imaginary coherence**, which is robust to volume conduction artifacts
(Nolte et al., 2004). Volume conduction effects are instantaneous (zero-lag),
so they contribute only to the real part of coherence. Taking the imaginary
part eliminates these spurious connections:

$$
\text{iCoh}(i, j) = \frac{\text{Im}\left(\mathbf{S}_{ij}\right)}{\sqrt{\mathbf{S}_{ii} \cdot \mathbf{S}_{jj}}}
$$

```python
from neurojax.source.champagne import imaginary_coherence

# Reconstruct source time courses (using complex analytic signal for coherence)
k4, k5 = jax.random.split(jax.random.PRNGKey(99))
real_part = jax.random.normal(k4, (n_sources, n_time))
imag_part = jax.random.normal(k5, (n_sources, n_time))
source_data_complex = real_part + 1j * imag_part

# Compute imaginary coherence with source 0 as reference
icoh = imaginary_coherence(source_data_complex, ref_idx=0)
print(f"Imaginary coherence shape: {icoh.shape}")
# (5,)

# Self-coherence (imaginary part) should be ~0
print(f"Self-coherence (source 0): {float(icoh[0]):.6f}")
np.testing.assert_allclose(float(icoh[0]), 0.0, atol=1e-5)

# All values should be bounded in [-1, 1]
assert jnp.all(jnp.abs(icoh) <= 1.0 + 1e-6)
```

```{admonition} Function signature
:class: note
`imaginary_coherence(source_data, ref_idx) -> jnp.ndarray`

- `source_data`: `(n_sources, n_time)` complex-valued analytic signal
- `ref_idx`: `int` -- index of the reference source
- **Returns**: `(n_sources,)` imaginary coherence values in $[-1, 1]$
```

---

## 8. ADMM Inverse Solver (Sparse L1)

NeuroJAX also provides an ADMM-based inverse solver that imposes L1 (sparsity)
constraints on the source estimates. This is useful when you expect focal
activations:

$$
\min_{\mathbf{X}} \frac{1}{2}\|\mathbf{Y} - \mathbf{G}\mathbf{X}\|_2^2 + \lambda \|\mathbf{X}\|_1
$$

```python
from neurojax.source.inverse_scico import solve_inverse_admm, InverseResult

result = solve_inverse_admm(data, gain, lambda_reg=0.1, maxiter=50)
print(f"Estimated sources: {result.sources.shape}")
print(f"Residuals: {result.residuals.shape}")

# Residuals should satisfy: residuals = Y - G @ X_hat
expected_residuals = data - gain @ result.sources
np.testing.assert_allclose(result.residuals, expected_residuals, atol=1e-5)
```

```{admonition} LCMV vs. ADMM L1
:class: tip
- **LCMV** is a spatial filter approach -- fast, works well for single dipole
  localization, but assumes uncorrelated sources.
- **ADMM L1** is a distributed source estimate with sparsity prior -- better
  for focal sources, but slower and requires tuning $\lambda$.
- **CHAMPAGNE** is a Bayesian approach -- handles correlated sources and
  automatically determines sparsity, but is the most computationally expensive.
```

---

## 9. Comparison with Minimum-Norm Approaches

NeuroJAX also implements the minimum-norm family of inverse methods
(`source.minimum_norm`), including MNE, dSPM, sLORETA, and eLORETA. These
compute a linear inverse operator $\mathbf{K}$ that maps sensor data to
source estimates:

$$
\hat{\mathbf{X}} = \mathbf{K}\mathbf{Y}
$$

where $\mathbf{K}$ depends on the regularization approach. A key diagnostic
is the **resolution matrix** $\mathbf{R} = \mathbf{K}\mathbf{G}$, which
describes how well each source can be resolved:

```python
from neurojax.source.inverse_scico import compute_resolution_matrix

# Pseudoinverse as a simple inverse operator
K = jnp.linalg.pinv(gain)   # (n_sources, n_sensors)
R = compute_resolution_matrix(gain, K)
print(f"Resolution matrix shape: {R.shape}")
# (5, 5)

# When n_sensors >= n_sources and gain is well-conditioned, R ~ I
np.testing.assert_allclose(R, jnp.eye(n_sources), atol=1e-4)
```

| Method | Spatial resolution | Correlated sources | Depth bias | Speed |
|--------|:------------------:|:------------------:|:----------:|:-----:|
| LCMV   | High (focal)       | Poor               | Yes (use NAI) | Fast |
| MNE/dSPM | Low (distributed)| Good               | Yes/corrected | Fast |
| sLORETA | Moderate          | Good               | No         | Fast  |
| CHAMPAGNE | High            | Good               | No         | Slow  |
| ADMM L1 | High (focal)     | Moderate           | Optional   | Moderate |

---

## 10. Putting It All Together

Here is a complete beamforming analysis condensed into a single script:

```python
import jax
import jax.numpy as jnp
from neurojax.source.beamformer import (
    make_lcmv_filter,
    apply_lcmv,
    lcmv_power_map,
    neural_activity_index,
)
from neurojax.source.champagne import champagne_solver

# 1. Set up forward model (or load from MNE)
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)
n_sensors, n_sources, n_time = 10, 5, 200

gain = jax.random.normal(k1, (n_sensors, n_sources))
sources_true = jnp.zeros((n_sources, n_time))
sources_true = sources_true.at[2].set(jax.random.normal(k2, (n_time,)))
data = gain @ sources_true + 0.1 * jax.random.normal(k3, (n_sensors, n_time))

# 2. Estimate data covariance
cov = (data @ data.T) / n_time

# 3. LCMV beamformer
weights = make_lcmv_filter(cov, gain, reg=0.05)
sources_hat = apply_lcmv(data, weights)
power = lcmv_power_map(cov, gain)

# 4. Verify unit-gain constraint
assert jnp.allclose(weights @ gain, jnp.eye(n_sources), atol=1e-4)

# 5. Localize active source
print(f"Active source (LCMV): {int(jnp.argmax(power))}")

# 6. CHAMPAGNE for comparison
gamma, _ = champagne_solver(cov, gain, max_iter=20)
print(f"Active source (CHAMPAGNE): {int(jnp.argmax(gamma))}")
```

---

## References

- Van Veen BD et al. (1997). Localization of brain electrical activity via
  linearly constrained minimum variance spatial filtering. *IEEE Trans Biomed
  Eng* 44(9):867-880.
- Sekihara K et al. (2001). Reconstructing spatio-temporal activities of neural
  sources using an MEG vector beamformer technique. *IEEE Trans Biomed Eng* 48(7):760-771.
- Gross J et al. (2001). Dynamic imaging of coherent sources: studying neural
  interactions in the human brain. *PNAS* 98(2):694-699.
- Wipf D et al. (2008). A unified Bayesian framework for MEG/EEG source
  imaging. *NeuroImage*.
- Nolte G et al. (2004). Identifying true brain interaction from EEG data
  using the imaginary part of coherency. *Clin Neurophysiol* 115(10):2292-2307.
