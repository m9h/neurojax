# Statistical Analysis: GLM and Permutation Testing

This tutorial covers NeuroJAX's General Linear Model (GLM) and permutation
testing infrastructure for MEG/EEG statistical inference. Everything is built
on JAX and Equinox, giving you automatic differentiation, GPU acceleration,
and `vmap`-parallelized permutation testing that replaces slow Python loops.
We also cover power spectral analysis with `PowerSpectrumModel` for
characterizing oscillatory signatures.

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

print(jax.devices())
```

---

## 1. Introduction to the General Linear Model

The General Linear Model (GLM) is the workhorse of neuroimaging statistics. It
expresses the measured signal at each sensor (or source) as a weighted
combination of known regressors plus residual noise:

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

where:
- $\mathbf{Y} \in \mathbb{R}^{T \times S}$ is the data ($T$ time points, $S$ sensors),
- $\mathbf{X} \in \mathbb{R}^{T \times P}$ is the **design matrix** ($P$ regressors),
- $\boldsymbol{\beta} \in \mathbb{R}^{P \times S}$ are the regression coefficients,
- $\boldsymbol{\varepsilon} \in \mathbb{R}^{T \times S}$ is Gaussian noise.

The GLM subsumes many classical analyses: t-tests, ANOVA, regression, and
correlation are all special cases defined by the structure of $\mathbf{X}$ and
the contrast vector.

---

## 2. Designing the Design Matrix

The design matrix encodes your experimental hypotheses. Each column is a
**regressor** -- a predicted time course of neural activity for one
experimental condition or confound.

```python
rng = np.random.default_rng(42)

n_time = 200       # Time points (e.g., trials or time samples)
n_regressors = 3   # Experimental conditions / covariates
n_sensors = 5      # MEG sensors (or source locations)

# A well-conditioned design matrix with 3 regressors
X = jnp.array(
    rng.standard_normal((n_time, n_regressors)),
    dtype=jnp.float32,
)
```

### Creating Data with Known Effects

To validate the GLM, we generate data where the true betas are known:

```python
# True regression coefficients: (regressors x sensors)
beta_true = jnp.array([
    [ 1.0,  2.0, -1.0,  0.5,  3.0],   # Regressor 0 effects
    [-0.5,  1.5,  0.0,  2.0, -1.0],   # Regressor 1 effects
    [ 0.3, -0.7,  1.2, -0.3,  0.8],   # Regressor 2 effects
], dtype=jnp.float32)

# Simulated data: Y = X @ beta + noise
noise = jnp.array(
    rng.standard_normal((n_time, n_sensors)).astype(np.float32) * 0.01,
)
Y = X @ beta_true + noise
```

```{admonition} Design matrix tips
:class: tip
- **Center continuous regressors** (subtract the mean) to improve numerical
  conditioning and interpretability of the intercept.
- **Check the condition number** of `X.T @ X` -- values above $10^{10}$
  indicate collinearity problems.
- **Include confound regressors** (head motion, cardiac artifacts, linear
  drift) to avoid false positives.
```

---

## 3. Fitting the GLM

NeuroJAX's `GeneralLinearModel` is an [Equinox](https://docs.kidger.site/equinox/)
module, so it follows a functional, immutable API. Calling `.fit()` returns a
**new** model instance with populated `betas` and `residuals`, leaving the
original unchanged.

Under the hood, the fit uses **QR decomposition** via
[Lineax](https://docs.kidger.site/lineax/), which is more numerically stable
than forming the normal equations $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y}$
directly:

$$
\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}}{\mathrm{argmin}} \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2
\quad \Longrightarrow \quad
\mathbf{Q}\mathbf{R}\boldsymbol{\beta} = \mathbf{Y}
\quad \Longrightarrow \quad
\hat{\boldsymbol{\beta}} = \mathbf{R}^{-1}\mathbf{Q}^\top\mathbf{Y}
$$

```python
from neurojax.glm import GeneralLinearModel

# Create the model (unfitted)
glm = GeneralLinearModel(X, Y)
print(f"Before fit: betas = {glm.betas}")
# Before fit: betas = None

# Fit the model (returns a new instance)
fitted = glm.fit()
print(f"After fit: betas shape = {fitted.betas.shape}")
# After fit: betas shape = (3, 5)

# The original model is unchanged (functional style)
assert glm.betas is None
assert fitted.betas is not None
```

```{admonition} Class signature
:class: note
`GeneralLinearModel(design_matrix, data)` -- an `eqx.Module`

- `design_matrix`: `Float[Array, "time regressors"]`
- `data`: `Float[Array, "time sensors"]`

**Attributes** (after `.fit()`):
- `betas`: `Float[Array, "regressors sensors"]` -- fitted coefficients
- `residuals`: `Float[Array, "time sensors"]` -- $\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}$
```

### Verifying Parameter Recovery

With low noise, the fitted betas should closely match the true values:

```python
np.testing.assert_allclose(
    np.array(fitted.betas), np.array(beta_true), atol=0.05,
)
print("Fitted betas match ground truth within tolerance.")

# Residuals should be small
max_residual = float(jnp.max(jnp.abs(fitted.residuals)))
print(f"Max absolute residual: {max_residual:.6f}")
assert max_residual < 0.5
```

### Noiseless Regression (Exact Recovery)

When there is no noise, the GLM should recover the exact betas:

```python
X_small = jnp.array(
    rng.standard_normal((100, 2)), dtype=jnp.float32,
)
beta_exact = jnp.array([[3.0, -1.0], [2.0, 4.0]], dtype=jnp.float32)
Y_exact = X_small @ beta_exact   # No noise

fitted_exact = GeneralLinearModel(X_small, Y_exact).fit()
np.testing.assert_allclose(
    np.array(fitted_exact.betas), np.array(beta_exact), atol=1e-4,
)
print("Noiseless regression: exact recovery confirmed.")
```

---

## 4. Computing t-Statistics and Contrasts

A **contrast vector** $\mathbf{c} \in \mathbb{R}^P$ specifies which linear
combination of the regression coefficients to test. The t-statistic for a
given contrast at sensor $s$ is:

$$
t_s = \frac{\mathbf{c}^\top \hat{\boldsymbol{\beta}}_{\cdot s}}{\text{SE}_s}
$$

where the standard error is:

$$
\text{SE}_s = \sqrt{\hat{\sigma}_s^2 \cdot \mathbf{c}^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{c}}
$$

and the residual variance estimate is:

$$
\hat{\sigma}_s^2 = \frac{\|\mathbf{Y}_{\cdot s} - \mathbf{X}\hat{\boldsymbol{\beta}}_{\cdot s}\|^2}{T - P}
$$

where $T - P$ is the degrees of freedom.

```python
# Set up a model with a known strong effect on regressor 0
n_time_t, n_reg, n_sens = 200, 2, 3
X_t = jnp.array(
    rng.standard_normal((n_time_t, n_reg)), dtype=jnp.float32,
)
# Regressor 0 has a strong effect (beta=5) on sensor 0, zero on sensor 1
beta_t = jnp.array([[5.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
noise_t = jnp.array(
    rng.standard_normal((n_time_t, n_sens)).astype(np.float32) * 0.1,
)
Y_t = X_t @ beta_t + noise_t

fitted_t = GeneralLinearModel(X_t, Y_t).fit()

# Contrast: test regressor 0 (ignore regressor 1)
contrast = jnp.array([1.0, 0.0])
t_values = fitted_t.get_t_stats(contrast)

print(f"t-statistics: {jnp.round(t_values, 2)}")
print(f"  Sensor 0 (true beta=5.0): t = {float(t_values[0]):.1f}")
print(f"  Sensor 1 (true beta=0.0): t = {float(t_values[1]):.1f}")

# Sensor 0 should have a very large t-statistic
assert jnp.abs(t_values[0]) > 10.0, "Strong effect should produce large t"
assert jnp.abs(t_values[1]) < 5.0, "Null effect should produce small t"
```

```{admonition} Method signature
:class: note
`fitted_model.get_t_stats(contrast) -> jnp.ndarray`

- `contrast`: `Float[Array, "regressors"]` -- contrast vector
- **Returns**: `(n_sensors,)` t-statistic at each sensor
- **Raises**: `ValueError` if the model has not been fitted
```

```{admonition} Common contrasts
:class: tip
- `[1, 0, 0]` -- test the first regressor alone
- `[1, -1, 0]` -- test the difference between regressors 0 and 1
- `[0, 0, 1]` -- test the third regressor alone
- `[1/3, 1/3, 1/3]` -- test the average across all three regressors
```

### Calling get_t_stats on an Unfitted Model

Attempting to compute statistics before fitting will raise an informative error:

```python
unfitted = GeneralLinearModel(jnp.ones((10, 2)), jnp.zeros((10, 3)))
try:
    unfitted.get_t_stats(jnp.array([1.0, 0.0]))
except ValueError as e:
    print(f"Expected error: {e}")
# Expected error: Model must be fitted before computing stats.
```

---

## 5. Permutation Testing with Max-t Correction

Parametric p-values from the t-distribution assume Gaussian noise, which may
not hold for MEG data. **Permutation testing** provides exact, assumption-free
inference by empirically constructing a null distribution.

### The Algorithm

1. **Fit the true model** and compute observed t-statistics $t_s^{\text{obs}}$
   at each sensor.
2. For each of $B$ permutations:
   a. Randomly shuffle the rows of $\mathbf{Y}$ (breaking the X-Y relationship
      while preserving spatial correlation).
   b. Fit the GLM on shuffled data and compute permutation t-statistics $t_s^{(b)}$.
3. **Max-t correction** (Nichols & Holmes, 2002): for each permutation, take
   the maximum $|t|$ across all sensors: $t_{\max}^{(b)} = \max_s |t_s^{(b)}|$.
   This builds a null distribution that controls the **family-wise error rate**
   (FWER) across sensors.
4. The corrected p-value at each sensor is:

$$
p_s = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}\left[t_{\max}^{(b)} \geq |t_s^{\text{obs}}|\right]
$$

### JAX-Accelerated Permutation Testing

NeuroJAX uses `jax.vmap` to run all $B$ permutations in parallel on the GPU,
rather than looping sequentially in Python. This can provide 100x+ speedups
over CPU-based implementations:

```python
from neurojax.glm import GeneralLinearModel, run_permutation_test

# Create a model with a strong signal on one sensor
n_time_p = 100
X_p = jnp.array(
    rng.standard_normal((n_time_p, 2)), dtype=jnp.float32,
)
beta_p = jnp.array([[50.0], [0.0]])   # Very strong effect
noise_p = jnp.array(
    rng.standard_normal((n_time_p, 1)).astype(np.float32) * 0.01,
)
Y_p = X_p @ beta_p + noise_p

model = GeneralLinearModel(X_p, Y_p)
contrast = jnp.array([1.0, 0.0])
key = jax.random.PRNGKey(123)

# Run 100 permutations
true_t, p_values = run_permutation_test(model, contrast, key, n_perms=100)

print(f"Observed t-statistic: {float(true_t[0]):.2f}")
print(f"Corrected p-value: {float(p_values[0]):.4f}")

# With such a strong signal, p should be very small
assert p_values[0] < 0.05, "Strong signal should be significant"
```

```{admonition} Function signature
:class: note
`run_permutation_test(model, contrast, key, n_perms=1000) -> (true_t, p_values)`

- `model`: `GeneralLinearModel` (unfitted -- the function calls `.fit()` internally)
- `contrast`: `Array` -- contrast vector of length `n_regressors`
- `key`: `PRNGKeyArray` -- JAX random key for reproducibility
- `n_perms`: `int` -- number of permutations (default 1000)
- **Returns**: `true_t` `(n_sensors,)` observed t-statistics, `p_values` `(n_sensors,)` corrected p-values

The function is wrapped with `@eqx.filter_jit` for XLA compilation.
```

### Validating Permutation Test Properties

The p-values from permutation testing must satisfy basic sanity checks:

```python
# Multi-sensor null data (no true effect)
n_time_null, n_reg_null, n_sens_null = 50, 2, 3
X_null = jnp.array(
    rng.standard_normal((n_time_null, n_reg_null)), dtype=jnp.float32,
)
Y_null = jnp.array(
    rng.standard_normal((n_time_null, n_sens_null)), dtype=jnp.float32,
)
model_null = GeneralLinearModel(X_null, Y_null)
contrast_null = jnp.array([1.0, 0.0])
key_null = jax.random.PRNGKey(0)

true_t_null, p_null = run_permutation_test(
    model_null, contrast_null, key_null, n_perms=50,
)

# p-values must be in [0, 1]
assert jnp.all(p_null >= 0.0) and jnp.all(p_null <= 1.0)

# Shapes match the number of sensors
assert true_t_null.shape == (n_sens_null,)
assert p_null.shape == (n_sens_null,)

print(f"Null data p-values: {jnp.round(p_null, 3)}")
print("All p-values in valid range: confirmed")
```

```{admonition} Choosing the number of permutations
:class: tip
- For exploratory analysis: 500-1000 permutations are usually sufficient.
- For publication: 5000-10000 permutations give more precise p-value estimates.
- On GPU with `vmap`, 5000 permutations of a moderate GLM runs in seconds.
- The minimum achievable p-value is $1/B$, so with 1000 permutations the
  smallest possible p-value is 0.001.
```

---

## 6. Interpreting Results

### Effect Sizes

The raw beta coefficients $\hat{\boldsymbol{\beta}}$ give the effect size in
the original data units. For standardized effect sizes, divide by the residual
standard deviation:

```python
# Cohen's d-like measure: beta / residual_std
residual_std = jnp.std(fitted_t.residuals, axis=0)
effect_sizes = fitted_t.betas / residual_std[None, :]
print(f"Standardized effect sizes:\n{jnp.round(effect_sizes, 2)}")
```

### Multiple Comparisons

The max-t correction from `run_permutation_test()` controls the **family-wise
error rate** (FWER) -- the probability of any false positive across all
sensors. This is more conservative than uncorrected testing but less
conservative than Bonferroni correction, because it accounts for the
correlation structure of the test statistics across sensors.

| Method | Controls | Conservatism | Assumption |
|--------|----------|:------------:|:----------:|
| Uncorrected | Per-test Type I | None | Gaussian |
| Bonferroni | FWER | Very conservative | None |
| Max-t permutation | FWER | Moderate | Exchangeability |
| Cluster-based permutation | FWER (cluster) | Liberal | Exchangeability |

---

## 7. Power Spectral Analysis

NeuroJAX provides a differentiable power spectrum model inspired by
[SpecParam/FOOOF](https://fooof-tools.github.io/fooof/), decomposing
neural power spectra into **aperiodic** (1/f-like) and **periodic**
(oscillatory peak) components.

### The Spectral Model

The total log-power spectrum is modeled as:

$$
\log_{10} P(f) = \underbrace{b - \chi \log_{10}(f)}_{\text{aperiodic}} + \underbrace{\sum_{k=1}^{K} a_k \exp\left(-\frac{(f - f_k)^2}{2\sigma_k^2}\right)}_{\text{periodic peaks}}
$$

where:
- $b$ is the aperiodic offset (broadband power level),
- $\chi$ is the aperiodic exponent (spectral slope),
- Each peak $k$ has center frequency $f_k$, power $a_k$, and bandwidth $\sigma_k$.

### Using PowerSpectrumModel

```python
from neurojax.spectral import PowerSpectrumModel

model_spec = PowerSpectrumModel()

# Define frequency axis
freqs = jnp.linspace(1.0, 50.0, 500)

# Parameters: [offset, exponent, peak_cf, peak_power, peak_bw]
# A spectrum with aperiodic slope and a single alpha peak at 10 Hz
params = jnp.array([1.0, 1.0, 10.0, 0.5, 1.0])
psd = model_spec(freqs, params)

print(f"PSD shape: {psd.shape}")
# PSD shape: (500,)

# The peak should be near 10 Hz
peak_idx = int(jnp.argmax(psd))
print(f"Peak frequency: {float(freqs[peak_idx]):.1f} Hz")
```

```{admonition} Parameter layout
:class: note
`params = [offset, exponent, peak1_cf, peak1_pw, peak1_bw, peak2_cf, peak2_pw, peak2_bw, ...]`

- First 2 parameters: aperiodic component (offset, exponent)
- Remaining parameters: groups of 3 per peak (center frequency, power, bandwidth)
- Power and bandwidth are passed through `jax.nn.softplus()` to enforce positivity
```

### Multiple Peaks

You can model spectra with multiple oscillatory peaks (e.g., alpha + beta):

```python
# Two peaks: alpha (10 Hz) and beta (30 Hz)
params_2peak = jnp.array([
    1.0, 1.0,        # aperiodic: offset, exponent
    10.0, 2.0, 1.0,  # alpha peak: 10 Hz, power=2, bw=1
    30.0, 1.5, 1.0,  # beta peak: 30 Hz, power=1.5, bw=1
])
psd_2peak = model_spec(freqs, params_2peak)
print(f"Two-peak PSD shape: {psd_2peak.shape}")
assert jnp.isfinite(psd_2peak).all()
```

### Aperiodic-Only Spectrum

With a large negative peak power (suppressed by softplus), the model reduces
to a pure 1/f spectrum:

```python
# Aperiodic only (peak power driven to ~0 by softplus)
params_aperiodic = jnp.array([2.0, 1.5, 10.0, -20.0, 1.0])
psd_aperiodic = model_spec(freqs, params_aperiodic)

# Power should decrease monotonically with frequency
diffs = jnp.diff(psd_aperiodic)
n_decreasing = int(jnp.sum(diffs < 0))
print(f"Monotonically decreasing samples: {n_decreasing}/{len(diffs)}")
assert n_decreasing > 80, "Aperiodic spectrum should mostly decrease"
```

### JIT Compatibility

The model is fully compatible with `jax.jit`:

```python
@jax.jit
def evaluate_spectrum(freqs, params):
    return model_spec(freqs, params)

psd_jit = evaluate_spectrum(freqs, params)
assert jnp.isfinite(psd_jit).all()
print("JIT compilation: successful")
```

---

## 8. Spectrum Fitting

The `fit_spectrum()` function uses [Optimistix](https://docs.kidger.site/optimistix/)
Levenberg-Marquardt optimization to fit the `PowerSpectrumModel` to observed
power spectra:

```python
from neurojax.spectral import PowerSpectrumModel, fit_spectrum

# Generate a synthetic spectrum with known parameters
true_params = jnp.array([2.0, 1.5, 10.0, 1.0, 1.0])
psd_target = model_spec(freqs, true_params)

# Fit the model
fitted_params = fit_spectrum(
    freqs, psd_target, n_peaks=1,
    initial_params=true_params * 1.1,  # Slightly perturbed initial guess
)

print(f"True params:   {jnp.round(jnp.array(true_params), 3)}")
print(f"Fitted params: {jnp.round(fitted_params, 3)}")

# Aperiodic parameters should be recovered
np.testing.assert_allclose(float(fitted_params[0]), 2.0, atol=0.5)
np.testing.assert_allclose(float(fitted_params[1]), 1.5, atol=0.5)
```

```{admonition} Function signature
:class: note
`fit_spectrum(freqs, power_spectrum, n_peaks=1, initial_params=None) -> jnp.ndarray`

- `freqs`: `(n_freqs,)` frequency axis in Hz
- `power_spectrum`: `(n_freqs,)` log-power spectral density
- `n_peaks`: `int` -- number of oscillatory peaks to model
- `initial_params`: optional initial parameter guess; if `None`, uses heuristic defaults
- **Returns**: `(2 + 3*n_peaks,)` fitted parameters
```

### Fitting a Real Sinusoidal Signal

A more realistic test: compute the FFT of a 10 Hz sinusoid and fit the
spectral model to recover the peak frequency:

```python
# Generate a 10 Hz sinusoid
sfreq = 256.0
n_samples = 2048
t = jnp.arange(n_samples) / sfreq
signal = jnp.sin(2 * jnp.pi * 10.0 * t)

# Power spectrum via FFT
fft_vals = jnp.fft.rfft(np.array(signal))
freqs_fft = jnp.fft.rfftfreq(n_samples, d=1.0 / sfreq)
psd_fft = jnp.abs(fft_vals) ** 2 / n_samples

# Restrict to frequencies above 1 Hz (log-space model)
mask = freqs_fft >= 1.0
freqs_pos = freqs_fft[mask]
psd_log = jnp.log10(psd_fft[mask] + 1e-20)

# Fit with informed initial guess
init = jnp.array([0.0, 0.0, 10.0, 5.0, 0.5])
fitted_fft = fit_spectrum(freqs_pos, psd_log, n_peaks=1, initial_params=init)

peak_cf = float(fitted_fft[2])
print(f"Fitted peak frequency: {peak_cf:.1f} Hz (expected ~10 Hz)")
assert abs(peak_cf - 10.0) < 3.0, f"Peak at {peak_cf}, expected ~10 Hz"
```

---

## 9. Model Comparison with Log-Likelihood and BIC

The `GeneralLinearModel` provides a `log_likelihood()` method that uses
[Distrax](https://github.com/google-deepmind/distrax) to compute the Gaussian
log-likelihood of the residuals. This enables model comparison via information
criteria:

$$
\text{BIC} = -2 \ln \hat{L} + k \ln T
$$

where $\hat{L}$ is the maximized likelihood, $k$ is the number of free
parameters, and $T$ is the number of observations.

```python
# Fit two competing models with different numbers of regressors
X_full = jnp.array(
    rng.standard_normal((100, 3)), dtype=jnp.float32,
)
Y_ll = jnp.array(
    rng.standard_normal((100, 3)), dtype=jnp.float32,
)

# Full model (3 regressors)
fitted_full = GeneralLinearModel(X_full, Y_ll).fit()
ll_full = float(fitted_full.log_likelihood())

# Reduced model (2 regressors)
fitted_reduced = GeneralLinearModel(X_full[:, :2], Y_ll).fit()
ll_reduced = float(fitted_reduced.log_likelihood())

print(f"Full model log-likelihood: {ll_full:.2f}")
print(f"Reduced model log-likelihood: {ll_reduced:.2f}")

# Log-likelihood should be finite and negative
assert jnp.isfinite(jnp.array(ll_full))
assert ll_full < 0

# BIC computation
n_obs = 100
n_sensors_ll = 3
k_full = 3 * n_sensors_ll     # parameters = regressors * sensors
k_reduced = 2 * n_sensors_ll

bic_full = -2 * ll_full + k_full * np.log(n_obs)
bic_reduced = -2 * ll_reduced + k_reduced * np.log(n_obs)

print(f"\nBIC (full model):    {bic_full:.2f}")
print(f"BIC (reduced model): {bic_reduced:.2f}")
print(f"Preferred model: {'full' if bic_full < bic_reduced else 'reduced'}")
```

```{admonition} Method signature
:class: note
`fitted_model.log_likelihood() -> jnp.ndarray`

- **Returns**: scalar log-likelihood summed over all time points and sensors
- **Raises**: `ValueError` if the model has not been fitted
- Models residuals as independent normals with per-sensor variance
```

```{admonition} When to use BIC vs. permutation testing
:class: tip
- **BIC/AIC** are useful for comparing nested models (e.g., "does adding this
  regressor improve the fit?"). They do not provide p-values.
- **Permutation testing** is used for hypothesis testing ("is this effect
  statistically significant?") and directly controls the false positive rate.
- In a complete analysis, you might use BIC to select the best model, then
  permutation testing to assess the significance of its contrasts.
```

---

## 10. Putting It All Together

A complete analysis pipeline combining GLM fitting, permutation testing, and
spectral characterization:

```python
import jax
import jax.numpy as jnp
import numpy as np
from neurojax.glm import GeneralLinearModel, run_permutation_test
from neurojax.spectral import PowerSpectrumModel, fit_spectrum

rng = np.random.default_rng(42)

# 1. Prepare design matrix and data
n_time, n_regressors, n_sensors = 200, 3, 5
X = jnp.array(rng.standard_normal((n_time, n_regressors)), dtype=jnp.float32)
beta_true = jnp.array([
    [1.0, 2.0, -1.0, 0.5, 3.0],
    [-0.5, 1.5, 0.0, 2.0, -1.0],
    [0.3, -0.7, 1.2, -0.3, 0.8],
], dtype=jnp.float32)
noise = jnp.array(
    rng.standard_normal((n_time, n_sensors)).astype(np.float32) * 0.01,
)
Y = X @ beta_true + noise

# 2. Fit the GLM
model = GeneralLinearModel(X, Y)
fitted = model.fit()
print(f"Beta recovery error: {float(jnp.max(jnp.abs(fitted.betas - beta_true))):.6f}")

# 3. Compute t-statistics for the first regressor
contrast = jnp.array([1.0, 0.0, 0.0])
t_stats = fitted.get_t_stats(contrast)
print(f"t-statistics: {jnp.round(t_stats, 1)}")

# 4. Permutation test for significance
key = jax.random.PRNGKey(0)
true_t, p_values = run_permutation_test(model, contrast, key, n_perms=500)
print(f"Corrected p-values: {jnp.round(p_values, 4)}")

# 5. Model comparison
ll = float(fitted.log_likelihood())
bic = -2 * ll + (n_regressors * n_sensors) * np.log(n_time)
print(f"BIC: {bic:.2f}")

# 6. Spectral analysis on a sensor
spec_model = PowerSpectrumModel()
freqs = jnp.linspace(1.0, 50.0, 200)
init_params = jnp.array([2.0, 1.0, 10.0, 1.0, 1.0])
psd = spec_model(freqs, init_params)
fitted_params = fit_spectrum(freqs, psd, n_peaks=1)
print(f"Fitted aperiodic exponent: {float(fitted_params[1]):.2f}")
```

---

## References

- Friston KJ et al. (1994). Statistical parametric maps in functional imaging:
  a general linear approach. *Human Brain Mapping* 2(4):189-210.
- Nichols TE, Holmes AP (2002). Nonparametric permutation tests for functional
  neuroimaging: a primer with examples. *Human Brain Mapping* 15(1):1-25.
- Donoghue T et al. (2020). Parameterizing neural power spectra into periodic
  and aperiodic components. *Nature Neuroscience* 23:1655-1665.
- Kidger P (2021). On Neural Differential Equations. PhD thesis, University
  of Oxford. (Equinox, Lineax, Optimistix, Diffrax.)
