# Blind Source Separation with ICA

Independent Component Analysis (ICA) recovers statistically independent
signals from observed mixtures.  In MEG/EEG analysis the "cocktail party
problem" arises naturally: sensor recordings are linear superpositions of
neural sources, physiological artifacts (heartbeat, eye blinks), and noise.
ICA decomposes these mixtures into their constituent components so that
artifacts can be identified and removed, or so that specific neural sources
can be studied in isolation.

This tutorial covers the full ICA workflow in NeuroJAX -- from synthetic
data generation through dimensionality estimation, FastICA decomposition,
probabilistic ICA, and validation of source recovery.

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
import numpy as np

print(jax.devices())  # should list at least one device
```

---

## 1. The ICA Mixing Model

The generative model behind ICA assumes that $p$ observed signals
$\mathbf{x}(t)$ are linear mixtures of $k$ latent independent sources
$\mathbf{s}(t)$:

$$
\mathbf{x}(t) = \mathbf{A}\,\mathbf{s}(t)
$$

where $\mathbf{A} \in \mathbb{R}^{p \times k}$ is the unknown **mixing
matrix**.  The goal of ICA is to estimate an **unmixing matrix**
$\mathbf{W} \approx \mathbf{A}^{-1}$ such that

$$
\hat{\mathbf{s}}(t) = \mathbf{W}\,\mathbf{x}(t)
$$

recovers the original sources up to arbitrary sign flips and permutation of
components (both are inherent ambiguities of ICA).

---

## 2. Generating Synthetic Sources

We start by creating three prototypical independent sources: a sinusoid, a
square wave, and Gaussian noise.  These have very different statistical
profiles -- the sinusoid is sub-Gaussian, the square wave is super-Gaussian,
and noise is Gaussian -- making them ideal test signals for ICA.

```python
n_samples = 2000
time = jnp.linspace(0, 8, n_samples)

# Three independent sources
s1 = jnp.sin(2 * time)                                      # sinusoid
s2 = jnp.sign(jnp.sin(3 * time))                            # square wave
s3 = jax.random.normal(jax.random.PRNGKey(0), (n_samples,))  # Gaussian noise

# Stack into a (3, n_samples) source matrix and standardize
S = jnp.stack([s1, s2, s3])
S = S / S.std(axis=1, keepdims=True)
```

```{note}
Standardizing each source to unit variance is good practice.  ICA cannot
recover the original scale of the sources (only their shape), so
normalizing up front avoids numerical issues during whitening.
```

---

## 3. Creating a Mixing Matrix and Mixed Signals

Now simulate the observation process.  A fixed mixing matrix $\mathbf{A}$
combines the sources into three observed channels:

```python
A = jnp.array([
    [1.0, 1.0, 1.0],
    [0.5, 2.0, 1.0],
    [1.5, 1.0, 2.0],
])

# Mixed observations: X = A @ S
X = jnp.dot(A, S)   # shape (3, 2000)
```

Each row of `X` now contains a superposition of all three sources -- just as
a single MEG sensor records a weighted mixture of all active neural and
artifactual sources.

---

## 4. PCA Whitening with `whiten_pca()`

Before running ICA, the data must be **whitened**: centered, decorrelated,
and scaled so that the covariance matrix of the transformed data equals the
identity.  Whitening reduces the ICA optimization from a general search for
$\mathbf{W}$ to a search over orthogonal matrices, which is far easier.

NeuroJAX provides `whiten_pca()` in `neurojax.analysis.decomposition`:

```python
from neurojax.analysis.decomposition import whiten_pca

# Whiten and optionally reduce to n_components
X_white, W_white, mu, eigenvalues, eigenvectors = whiten_pca(X, n_components=3)
```

The function returns five values:

| Return value | Shape | Description |
|---|---|---|
| `X_white` | `(n_components, n_samples)` | Whitened data |
| `W_white` | `(n_components, n_features)` | Whitening matrix |
| `mu` | `(n_features, 1)` | Channel means |
| `eigenvalues` | `(n_components,)` | Retained eigenvalues (descending) |
| `eigenvectors` | `(n_features, n_components)` | Corresponding eigenvectors |

After whitening, the covariance of `X_white` should be close to the identity:

```python
cov_white = jnp.dot(X_white, X_white.T) / (X_white.shape[1] - 1)
print("Covariance after whitening (should be ~I):")
print(jnp.round(cov_white, 2))
```

```{tip}
When `n_components=None`, `whiten_pca()` automatically estimates the
dimensionality using `estimate_dimension_laplace()` (see next section).
This is the recommended default for real MEG/EEG data where the true
number of sources is unknown.
```

---

## 5. Dimensionality Estimation

Choosing the right number of components is critical: too few discards signal,
too many introduces noise components that destabilize ICA.  NeuroJAX provides
two complementary approaches.

### 5.1 Laplace Approximation (`estimate_dimension_laplace`)

This implements Minka's (2000) Bayesian PCA model selection.  Given the
eigenvalue spectrum of the data covariance, it estimates the number of
components that explains 95% of the variance:

```python
from neurojax.analysis.decomposition import estimate_dimension_laplace

# Full eigenspectrum from PCA
_, _, _, all_eigenvalues, _ = whiten_pca(X, n_components=3)

# Estimate dimensionality
k_hat = estimate_dimension_laplace(all_eigenvalues, n_samples=X.shape[1])
print(f"Estimated dimensionality: {k_hat}")
```

### 5.2 PPCA with AIC/BIC (`PPCA.estimate_dimensionality`)

The `PPCA` class in `neurojax.analysis.dimensionality` evaluates the Bayesian
model evidence for each candidate dimensionality $k = 1, \ldots, k_{\max}$
using information criteria:

```python
from neurojax.analysis.dimensionality import PPCA

# BIC-based estimate (conservative)
k_bic = PPCA.estimate_dimensionality(X, method="bic")
print(f"BIC estimate: {k_bic}")

# AIC-based estimate (more liberal)
k_aic = PPCA.estimate_dimensionality(X, method="aic")
print(f"AIC estimate: {k_aic}")

# Consensus (average of AIC and BIC, rounded)
k_consensus = PPCA.estimate_dimensionality(X, method="consensus")
print(f"Consensus estimate: {k_consensus}")
```

You can also inspect the full evidence curves to see whether there is a
clear "elbow":

```python
evidence = PPCA.get_laplace_evidence(X, max_dim=5)
print("Log-evidence per k:", evidence)

scores = PPCA.get_consensus_evidence(X, max_dim=5)
print("AIC scores:", scores["aic"])
print("BIC scores:", scores["bic"])
```

```{note}
For real MEG data, BIC tends to underestimate and AIC tends to
overestimate the number of components.  The consensus method (default)
averages the two, which is the strategy used by FSL MELODIC for fMRI
probabilistic ICA.
```

---

## 6. FastICA Source Recovery

NeuroJAX offers two FastICA implementations depending on your workflow.

### 6.1 Equinox Module (`preprocessing.ica.FastICA`)

The `FastICA` class in `neurojax.preprocessing.ica` is an Equinox module that
handles whitening, ICA, and source projection in a single object.  Because
Equinox modules are immutable, `fit()` returns a **new** instance with
populated attributes:

```python
from neurojax.preprocessing.ica import FastICA

ica = FastICA(n_components=3, max_iter=1000, tol=1e-5)

# fit() returns a new module with learned parameters
model = ica.fit(X, key=jax.random.PRNGKey(42))

# Inspect fitted attributes
print(f"Mixing matrix shape:  {model.mixing_.shape}")   # (3, 3)
print(f"Components shape:     {model.components_.shape}")  # (3, 2000)
print(f"Mean shape:           {model.mean_.shape}")       # (3, 1)
```

Recover the independent components:

```python
S_recovered = model.apply(X)   # shape (3, 2000)
```

The `apply()` method can also project new, unseen data into the learned
component space:

```python
# Apply to new data with the same channel layout
X_new = jax.random.normal(jax.random.PRNGKey(99), (3, 500))
S_new = model.apply(X_new)   # shape (3, 500)
```

### 6.2 Functional API (`analysis.decomposition.fastica`)

For more control, use the functional API which operates on pre-whitened data:

```python
from neurojax.analysis.decomposition import whiten_pca, fastica

# Step 1: Whiten
X_white, W_white, mu, evals, evecs = whiten_pca(X, n_components=3)

# Step 2: Run FastICA on whitened data
W_ica = fastica(X_white, n_components=3, random_key=jax.random.PRNGKey(42),
                max_iter=200, tol=1e-4)

# Step 3: Unmix
S_recovered = jnp.dot(W_ica, X_white)
```

The returned `W_ica` is an orthogonal unmixing matrix.  The functional API
is useful when you need to integrate ICA into a custom pipeline or want
access to the intermediate whitening parameters.

```{tip}
The FastICA algorithm uses `tanh` (logcosh) as its contrast function.
This is a good default that works well for both sub-Gaussian (sinusoidal)
and super-Gaussian (sparse/impulsive) sources.  The algorithm converges
when the change in the unmixing matrix drops below `tol`.
```

---

## 7. Probabilistic ICA with `probabilistic_ica()`

Probabilistic ICA (PICA) extends standard ICA by embedding it within a
probabilistic PCA framework.  The pipeline is:

1. **PPCA whitening** -- reduce dimensionality using Bayesian model
   selection to choose $k$ automatically
2. **FastICA** -- recover independent components in the whitened space
3. **Z-scoring** -- normalize components for thresholding and interpretation

This mirrors the approach used by FSL MELODIC for fMRI analysis.

```python
from neurojax.analysis.decomposition import probabilistic_ica

S_z, Mixing, pca_eigenvalues = probabilistic_ica(
    X, n_components=3, key=jax.random.PRNGKey(10)
)

print(f"Z-scored ICs shape: {S_z.shape}")        # (3, 2000)
print(f"Mixing matrix shape: {Mixing.shape}")     # (3, 3) if X is (3, n)
print(f"PCA eigenvalues: {pca_eigenvalues}")
```

When `n_components=None`, PICA uses `estimate_dimension_laplace()` to
determine the number of components automatically:

```python
S_z_auto, Mixing_auto, evals_auto = probabilistic_ica(X, n_components=None)
print(f"Auto-selected {S_z_auto.shape[0]} components")
```

The Z-scored components (`S_z`) have zero mean and unit variance, which makes
it straightforward to threshold them (e.g., $|z| > 3$) for identifying
activations or artifacts.

---

## 8. Validating Source Recovery

ICA recovers sources up to arbitrary sign flips and permutations.  To
validate recovery, compute the cross-correlation matrix between the true
and recovered sources:

```python
# Using the Equinox FastICA model from Section 6.1
S_recovered = model.apply(X)

# Cross-correlation matrix
correlation_matrix = jnp.corrcoef(S, S_recovered)
# The matrix is (6, 6): top-left 3x3 is S vs S, bottom-right 3x3 is
# S_rec vs S_rec, and top-right 3x3 is S vs S_rec (the one we want)
cross_corr = correlation_matrix[:3, 3:]

print("Cross-correlation (S vs S_recovered):")
print(jnp.round(cross_corr, 3))

# For each true source, find the best-matching recovered component
max_correlations = jnp.max(jnp.abs(cross_corr), axis=1)
print(f"Max |correlation| per source: {max_correlations}")

# Good recovery: all values should be > 0.9
assert jnp.all(max_correlations > 0.9), "ICA failed to recover sources"
print("All sources recovered successfully!")
```

```{note}
We take `jnp.abs()` of the correlations because ICA can recover sources
with flipped sign.  A correlation of -0.99 is just as good as +0.99 --
it means the source was recovered perfectly but with opposite polarity.
```

For a more structured evaluation, you can match components to sources using
the Hungarian algorithm or simply inspect which recovered component
correlates most strongly with each source:

```python
for i in range(S.shape[0]):
    correlations = [float(jnp.abs(jnp.corrcoef(S_recovered[j], S[i])[0, 1]))
                    for j in range(S_recovered.shape[0])]
    best_match = int(jnp.argmax(jnp.array(correlations)))
    print(f"Source {i} best matched by component {best_match} "
          f"(|r| = {correlations[best_match]:.4f})")
```

---

## 9. Practical Tips

### Choosing the number of components

| Strategy | When to use |
|---|---|
| `PPCA.estimate_dimensionality(X, method="consensus")` | Automated analysis, no prior knowledge |
| `PPCA.estimate_dimensionality(X, method="bic")` | Conservative estimate (fewer components) |
| `whiten_pca(X, n_components=None)` | Let the Laplace approximation decide |
| Fixed `n_components` | When you know the number of sources a priori |

For MEG data with ~300 sensors, typical ICA decompositions use 40--80
components.  For EEG with 64 channels, 20--40 components is common.

### Random seed matters

FastICA is initialized with a random unmixing matrix, so the random key
affects convergence.  If results seem unstable, try:

- Increasing `max_iter` (default 200, try 1000)
- Decreasing `tol` (default 1e-4, try 1e-5)
- Running with multiple seeds and checking consistency

### Interpreting mixing matrices

The mixing matrix $\mathbf{A}$ (available as `model.mixing_`) maps from
component space back to sensor space.  Each column of $\mathbf{A}$ is a
**spatial pattern** (or topographic map) for one independent component.
For MEG/EEG:

- Focal dipolar patterns suggest neural sources
- Frontal patterns with eye-blink time courses suggest EOG artifacts
- Regular periodic time courses may indicate cardiac artifacts

### Equinox immutability

Because `FastICA` is an Equinox module, `fit()` returns a new object rather
than modifying the original in place.  Always capture the return value:

```python
# Correct
model = ica.fit(X, key=jax.random.PRNGKey(42))

# Wrong -- ica is unchanged after this call!
ica.fit(X, key=jax.random.PRNGKey(42))
```

---

## Complete Example

Putting it all together:

```python
import jax
import jax.numpy as jnp
from neurojax.preprocessing.ica import FastICA
from neurojax.analysis.decomposition import whiten_pca, probabilistic_ica
from neurojax.analysis.dimensionality import PPCA

# 1. Generate sources
n_samples = 2000
time = jnp.linspace(0, 8, n_samples)
s1 = jnp.sin(2 * time)
s2 = jnp.sign(jnp.sin(3 * time))
s3 = jax.random.normal(jax.random.PRNGKey(0), (n_samples,))
S = jnp.stack([s1, s2, s3])
S = S / S.std(axis=1, keepdims=True)

# 2. Mix
A = jnp.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
X = A @ S

# 3. Estimate dimensionality
k = PPCA.estimate_dimensionality(X, method="consensus")
print(f"Estimated {k} components")

# 4. Run ICA
ica = FastICA(n_components=3, max_iter=1000, tol=1e-5)
model = ica.fit(X, key=jax.random.PRNGKey(42))

# 5. Recover sources
S_recovered = model.apply(X)

# 6. Validate
cross_corr = jnp.corrcoef(S, S_recovered)[:3, 3:]
max_corrs = jnp.max(jnp.abs(cross_corr), axis=1)
print(f"Source recovery correlations: {max_corrs}")
```
