# Learning Linear Dynamics with the Koopman Operator

This tutorial introduces the **Koopman operator** framework for analysing
nonlinear dynamical systems through a linear lens.  We use NeuroJAX's
`KoopmanEstimator` -- an SVD-based Dynamic Mode Decomposition (DMD)
implementation -- to estimate the operator, extract eigenvalues and modes, and
make multi-step predictions.

```{contents}
:depth: 3
:local:
```

## Prerequisites

Install NeuroJAX with its dynamics dependencies:

```bash
uv sync --extra dynamics
```

```python
import jax
import jax.numpy as jnp
from neurojax.dynamics.koopman import KoopmanEstimator

print(jax.devices())  # should list at least one device
```

---

## 1. Introduction to Koopman Theory

Consider a discrete-time nonlinear dynamical system:

$$
\mathbf{x}_{k+1} = \mathbf{F}(\mathbf{x}_k)
$$

The **Koopman operator** $\mathcal{K}$ is an infinite-dimensional *linear*
operator that acts on **observable functions** $g: \mathbb{R}^n \to \mathbb{R}$:

$$
\mathcal{K} g = g \circ \mathbf{F}
$$

In other words, $\mathcal{K}$ advances any measurement function one time step
forward.  The key insight is that *even though $\mathbf{F}$ is nonlinear, the
Koopman operator is linear* -- at the cost of being infinite-dimensional.

In practice, we approximate $\mathcal{K}$ by a finite-dimensional matrix using
**Dynamic Mode Decomposition (DMD)**.  Given snapshot pairs
$(\mathbf{X}, \mathbf{Y})$ where $\mathbf{Y} = \mathbf{F}(\mathbf{X})$, DMD
finds the best-fit linear operator:

$$
\mathbf{A} = \mathbf{Y} \mathbf{X}^\dagger
$$

where $\mathbf{X}^\dagger$ is the pseudoinverse of $\mathbf{X}$.  In practice,
this is computed via truncated SVD for numerical stability and rank control.

### The DMD Algorithm

Given $\mathbf{X}, \mathbf{Y} \in \mathbb{R}^{n \times m}$:

1. Compute the SVD: $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^*$
2. Optionally truncate to rank $r$: keep the leading $r$ singular values
3. Project onto the POD basis:
   $\tilde{\mathbf{A}} = \mathbf{U}_r^* \mathbf{Y} \mathbf{V}_r \boldsymbol{\Sigma}_r^{-1}$
4. Compute eigendecomposition:
   $\tilde{\mathbf{A}} \mathbf{W} = \mathbf{W} \boldsymbol{\Lambda}$
5. Recover DMD modes: $\boldsymbol{\Phi} = \mathbf{Y} \mathbf{V}_r \boldsymbol{\Sigma}_r^{-1} \mathbf{W}$

The eigenvalues $\boldsymbol{\Lambda}$ encode the system's characteristic
frequencies and growth/decay rates.

---

## 2. Setting Up a Rotational Dynamical System

We start with a simple 2D rotation -- a linear system where DMD can recover
the exact dynamics.  The discrete-time map is:

$$
\begin{bmatrix} x_{k+1} \\ y_{k+1} \end{bmatrix} =
\begin{bmatrix} \cos\omega & -\sin\omega \\ \sin\omega & \cos\omega \end{bmatrix}
\begin{bmatrix} x_k \\ y_k \end{bmatrix}
$$

with rotation angle $\omega = \pi/4$ (45 degrees per step):

```python
theta = jnp.pi / 4  # 45-degree rotation

K_true = jnp.array([
    [jnp.cos(theta), -jnp.sin(theta)],
    [jnp.sin(theta),  jnp.cos(theta)],
])
print("True Koopman operator (rotation matrix):")
print(K_true)
```

Now generate snapshot pairs.  We draw random initial conditions and apply the
rotation:

```python
n_samples = 100

# Random initial states: shape (2, n_samples)
X = jax.random.normal(jax.random.PRNGKey(0), (2, n_samples))

# One-step-forward snapshots
Y = K_true @ X

print(f"X shape: {X.shape}")  # (2, 100)
print(f"Y shape: {Y.shape}")  # (2, 100)
```

Note the data layout: each column is a state vector, each row is a state
dimension.  This is the standard DMD convention (features-by-snapshots).

---

## 3. Computing the Koopman Operator with `KoopmanEstimator.fit()`

The `KoopmanEstimator` wraps the SVD-based DMD algorithm.  The `rank`
parameter controls truncation -- set `rank=0` for full rank (no truncation):

```python
estimator = KoopmanEstimator(rank=0)  # Full-rank estimation
K_est, evals, modes = estimator.fit(X, Y)
```

The `fit()` method returns a tuple of three arrays:

```
KoopmanEstimator.fit(
    X: jnp.ndarray,       # (n, m) snapshot matrix at time k
    Y: jnp.ndarray,       # (n, m) snapshot matrix at time k+1
) -> Tuple[
    jnp.ndarray,           # (n, n) estimated Koopman operator K_est
    jnp.ndarray,           # (r,) complex eigenvalues
    jnp.ndarray,           # (n, r) DMD modes (columns)
]
```

Let us verify the estimated operator matches the ground truth:

```python
print("Estimated Koopman operator:")
print(K_est)

print(f"\nMax error: {jnp.max(jnp.abs(K_est - K_true)):.2e}")
assert jnp.allclose(K_est, K_true, atol=1e-5)
```

With clean linear data and sufficient samples, DMD recovers the exact operator
to near machine precision.

---

## 4. Eigenvalue Analysis

The eigenvalues of the Koopman operator encode the fundamental dynamical
properties of the system.  For a discrete-time system:

- **Magnitude** $|\lambda|$: growth ($>1$), decay ($<1$), or neutrally
  stable ($= 1$)
- **Phase** $\angle\lambda$: oscillation frequency in radians per step

For a rotation matrix with angle $\omega$, the eigenvalues are
$\lambda_{1,2} = e^{\pm i\omega}$:

$$
|\lambda| = 1 \quad \text{(neutrally stable)}, \qquad
\angle\lambda = \pm\omega
$$

```python
print("Eigenvalues:")
for i, ev in enumerate(evals):
    print(f"  lambda_{i}: {ev:.4f}")
    print(f"    |lambda| = {jnp.abs(ev):.4f}")
    print(f"    angle    = {jnp.angle(ev):.4f} rad "
          f"({jnp.degrees(jnp.angle(ev)):.1f} deg)")
```

Verify against the expected eigenvalues $e^{\pm i\pi/4}$:

```python
expected_evals = jnp.array([jnp.exp(1j * theta), jnp.exp(-1j * theta)])

# Sort by phase angle for stable comparison
evals_sorted = jnp.sort(jnp.angle(evals))
expected_sorted = jnp.sort(jnp.angle(expected_evals))

assert jnp.allclose(evals_sorted, expected_sorted, atol=1e-5)
print(f"\nExpected angles: {expected_sorted}")
print(f"Estimated angles: {evals_sorted}")
```

### Converting to continuous-time frequencies

If snapshots are separated by time interval $\Delta t$, the continuous-time
frequency and decay rate are:

$$
\omega_c = \frac{\angle\lambda}{\Delta t}, \qquad
\gamma = \frac{\ln|\lambda|}{\Delta t}
$$

This conversion is essential when applying DMD to neural time-series sampled
at a known rate.

---

## 5. Multi-Step Prediction

Once the Koopman operator is estimated, $k$-step prediction is simply matrix
exponentiation:

$$
\mathbf{x}_{t+k} = \mathbf{K}^k \, \mathbf{x}_t
$$

The `KoopmanEstimator.predict()` method implements this:

```python
x0 = jnp.array([1.0, 0.0])

# Predict 10 steps ahead using a smaller rotation for a clearer trajectory
theta_small = 0.1
K_small = jnp.array([
    [jnp.cos(theta_small), -jnp.sin(theta_small)],
    [jnp.sin(theta_small),  jnp.cos(theta_small)],
])

estimator = KoopmanEstimator()
pred = estimator.predict(x0, t=10, K=K_small)

# Compare to ground truth via matrix power
K10 = jnp.linalg.matrix_power(K_small, 10)
true_pred = K10 @ x0

print(f"Predicted state at t=10: {pred}")
print(f"True state at t=10:      {true_pred}")
assert jnp.allclose(pred, true_pred)
```

The `predict()` method signature is:

```
KoopmanEstimator.predict(
    x0: jnp.ndarray,      # (n,) initial state vector
    t: int,                # number of steps to predict ahead
    K: jnp.ndarray,       # (n, n) Koopman operator
) -> jnp.ndarray           # (n,) predicted state at step t
```

For generating full trajectories, simply iterate:

```python
def predict_trajectory(x0, K, n_steps):
    """Generate a full trajectory using the Koopman operator."""
    trajectory = [x0]
    x = x0
    for _ in range(n_steps):
        x = K @ x
        trajectory.append(x)
    return jnp.stack(trajectory)

traj = predict_trajectory(x0, K_small, n_steps=50)
print(f"Trajectory shape: {traj.shape}")  # (51, 2)
```

---

## 6. Comparison with Dynamic Mode Decomposition

The `KoopmanEstimator` in NeuroJAX *is* a DMD implementation.  The Koopman
operator and DMD are closely related:

| Aspect | Koopman Operator | DMD |
|--------|-----------------|-----|
| Nature | Infinite-dimensional linear operator | Finite-dimensional matrix approximation |
| Theory | Exact for any nonlinear system | Best rank-$r$ linear approximation |
| Input | Observable functions | Raw state snapshots |
| Output | Spectral decomposition in observable space | Spatial modes + eigenvalues |

DMD (Schmid, 2010) is the most common *numerical method* for approximating the
Koopman operator from data.  The rank parameter in `KoopmanEstimator` controls
the trade-off between approximation fidelity and noise robustness:

```python
# Full-rank DMD: captures all dynamics, but also all noise
est_full = KoopmanEstimator(rank=0)
K_full, evals_full, modes_full = est_full.fit(X, Y)

# Rank-truncated DMD: retains only the dominant modes
est_trunc = KoopmanEstimator(rank=2)
K_trunc, evals_trunc, modes_trunc = est_trunc.fit(X, Y)
```

For noisy data, rank truncation acts as a regulariser, discarding singular
values associated with measurement noise.  The choice of rank can be guided by
the singular value spectrum -- look for a gap between signal-dominated and
noise-dominated singular values.

NeuroJAX also provides `windowed_dmd` for tracking how the dominant DMD
eigenvalues (and hence oscillation frequencies) change over time:

```python
from neurojax.dynamics import windowed_dmd

result = windowed_dmd(
    X_timeseries,       # (n_features, n_timepoints)
    window_size=200,
    step_size=20,
    rank=5,
)
# result.eigenvalues: dominant eigenvalues per window
# result.frequencies: continuous-time frequencies per window
```

---

## 7. Applications to Whole-Brain Dynamics

The Koopman/DMD framework is naturally suited to analysing oscillatory neural
dynamics, where the goal is to extract dominant spatial modes and their
associated frequencies from high-dimensional recordings.

### Resting-state MEG/EEG

Source-reconstructed resting-state data can be decomposed into DMD modes that
correspond to known resting-state networks.  Each mode has an associated
eigenvalue whose phase encodes the oscillation frequency:

```python
from neurojax.dynamics import KoopmanEstimator

# X_sources: (n_regions, n_timepoints) source-reconstructed parcellated data
# Split into snapshot pairs
X_snap = X_sources[:, :-1]  # all but last
Y_snap = X_sources[:, 1:]   # all but first

est = KoopmanEstimator(rank=10)  # keep top 10 modes
K_brain, evals_brain, modes_brain = est.fit(X_snap, Y_snap)

# Convert eigenvalue phases to frequencies (Hz)
sfreq = 250.0  # sampling frequency
dt = 1.0 / sfreq
freqs = jnp.angle(evals_brain) / (2 * jnp.pi * dt)
decay_rates = jnp.log(jnp.abs(evals_brain)) / dt

print("Dominant DMD frequencies (Hz):")
for i, (f, d) in enumerate(zip(freqs, decay_rates)):
    print(f"  Mode {i}: {jnp.abs(f):.1f} Hz, decay = {d:.3f} s^-1")
```

Modes with frequencies in the alpha band (8--13 Hz) and beta band (13--30 Hz)
typically correspond to the dominant resting-state oscillations.

### Tracking oscillatory mode changes

Brain dynamics are non-stationary: oscillatory modes wax and wane as the brain
transitions between functional states.  `windowed_dmd` tracks these transitions
by applying DMD in sliding windows:

```python
from neurojax.dynamics import windowed_dmd

result = windowed_dmd(
    X_sources,
    window_size=500,    # 2 seconds at 250 Hz
    step_size=50,       # 200 ms steps
    rank=5,
)

# Identify windows where alpha-band modes dominate
alpha_mask = (jnp.abs(result.frequencies) > 8) & (jnp.abs(result.frequencies) < 13)
```

This provides a complementary, physics-informed view alongside statistical
state-segmentation methods (HMM, DyNeMo).

### Connection to neural mass models

The eigenvalues of the Koopman operator linearised around a fixed point of a
neural mass model (e.g., Jansen-Rit) correspond to the model's resonant
frequencies and stability properties.  By comparing DMD eigenvalues from data
with those predicted by a fitted neural mass model, you can validate biophysical
models against empirical dynamics.

---

## References

- Schmid, P. J. (2010). Dynamic mode decomposition of numerical and
  experimental data. *Journal of Fluid Mechanics*, 656, 5-28.
  [doi:10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217)

- Brunton, S. L., Budisic, M., Kaiser, E., & Kutz, J. N. (2021). Modern
  Koopman theory for dynamical systems. *SIAM Review*, 64(2), 229-340.
  [doi:10.1137/21M1401243](https://doi.org/10.1137/21M1401243)

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing
  equations from data by sparse identification of nonlinear dynamical systems.
  *Proceedings of the National Academy of Sciences*, 113(15), 3932-3937.
  [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

- Kunert-Graf, J. M., Eschenburg, K. M., Galas, D. J., Kutz, J. N.,
  Rane, S. D., & Brunton, B. W. (2019). Extracting reproducible time-resolved
  resting state networks using dynamic mode decomposition. *Frontiers in
  Computational Neuroscience*, 13, 75.
