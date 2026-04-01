# Discovering Dynamical Systems from Data with SINDy

This tutorial demonstrates how to use the **Sparse Identification of Nonlinear
Dynamics (SINDy)** algorithm to recover governing equations directly from
time-series data.  We walk through the complete workflow using the Lorenz
system as a benchmark, then discuss applications to neural dynamics.

```{contents}
:depth: 3
:local:
```

## Prerequisites

Install NeuroJAX with its dynamics dependencies:

```bash
uv sync --extra dynamics
```

You need JAX, Diffrax (for ODE integration), and the `jaxctrl` backend that
NeuroJAX wraps:

```python
import jax
import jax.numpy as jnp
import diffrax
from neurojax.dynamics.sindy import SINDyOptimizer, polynomial_library

print(jax.devices())  # should list at least one device
```

---

## 1. Introduction to Data-Driven Dynamics Discovery

Classical physics derives governing equations from first principles.  In many
complex systems -- turbulence, neuroscience, ecology -- the equations are
unknown or intractable.  **SINDy** (Brunton, Proctor & Kutz, 2016) flips the
script: given measurements of a system's state $\mathbf{x}(t)$ and its
time derivatives $\dot{\mathbf{x}}(t)$, it discovers the *sparsest* set of
nonlinear terms that explain the dynamics.

The key equation is:

$$
\dot{\mathbf{X}} = \boldsymbol{\Theta}(\mathbf{X})\,\boldsymbol{\Xi}
$$

where:

- $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the state-measurement matrix
  ($m$ time points, $n$ state variables),
- $\dot{\mathbf{X}} \in \mathbb{R}^{m \times n}$ contains the corresponding
  time derivatives,
- $\boldsymbol{\Theta}(\mathbf{X}) \in \mathbb{R}^{m \times p}$ is a
  **library** of candidate nonlinear functions (polynomials, trigonometric
  terms, etc.),
- $\boldsymbol{\Xi} \in \mathbb{R}^{p \times n}$ is a sparse coefficient
  matrix that selects the active terms.

SINDy finds $\boldsymbol{\Xi}$ via **Sequentially Thresholded Least Squares
(STLS)**: iteratively solve a least-squares problem then zero out coefficients
below a sparsity threshold.

---

## 2. Generating Trajectory Data from the Lorenz System

The Lorenz system is a canonical chaotic attractor defined by:

$$
\begin{aligned}
\dot{x} &= \sigma(y - x) \\
\dot{y} &= x(\rho - z) - y \\
\dot{z} &= xy - \beta z
\end{aligned}
$$

with the standard parameters $\sigma = 10$, $\rho = 28$, $\beta = 8/3$.

We define the system as a Diffrax-compatible vector field and integrate it
using the 5th-order Tsitouras solver:

```python
def lorenz_system(t, y, args):
    """Lorenz 63 vector field for Diffrax."""
    sigma, rho, beta = args
    x, y, z = y
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.stack([dx, dy, dz])

# Ground-truth parameters
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
args = (sigma, rho, beta)

# Initial condition and time span
y0 = jnp.array([1.0, 1.0, 1.0])
t0, t1 = 0.0, 2.0
dt = 0.01

# Integrate with Diffrax
term = diffrax.ODETerm(lorenz_system)
solver = diffrax.Tsit5()
saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))

sol = diffrax.diffeqsolve(
    term, solver, t0, t1, dt, y0, args=args, saveat=saveat
)
X = sol.ys  # shape: (200, 3) — state measurements
```

We also need the time derivatives at each measurement point.  Since we know the
true vector field here, we evaluate it directly (in practice you would use
numerical differentiation):

```python
# Compute true derivatives via vmap over the vector field
dX = jax.vmap(lambda x: lorenz_system(0, x, args))(X)
# dX shape: (200, 3) — derivative measurements
```

The result is a pair of matrices `(X, dX)` ready for SINDy.

---

## 3. Building the Polynomial Feature Library

The library $\boldsymbol{\Theta}(\mathbf{X})$ defines what functional forms
SINDy can discover.  A **polynomial library of degree 2** for three state
variables $(x, y, z)$ includes:

$$
\boldsymbol{\Theta}(\mathbf{X}) = \begin{bmatrix}
1 & x & y & z & x^2 & xy & xz & y^2 & yz & z^2
\end{bmatrix}
$$

That is 10 candidate terms per equation.  The `polynomial_library` function
from `neurojax.dynamics` constructs this matrix:

```python
from neurojax.dynamics import polynomial_library

# Evaluate the library at all data points
Theta = polynomial_library(X, degree=2)
print(f"Library shape: {Theta.shape}")
# Expected: (200, 10)
```

The column ordering is:

| Index | 0 | 1 | 2 | 3 | 4   | 5  | 6  | 7   | 8  | 9   |
|-------|---|---|---|---|-----|----|----|-----|----|-----|
| Term  | 1 | x | y | z | x^2 | xy | xz | y^2 | yz | z^2 |

This ordering is important for interpreting the coefficient matrix
$\boldsymbol{\Xi}$ that SINDy returns.

---

## 4. Sparse Regression with `SINDyOptimizer.fit()`

The `SINDyOptimizer` implements the STLS algorithm.  Two key hyperparameters
control the sparsity--accuracy tradeoff:

- **`threshold`**: coefficients with magnitude below this value are zeroed out
  at each iteration.  Higher values produce sparser models.
- **`max_iter`**: maximum number of STLS iterations.

```python
from neurojax.dynamics.sindy import SINDyOptimizer

optimizer = SINDyOptimizer(threshold=0.5, max_iter=20)

# Fit: pass data, derivatives, and a library function
Xi = optimizer.fit(X, dX, lambda x: polynomial_library(x, degree=2))
```

The returned `Xi` has shape `(p, n)` -- here `(10, 3)` -- where each column
holds the coefficients for one equation ($\dot{x}$, $\dot{y}$, $\dot{z}$).

The `fit()` method signature is:

```
SINDyOptimizer.fit(
    X: jnp.ndarray,       # (m, n) state measurements
    dX: jnp.ndarray,      # (m, n) derivative measurements
    lib_fn: Callable,     # X -> Theta(X), the library function
) -> jnp.ndarray          # (p, n) sparse coefficient matrix Xi
```

---

## 5. Recovering the Governing Equations

Now we compare the discovered coefficient matrix `Xi` against the ground-truth
Lorenz equations.

### Equation 1: $\dot{x} = \sigma(y - x)$

The first column of `Xi` should show $\sigma = 10$ at the $y$ index and
$-\sigma = -10$ at the $x$ index, with all other entries near zero:

```python
# Column 0 of Xi corresponds to dx/dt
print("dx/dt coefficients:")
print(f"  Coeff of y (idx 2): {Xi[2, 0]:.2f}  (expected  10.0)")
print(f"  Coeff of x (idx 1): {Xi[1, 0]:.2f}  (expected -10.0)")

assert jnp.isclose(Xi[2, 0], 10.0, atol=1.5)
assert jnp.isclose(Xi[1, 0], -10.0, atol=1.5)
```

### Equation 2: $\dot{y} = x(\rho - z) - y = \rho x - y - xz$

```python
# Column 1 of Xi corresponds to dy/dt
print("dy/dt coefficients:")
print(f"  Coeff of x  (idx 1): {Xi[1, 1]:.2f}  (expected  28.0)")
print(f"  Coeff of y  (idx 2): {Xi[2, 1]:.2f}  (expected  -1.0)")
print(f"  Coeff of xz (idx 6): {Xi[6, 1]:.2f}  (expected  -1.0)")

assert jnp.isclose(Xi[1, 1], rho, atol=1.5)
assert jnp.isclose(Xi[2, 1], -1.0, atol=1.5)
assert jnp.isclose(Xi[6, 1], -1.0, atol=1.5)
```

### Equation 3: $\dot{z} = xy - \beta z$

```python
# Column 2 of Xi corresponds to dz/dt
print("dz/dt coefficients:")
print(f"  Coeff of xy (idx 5): {Xi[5, 2]:.2f}  (expected   1.0)")
print(f"  Coeff of z  (idx 3): {Xi[3, 2]:.2f}  (expected  -2.67)")

assert jnp.isclose(Xi[5, 2], 1.0, atol=1.5)
assert jnp.isclose(Xi[3, 2], -beta, atol=1.5)
```

### Sparsity check

The Lorenz system has exactly 7 active terms across all three equations
(2 + 3 + 2).  A well-tuned SINDy should zero out all the rest:

```python
non_zeros = jnp.sum(jnp.abs(Xi) > 0.1)
print(f"Active terms: {int(non_zeros)} (expected <= 10)")
assert non_zeros <= 10
```

---

## 6. Prediction: Simulating the Recovered System Forward

Once SINDy has identified the coefficient matrix $\boldsymbol{\Xi}$, you can
build a **surrogate ODE** and simulate it forward using the same Diffrax
infrastructure:

```python
def sindy_rhs(t, y, args):
    """Surrogate vector field from discovered SINDy coefficients."""
    Xi = args
    # Evaluate the polynomial library at the current state
    # y has shape (3,), need to reshape for polynomial_library
    theta = polynomial_library(y[None, :], degree=2)  # (1, 10)
    # Compute derivatives: theta @ Xi -> (1, 3), squeeze to (3,)
    return (theta @ Xi).squeeze()

# Simulate the discovered system from the same initial condition
term_discovered = diffrax.ODETerm(sindy_rhs)
sol_discovered = diffrax.diffeqsolve(
    term_discovered, solver, t0, t1, dt, y0, args=Xi, saveat=saveat
)
X_pred = sol_discovered.ys

# Compare trajectories
error = jnp.mean(jnp.abs(X_pred - X))
print(f"Mean absolute prediction error: {error:.4f}")
```

For the Lorenz system with clean data and well-tuned threshold, the prediction
error should be small over the integration window.  Note that chaotic systems
will diverge over long horizons regardless of model accuracy -- this is a
fundamental property of sensitive dependence on initial conditions.

---

## 7. Applications to Neural Dynamics

SINDy is particularly well-suited for discovering **effective dynamics** from
neural time-series.  In the neuroscience context:

### Source-reconstructed MEG/EEG

After source reconstruction (e.g., via NeuroJAX's `source.beamformer` or
`source.minimum_norm`), you have time-series at parcellated brain regions.
SINDy can discover how these regions interact:

```python
from neurojax.dynamics import SINDyOptimizer, polynomial_library, windowed_sindy

# X_sources: (n_timepoints, n_regions) from source reconstruction
# dX_sources: numerical derivative (e.g., finite differences)
dX_sources = jnp.gradient(X_sources, dt, axis=0)

# Discover region-level interaction equations
optimizer = SINDyOptimizer(threshold=0.1, max_iter=30)
Xi_brain = optimizer.fit(
    X_sources, dX_sources,
    lambda x: polynomial_library(x, degree=2)
)
```

### Windowed SINDy for non-stationary dynamics

Brain dynamics are typically non-stationary -- the system switches between
different regimes (e.g., task vs. rest, pre-ictal vs. ictal).
`windowed_sindy` applies SINDy in sliding windows and tracks how the
governing Jacobian eigenvalues evolve:

```python
from neurojax.dynamics import windowed_sindy

result = windowed_sindy(
    X_sources, dX_sources,
    window_size=100,
    step_size=10,
    degree=2,
    threshold=0.1,
)
# result.eigenvalues: (n_windows, n_regions) Jacobian eigenvalues over time
# result.coefficients: (n_windows, p, n_regions) time-varying Xi
```

Change points in the eigenvalue trajectories can be compared against HMM or
DyNeMo state transitions, providing an interpretable dynamical complement to
purely data-driven state segmentation.

### Whole-brain neural mass models

SINDy can also be used to identify effective parameters in neural mass models
(e.g., Jansen-Rit, Wilson-Cowan) from simulated or recorded data, bridging
the gap between biophysical models and observed dynamics.

---

## References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing
  equations from data by sparse identification of nonlinear dynamical systems.
  *Proceedings of the National Academy of Sciences*, 113(15), 3932-3937.
  [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

- Kaiser, E., Kutz, J. N., & Brunton, S. L. (2018). Sparse identification of
  nonlinear dynamics for model predictive control in the low-data limit.
  *Proceedings of the Royal Society A*, 474(2219), 20180335.

- Mangan, N. M., Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).
  Inferring biological networks by sparse identification of nonlinear dynamics.
  *IEEE Transactions on Molecular, Biological, and Multi-Scale Communications*,
  2(1), 52-63.
