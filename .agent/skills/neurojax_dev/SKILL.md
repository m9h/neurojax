---
name: neurojax_dev
description: Guidelines for developing NeuroJAX (OSL-JAX) components.
---

# NeuroJAX (OSL-JAX) Development

## Philosophy
NeuroJAX follows the "Kidger Stack" philosophy:
- **State**: All models must be `equinox.Module`. State is explicit and immutable.
- **Solvers**: Use `lineax` for linear solves and `optimistix` for non-linear optimization.
- **Differentiation**: Everything must be differentiable. Avoid `numpy` (except for I/O); use `jax.numpy`.
- **Typing**: Use `jaxtyping` to enforce shapes, e.g., `Float[Array, "time sensors"]`.

## Directory Structure
- `src/neurojax/glm.py`: Mass-univariate statistics.
- `src/neurojax/inverse/`: Beamformers and source reconstruction.
- `src/neurojax/models/`: Biophysical and generative models (`Diffrax`, `Equinox`).
- `src/neurojax/utils/`: Bridges and helpers.

## Common Patterns

### GLM / Linear Solvers
When solving $Ax=b$, prefer `lineax` over `jnp.linalg.solve`:
```python
operator = lx.MatrixLinearOperator(A)
solution = lx.linear_solve(operator, b, solver=lx.QR())
```

### Random Keys
Passing `key` is mandatory for stochastic operations. Split keys early:
```python
keys = jax.random.split(key, num=100)
jax.vmap(func)(keys)
```
