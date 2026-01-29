# NeuroJAX Contribution Guidelines

NeuroJAX (OSL-JAX) is a modular, GPU-accelerated port of the OSL analysis stack using JAX. It leverages the "Kidger scientific stack" to provide robust, differentiable, and fast electrophysiology analysis tools.

To contribute your work to NeuroJAX, we ask that you adhere to these guidelines.

## Prerequisite Reading (The "Kidger Stack")

Development relies on a set of high-quality libraries built on top of JAX:

1.  **JAX**: The core engine for composable transformations.
2.  **Equinox**: For building stateful, object-oriented models safely within JAX.
3.  **Optimistix**: For non-linear least squares (NLLS) fitting and root finding.
4.  **Lineax**: For efficient linear solvers.
5.  **Jaxtyping**: For type annotations and shape checking.

## Usage of `uv`

We use `uv` for dependency management. Please ensure you have it installed.

```bash
uv sync
```

## Developing New Models

Models should be implemented as `equinox.Module` classes to ensure they are valid PyTrees and JIT-compatible.

## Pull Requests

Please open a pull request on the GitHub repository for any contributions. Ensure tests pass before submitting.
