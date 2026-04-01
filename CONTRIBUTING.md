# Contributing to NeuroJAX

NeuroJAX is a modular, GPU-accelerated JAX port of the OSL electrophysiology
analysis stack. It leverages the "Kidger scientific stack" (Equinox, Optimistix,
Lineax, Diffrax, jaxtyping) for differentiable, composable analysis of
MEG/EEG data.

We welcome contributions of all kinds: bug fixes, new analysis modules,
documentation improvements, and benchmark extensions. This guide covers the
conventions and workflow we follow.

---

## 1. Getting Started

### Prerequisites

- **Python 3.11 -- 3.12** (see `requires-python` in `pyproject.toml`)
- **uv** for all package management. Do not use `pip`, `conda`, or `poetry`.
  Install uv: <https://docs.astral.sh/uv/getting-started/installation/>
- **Git**

### Clone and install

```bash
git clone https://github.com/<org>/neurojax.git
cd neurojax
uv sync                          # install all deps + dev extras
uv sync --extra test --extra doc # explicit extras if needed
```

`uv sync` resolves everything from `pyproject.toml` and `uv.lock`. The project
is automatically installed in editable mode.

### Verify the installation

```bash
uv run pytest                    # run full test suite
uv run neurojax --help           # CLI smoke test
```

---

## 2. Development Workflow

### Branching

Create a branch from `main` using one of these prefixes:

| Prefix       | Purpose                              |
|--------------|--------------------------------------|
| `feature/`   | New analysis modules or capabilities |
| `fix/`       | Bug fixes                            |
| `docs/`      | Documentation-only changes           |
| `refactor/`  | Internal restructuring, no new API   |
| `bench/`     | Benchmark or performance work        |

Example: `feature/specparam-state-decomposition`, `fix/sf-table-stability`.

### Commits

Write concise commit messages in imperative mood. Follow the conventions visible
in the project history:

```
Add specparam 2.0 per-state spectral decomposition, 9 tests green
Fix Sf table numerical stability, test on WAND QMT data
```

- Start with a verb: `Add`, `Fix`, `Update`, `Refactor`, `Remove`
- Mention test status when relevant
- Keep the first line under 72 characters

### Pull requests

- One logical change per PR.
- Descriptive title. Use the body to explain *why*, not just *what*.
- Link related issues with `Closes #N` or `Relates to #N`.
- Ensure `pytest` passes and `ruff check` reports no errors before requesting review.
- Request review from at least one maintainer.

---

## 3. Code Style

### Formatting and linting

We use **Ruff** for linting and import sorting. The rules are configured in
`pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "F", "I"]
```

Run before committing:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Docstrings

Use **Google-style docstrings** (parsed by Napoleon/sphinx-autodoc-typehints):

```python
def compute_csd(epochs: mne.Epochs, fmin: float, fmax: float) -> jax.Array:
    """Compute cross-spectral density matrix for a frequency band.

    Parameters:
        epochs: MNE Epochs object with sensor-level data.
        fmin: Lower frequency bound in Hz.
        fmax: Upper frequency bound in Hz.

    Returns:
        Complex-valued CSD matrix of shape (n_channels, n_channels).

    Raises:
        ValueError: If fmin >= fmax.
    """
```

### Type annotations

- Use `jaxtyping` for array shape annotations:
  ```python
  from jaxtyping import Float, Array
  def beamform(W: Float[Array, "sources channels"],
               data: Float[Array, "channels time"]) -> Float[Array, "sources time"]:
  ```
- Annotate all public function signatures.
- Use `beartype` or `typeguard` runtime checks sparingly (only at module boundaries).

### Module conventions

- **Module-level docstrings** are required for all new files.
- **Equinox modules** (`eqx.Module`) for all differentiable, JIT-compatible
  objects. They are immutable pytrees -- use `eqx.tree_at` for updates, never
  `__setattr__`.
- Keep JAX transforms (`jit`, `vmap`, `grad`) at the call site or in thin
  wrapper functions, not buried inside class methods.
- Follow existing patterns in `src/neurojax/` -- look at neighbouring modules
  for conventions.

---

## 4. Testing

### Running tests

```bash
uv run pytest                              # full suite with coverage
uv run pytest tests/test_models.py -v      # single file
uv run pytest -k "test_beamform" -v        # by name
```

Coverage is reported automatically via `--cov=neurojax --cov-report=term-missing`
(configured in `pyproject.toml`).

### Test requirements

- **All new features need tests.** No exceptions.
- **All bug fixes need a regression test** that fails without the fix.
- Use `pytest` fixtures for shared setup (test data, MNE objects, JAX keys).
- Place test files in `tests/` with the naming pattern `test_<module>.py`.

### JAX-specific testing

JAX code needs targeted tests beyond standard unit tests:

- **JIT compilation**: verify that functions work under `jax.jit` (catches
  side effects and Python-mode-only logic).
- **vmap compatibility**: test that batched versions produce the same results
  as manual loops.
- **Gradient correctness**: use `jax.grad` or `jax.jacobian` on loss functions
  and check against finite differences (`jax.test_util.check_grads`).
- **Numerical stability**: test with `jax.config.update("jax_enable_x64", True)`
  when precision matters.
- **Determinism**: fix `jax.random.PRNGKey` seeds for reproducible tests.

---

## 5. Documentation

### Where docs live

```
docs/
├── conf.py                # Sphinx configuration
├── index.md               # Landing page
├── tutorials/             # MyST markdown tutorials
├── reference/             # API reference (auto-generated)
├── references.bib         # BibTeX bibliography
└── Makefile               # Build target
```

### Writing docs

- **Tutorials** go in `docs/tutorials/` as MyST markdown (`.md`) files.
- **API reference** is auto-generated via `sphinx-autodoc-typehints`. Write
  good docstrings and the API docs follow.
- **Cross-references** use MyST syntax: `` {func}`neurojax.glm.fit_glm` ``.
- **Citations**: add entries to `references.bib` and cite with
  `` {cite}`AuthorYear` ``.

### Build locally

```bash
cd docs && make html
open _build/html/index.html
```

---

## 6. AI-Assisted Development

We actively use AI coding agents (Claude Code, Copilot, etc.) and have
found them valuable for accelerating development. This section defines our
conventions for responsible AI-assisted contribution.

### CLAUDE.md conventions

Each project should have a `CLAUDE.md` at the repository root that orients
AI agents. This file should contain:

- **Project summary** -- what the project does, in 2-3 sentences.
- **Tech stack table** -- key libraries and why they are used.
- **Critical conventions** -- things an agent must not violate (e.g., "uv only",
  "Equinox immutability", unit conventions).
- **Directory map** -- one-level layout of the source tree with brief
  descriptions.
- **What NOT to do** -- explicit anti-patterns.

Keep `CLAUDE.md` up to date as the project evolves. It is not documentation for
users -- it is a machine-readable project briefing.

### Writing agent-friendly code

AI agents navigate code through docstrings, type hints, and naming. Help them
help you:

- Write **complete Google-style docstrings** on all public functions and classes.
- Use **jaxtyping shape annotations** so agents understand array semantics.
- Use **descriptive names** -- `compute_csd_matrix` not `csd`, `n_channels`
  not `nc`.
- Keep **module-level docstrings** that explain what the file is for.
- Avoid deeply nested logic; prefer small, composable functions.

### Reviewing AI-generated code

All AI-generated code must pass human review. Pay special attention to:

- **Hallucinated imports or APIs** -- agents may invent function names or use
  deprecated interfaces. Verify every import.
- **Unnecessary abstractions** -- agents sometimes over-engineer with extra
  classes or patterns. Prefer simplicity.
- **Security issues** -- watch for hardcoded paths, leaked credentials, or
  unsafe deserialization.
- **Test quality** -- agents write tests that pass but may not test meaningful
  behavior. Check that assertions are substantive, not tautological.
- **Scientific correctness** -- agents do not understand the physics. Verify
  formulas, units, and numerical ranges against published references.

### Commit attribution

When an AI agent contributed substantially to a commit, add a co-author trailer:

```
Add specparam per-state decomposition

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

This is informational, not a transfer of responsibility. The human committer
is accountable for the code.

### What agents do well

- Generating boilerplate (test scaffolding, docstrings, `__init__.py` exports)
- Refactoring and renaming across files
- Writing documentation and tutorial drafts
- Translating pseudocode or math into JAX implementations
- Code review checklists and static analysis

### What needs human oversight

- **Architecture decisions** -- module boundaries, API design, dependency choices
- **Performance-critical JAX transforms** -- custom `jit` partitioning, `vmap`
  axis management, `pmap` sharding strategies
- **Scientific correctness** -- model equations, signal processing pipelines,
  statistical methods
- **Security-sensitive code** -- authentication, file I/O, network access
- **MNE-specific patterns** -- MNE-Python has many implicit conventions that
  agents may not know

### PR labels

Tag pull requests that were substantially AI-assisted with the `ai-assisted`
label. This helps with auditing and lets reviewers know to apply extra scrutiny
to the areas listed above.

---

## 7. Scientific Standards

NeuroJAX implements published methods from the electrophysiology and neuroimaging
literature. We hold contributions to a high scientific bar.

### Citations

- Reference the original method paper in docstrings and module-level docs.
  Use the format: `See: Author et al. (Year), "Title", Journal. DOI: ...`
- Add BibTeX entries to `docs/references.bib`.
- If implementing a variant or extension, cite both the original and the
  modification.

### Mathematical notation

Include mathematical notation in docstrings where it aids understanding:

```python
def fit_ar_model(data: Float[Array, "T"], order: int) -> Float[Array, "order"]:
    """Fit an autoregressive model of given order.

    Solves the Yule-Walker equations:

        R * a = r

    where R is the Toeplitz autocorrelation matrix and r is the
    autocorrelation vector at lags 1..order.

    Parameters:
        data: Time series of length T.
        order: AR model order (number of coefficients).

    Returns:
        AR coefficients a_1, ..., a_p.

    See: Marple (1987), "Digital Spectral Analysis with Applications".
    """
```

### Validation

- Validate implementations against known benchmarks or reference
  implementations (e.g., MNE-Python, OSL, FieldTrip).
- Include validation scripts or notebooks in `examples/` or `docs/tutorials/`.
- Report numerical agreement (correlation, RMSE, max absolute error) in PR
  descriptions when introducing new methods.

---

## Prerequisite Reading (The "Kidger Stack")

Development relies on a set of high-quality libraries built on top of JAX.
Familiarize yourself with them before contributing:

1. **JAX** -- composable transformations (jit, vmap, grad): <https://jax.readthedocs.io>
2. **Equinox** -- pytree-based neural networks and models: <https://docs.kidger.site/equinox/>
3. **Optimistix** -- nonlinear least squares and root finding: <https://docs.kidger.site/optimistix/>
4. **Lineax** -- linear solvers: <https://docs.kidger.site/lineax/>
5. **Diffrax** -- differential equation solvers: <https://docs.kidger.site/diffrax/>
6. **jaxtyping** -- type annotations with shape checking: <https://docs.kidger.site/jaxtyping/>

---

Thank you for contributing to NeuroJAX. If you have questions, open an issue or
start a discussion on the repository.
