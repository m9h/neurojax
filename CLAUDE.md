# CLAUDE.md — NeuroJAX Project Guide

## What is this project?

NeuroJAX is a differentiable, GPU-accelerated toolkit for multi-modal
neuroimaging analysis in JAX. It provides 15 source imaging methods,
differentiable FEM/BEM head modeling, quantitative MRI fitting, MRS
spectroscopy, and neural dynamics — all in a single computational graph
where `jax.grad` flows end-to-end.

## Tech stack

| Layer | Library | Purpose |
|-------|---------|---------|
| Neural nets | Equinox | Pytree modules, `eqx.Module` |
| Dynamics | Diffrax | Neural ODEs/SDEs |
| Optimisation | Optax | Adam, gradient clipping |
| Linear algebra | Lineax | GLM, beamforming |
| Graphs | jraph | GNN on cortical mesh |
| Forward models | OpenMEEG, JAX-FEM+PETSc | BEM/FEM |
| Data I/O | MNE-Python, nibabel, meshio | Sensors, volumes, meshes |

## Directory structure

```
src/neurojax/
  source/          — 15 inverse solvers (MNE, LAURA, VARETA, PI-GNN, ...)
  geometry/        — BEM, FEM, CHARM segmentation, surface I/O
  qmri/            — DESPOT, QMT, qBOLD, neural relaxometry (PINN/NODE)
  analysis/        — analytic signal, spectral, connectivity, GLM, SINDy
  preprocessing/   — ASR, ICA, filtering, artifact removal
  models/          — HMM, DyNeMo
  bench/           — leadfield forward, vbjax adapter, monitors
  io/              — WAND MEG, Connectome, UCI, Wakeman-Henson loaders
  spatial/         — EEG graph, spatial filters

tests/             — pytest, TDD red-green methodology
examples/          — 59 runnable scripts
docs/tutorials/    — 9 markdown tutorials
paper/             — Sweave (knitr + tectonic) manuscript
scripts/           — WAND processing pipeline (20+ shell scripts)
```

## Critical conventions

### Always do
- Use **red-green TDD**: write failing tests first, then implement
- Use **uv** for dependency management (`uv sync`, `uv add`, `uv run`)
- Use **tectonic** for LaTeX builds (not pdflatex)
- Report all SVD/regularisation choices explicitly — never hide thresholds
- Check memory before large matrix operations (`psutil.virtual_memory()`)
- Use Tikhonov regularisation, not truncated SVD with hard rank cutoff
- Keep complex signal processing (Hilbert/analytic) available at every stage
- Prefer complete biophysical models; solve optimisation after

### Never do
- Silent segfaults (always estimate and warn about memory requirements)
- Arbitrary SVD rank thresholds without reporting
- `np.trapz` (removed in numpy 2.x; use `np.trapezoid`)
- Hard-coded paths without fallback
- Commit `.env`, credentials, or large binary files

### Testing
```bash
uv run pytest tests/ -v --no-cov     # full suite
uv run pytest tests/test_source_gnn.py -v  # PI-GNN
uv run pytest tests/test_laura.py -v       # LAURA
uv run pytest tests/test_qmri.py -v        # qMRI
```

### Dependency management
```bash
uv add <package>          # add dependency
uv sync                   # install all
uv run <command>          # run in venv
```

## Research heritage

CAUCHY/SimBio (1993+) → OpenMEEG (INRIA Rennes) → Wendling neural mass
(LTSI Rennes) → Fijee (Cobigo, C++/FEniCS) → NeuroJAX (differentiable JAX).
Also: VARETA/HIGGS (Valdes-Sosa), OSL/osl-dynamics (Oxford), SCI head model
(Utah), Kidger stack. Validated on WAND (170-subject multi-modal, CUBRIC).

## Key anti-patterns to avoid

- **Premature abstraction**: Three similar lines > one premature helper
- **Over-engineering**: Don't add features, refactor, or "improve" beyond what's asked
- **Backwards-compat hacks**: Delete unused code, don't comment it out
- **Mock databases in tests**: Use real data paths with `skipif` for missing data
