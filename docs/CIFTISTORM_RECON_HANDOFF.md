# Handoff: CiftiStorm-equivalent EMEG Recon → the neurojax agent

**From:** the hippy-feat session (2026-10-05). **Decision (Morgan):** the whole
CiftiStorm-equivalent (structural → forward → Bayesian spectral source imaging →
CIFTI connectomes) lives in neurojax. hippy-feat keeps only fMRI-specific
harmonic metrics and the drug-dataset audits.

CiftiStorm reference: Valdés-Sosa et al., *Front. Neurosci.* 2024 (PMC11047451).
Three modules: (1) structural (HCP/Ciftify surfaces), (2) forward model + leadfield
QC + geometric corrections, (3) Bayesian frequency-domain ESI estimating the source
cross-spectrum (HIGGS-style) → HCP/megconnectome CIFTI connectomes.

## What landed today (all committed on `main`)

| Commit | What | Where |
|---|---|---|
| `6ef9eab` | CIFTI-2 I/O on the fs_LR grayordinate contract: `load_dtseries`, `surface_vertex_mask`, `cortex_to_surface`, `save_dscalar` | `src/neurojax/io/cifti.py`, `tests/test_cifti_io.py` |
| `6ef9eab` | Surface-scale harmonic basis: `mesh_edges`, `connectome_adjacency` (Atasoy mesh + tract graph, scipy.sparse), Lanczos path inside `connectome_harmonics` for sparse input, CHD metrics `harmonic_energy` (λ²-weighted), `log_binned_spectrum`, `repertoire_entropy`. Dense/differentiable path unchanged. | `src/neurojax/spatial/harmonics.py`, `tests/test_harmonics.py` |
| `a36452e` | HADES variant (Vohryzek 2024, github.com/jvohryzek/HADES): `functional_harmonics_adjacency` (row-median + kNN-300 on dense FC), `cosine_projection`, `harmonic_contributions`, `winner_takes_all` / `fractional_occupancy` / `lifetimes` / `transition_matrix` (LEiDA rules), `latent_spread`. Mirrored in hippy-feat `jaxoccoli/harmonics.py` (e59974d). | `src/neurojax/spatial/harmonics.py`, `tests/test_hades.py` |
| `27856c2` | `higgs_spectral(csd, leadfield, alpha, n_iter)`: per-frequency HIGGS EM on sensor cross-spectra as ONE jitted call (vmap over freqs, lax.scan over EM, grad flows through `alpha` + leadfield). `hermitian_glasso` made jit/vmap-safe (ADMM ρ is now traced). | `src/neurojax/source/higgs.py`, `src/neurojax/source/tests/test_higgs_spectral.py` |

Verified end to end on real data: a ds006072 CIFTI (`/data/datasets/ds006072/NON_BIDS/ciftis/`)
+ templateflow fs_LR 32k midthickness → 59,412 grayordinates, 0 isolated vertices after
the medial-wall mask, 50 modes in 9.6 s CPU.

Run tests with: `PYTHONPATH=src .venv-models/bin/python -m pytest <files> -o addopts=""`
(the default `.venv` has no jax; `.venv-models` now also has `jraph`, which I pip-installed
because `neurojax.source` imports it at package level).

## Pre-existing failures (NOT from today's commits — verified with the changes stashed)

- `src/neurojax/source/tests/test_higgs.py`: 3 failures in the time-domain `higgs_em`
  recovery tests (`test_source_locations_recovered`, `test_nonzero_edges_recovered`,
  `test_higgs_lower_error_than_two_step`).
- `tests/test_spatial.py`: 13 errors were the missing `jraph`; re-check after the install.

## Remaining steps (the plan Morgan approved)

4. **OpenMEEG (or DUNEuro) leadfield import as a Lineax linear operator**, with the
   CiftiStorm geometric corrections (source depth / orientation) as differentiable
   parameters, and the leadfield QC (condition number, reciprocity, point-spread) —
   `src/neurojax/bench/monitors/leadfield.py` has a start.
5. **`src/neurojax/pipeline/recon.py` end to end on ONE WAND subject** (MEG + MRI +
   FreeSurfer recons exist: `/data/raw/wand/derivatives/freesurfer`, see
   `scripts/wand_processing/`). Stages: structural → leadfield (step 4) → multitaper
   sensor CSD (`neurojax.analysis.multitaper`) → `higgs_spectral` → source CSD →
   coherence / imaginary coherence / partial coherence → CIFTI `pconn` (writer not yet
   written; `save_dscalar` shows the nibabel axes pattern).
6. **Full hgLASSO noise GGM** in `higgs_spectral` (noise is kept diagonal today for
   stability — the HIGGS paper estimates a full noise precision too).
7. **CIFTI pconn / dconn writers** (`io/cifti.py`), then the harmonics on the WAND
   structural connectome via `connectome_adjacency(..., tract_edges=...)` — note
   `scripts/real_data/wand_connectome_harmonics.py` already builds a Desikan-68
   structural graph with the dense path; the sparse path is for vertex-level graphs.

## Cross-repo contract

- hippy-feat `jaxoccoli/harmonics.py` carries a self-contained copy of the graph builder
  + metrics (jaxoccoli runs in the NGC SIF without neurojax). neurojax is canonical;
  keep the two in sync if the API changes.
- Open drug-fMRI data on the NAS for the harmonic audits (fMRI side, hippy-feat):
  `/data/datasets/ds003059` (LSD, preprocessed MNI), `/data/datasets/ds003171`
  (propofol, raw + DWI), `/data/datasets/ds006072` (psilocybin, processed CIFTIs +
  FD traces + DWI; raw multi-echo BOLD NOT pulled, 3.25 TB).
- fs_LR 32k meshes: `/home/mhough/.cache/templateflow/tpl-fsLR/tpl-fsLR_den-32k_hemi-{L,R}_midthickness.surf.gii`
  (both templateflow copies on this machine were 0-byte stubs; these are the real files).
