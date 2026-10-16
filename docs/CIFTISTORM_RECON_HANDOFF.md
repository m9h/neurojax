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

## Step 4 landed (2026-10-15, neurojax session)

`src/neurojax/source/leadfield.py` + `src/neurojax/source/tests/test_leadfield.py`
(8 tests; run with `JAX_PLATFORMS=cpu` from the sandbox — the CUDA plugin cannot
see the GB10 there, Slurm jobs can):

- `Leadfield(gain_free, normals, positions, depth, tilt)` — `eqx.Module`; `gain()`
  is the fixed-orientation, depth-corrected gain (column scale `||g||^-depth`,
  mean-normalised), `orientations()` = `normalize(normals + tilt)`. `depth` and
  `tilt` are trainable, the raw gain and normals are `stop_gradient`ed geometry.
  `operator()` returns `lineax.MatrixLinearOperator` (verified with
  `lx.linear_solve(..., AutoLinearSolver(well_posed=False))`).
- `Leadfield.from_mne_forward(fwd)` — free- or fixed-orientation `mne.Forward`
  (so OpenMEEG via `mne.make_bem_solution(solver="openmeeg")` once the package is
  installed; it is in pyproject but not in `.venv-models`). Normals come from
  `fwd["src"][k]["nn"][vertno]`, not `source_nn` (identity frame for free ori).
- `leadfield_qc(gain)` — singular values, condition number, effective rank at a
  *reported* `rank_tol`, column norms, robust-z depth outliers. No reciprocity
  check: it needs the BEM, not the gain matrix — left out rather than faked.
- `point_spread_width(W, gain, positions)` — PSF spatial dispersion in metres
  (Molins 2008 / Hauk 2011) on `minimum_norm.resolution_matrix`.
- `lineax 0.1.1` pip-installed into `.venv-models`.

## Step 5 ran end to end on WAND sub-01187 (2026-10-15)

`scripts/run_wand_recon_spectral.py` (sbatch on the GB10): CTF resting MEG (273 ch,
120 s, 250 Hz) → sphere forward on the individual oct6 source space (7238 sources)
→ `Leadfield(depth=0.8)` + QC → aparc parcel gain (273×68, mean column per label,
rescaled to unit mean column norm, scale 4.21e-6 saved) → 60 × 2 s multitaper CSD
(bandwidth 2 Hz, 1–45 Hz in 1 Hz steps) → `higgs_spectral(alpha=0.05, n_iter=20)`
(2.1 s on GPU for 45 freqs) → coherence / imaginary coherency / partial coherence
+ per-frequency HIGGS leakage → `~/wand_recon_spectral_sub-01187.npz`.

Numbers: vertex gain cond 9.9e6, rank 109/273 (sphere MEG), 207 depth outliers
(deep/medial sources); parcel gain cond 8e3, full rank. Alpha-band (8–12 Hz)
precision edge density 0.80, mean leakage 0.12. Top alpha partial-coherence
pairs are homologous midline pairs with low leakage: paracentral L–R 0.81
[leak 0.02], caudal anterior cingulate L–R 0.63 [0.05], posterior cingulate L –
paracentral R 0.60 [0.10]. Alpha sweep (CPU, 8–12 Hz): edge density 0.81 /
0.64 / 0.53 / 0.44 / 0.35 for alpha 0.05 / 0.2 / 0.5 / 1 / 2, EM nll monotone
for alpha ≤ 2. No alpha selection criterion yet (BIC/EBIC on the hgLASSO is the
obvious next step).

Two solver fixes this needed (both in `higgs_spectral`, tested):

- **Scale.** The EM runs on each frequency's CSD normalised to unit mean sensor
  power and scales outputs back (`test_scale_equivariant_in_sensor_power`). On
  the raw MEG CSD (~1.7e-4 per channel) the unit prior / absolute alpha / jitter
  made the EM oscillate (nll ±3000 per iteration, noise variances 10⁵× sensor
  power); on the raw gain (~1e-6 T/Am) the sources never explained anything
  (zero edges).
- **GPU precision.** `jax.default_matmul_precision("highest")` around the EM.
  With the TF32 default, identical GPU runs gave edge density 0.88 vs 0.94 and
  different top pairs; with full precision GPU runs are identical to each other
  and to CPU.

## Steps 6, 7 and EBIC alpha selection (2026-10-15, same session)

- **EBIC.** `higgs_select_alpha(csd, G, alphas, n_eff, gamma=0.5, **kw)` runs the
  EM per alpha and scores `2 n_eff nll + k log n_eff + 4 γ k log P` (Foygel &
  Drton 2010; `k` = non-zero source edges over frequencies, `n_eff` = windows ×
  tapers). On the synthetic fixture it picks alpha=0.05 → 5 edges (4 planted + 1).
  The driver now sweeps `[0.05, 0.2, 0.5, 1, 2, 5]` with `n_eff = 60 × 3` and
  prints the table; the chosen alpha is saved in the npz.
- **Step 6, full noise GGM.** `higgs_spectral(..., noise_alpha=<float>)` replaces
  the diagonal noise M-step with an hgLASSO on the expected residual
  cross-spectrum (penalty relative to the residual's mean power, diagonal floored
  at `noise_floor`); `noise_precision` (F, M, M) is returned either way. Test
  with nearest-neighbour-coupled sensor noise: the diagonal model has zero
  off-diagonal noise by construction, the GGM ranks the planted couplings on
  top and fits at least as well; planted source edges survive. On WAND
  sub-01187 (`NOISE_ALPHA=0.05` env → `~/wand_recon_spectral_sub-01187_noiseggm.npz`):
  176 s on the GPU for the 6-alpha sweep (vs 9 s diagonal), EBIC prefers it
  (best EBIC −6.03e6 vs −4.53e6 diagonal), and it selects a far sparser source
  graph: alpha 0.5, edge density 0.047 (alpha-band 0.09) vs the diagonal
  model's alpha 2, density 0.26 (alpha-band 0.35). The top alpha partial-
  coherence pairs are the same homologous midline pairs under both noise
  models (paracentral L–R 0.84 / 0.72, caudal anterior cingulate L–R 0.75 /
  0.62, rostral anterior cingulate L–R 0.81 / 0.49, all leakage ≤ 0.2), so the
  correlated-noise model mostly removes the long tail of weak edges.
- **Step 7, CIFTI connectome writers.** `io/cifti.py`: `save_pconn(path, (P,P) or
  (F,P,P), parcels=[(name, {structure: vertex ids})], n_mesh_vertices)` with the
  `ConnParcels` intent (a stack writes one file per slice), `load_pconn`, and
  `save_dconn(path, (G,G), series)` with `ConnDense`. 4 tests in
  `tests/test_cifti_io.py`. The driver writes alpha-band coherence / |imaginary
  coherency| / partial coherence pconns to
  `/data/raw/wand/derivatives/neurojax-meg/<sub>/ses-01/` on the subject's own
  FreeSurfer aparc vertices — **not fs_LR**: resampling the individual white
  surface to fs_LR 32k (`wb_command -surface-resample` / `-label-resample` via
  Ciftify-style spheres) is the missing piece for the HCP contract.

## Remaining steps (the plan Morgan approved)

4. ~~OpenMEEG leadfield import as a Lineax operator~~ — done, see above. Still
   open: a DUNEuro/FEM path, and the reciprocity QC.
5. ~~End to end on one WAND subject~~ — done (`scripts/run_wand_recon_spectral.py`,
   sub-01187, both noise models). Open: fold it into `pipeline/recon.py` as a
   function (the script is still a flat driver like `run_wand_recon.py`), run the
   cohort, and replace the sphere with the BEM/OpenMEEG forward.
6. ~~Full hgLASSO noise GGM~~ — done (`noise_alpha`).
7. ~~CIFTI pconn / dconn writers~~ — done; fs_LR resampling of the individual
   surface still missing (see above). The harmonics-on-tract-graph part of step
   7 is untouched.

Original wording of 5–7 for reference:

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
