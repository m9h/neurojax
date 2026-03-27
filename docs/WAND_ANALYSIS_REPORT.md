# WAND Multi-Modal Analysis Report

## Dataset

**WAND** (Welsh Advanced Neuroimaging Database) — 170 healthy volunteers, CUBRIC Cardiff.
Ultra-strong gradient (300 mT/m) Connectom scanner + CTF 275-channel MEG + TMS-EEG.

**Reference subject:** sub-08033 (all modalities confirmed, downloaded locally: 10.24 GB)

| Session | Modalities | Key Data |
|---------|-----------|----------|
| ses-01 | MEG | CTF 275ch: resting, auditory-motor, MMN, simon, visual + headshape |
| ses-02 | DWI + anat | AxCaliber (4 shells, b=0–15500), CHARMED (AP+PA), T1w, VFA (SPGR/SSFP), QMT (16 volumes) |
| ses-03 | fMRI + anat | T1w, T2w (gradient-corrected), fieldmaps, perfusion (ASL) |
| ses-04 | MRS + anat | Magnetic resonance spectroscopy, T1w, fieldmaps |
| ses-05 | MRS + anat | Second MRS session, T1w |
| ses-06 | fMRI + anat | MP2RAGE, multi-echo GRE (7 echoes), T2w, fieldmaps |
| ses-08 | TMS | TMS-EEG SICI protocol (3 runs, .mat format) |

---

## Flagship Study: Unified TMS Intervention

### Core Thesis
Can we predict an individual's TMS response from their structural, functional, and metabolic brain architecture?

### Phase 1: Individual Brain Models
Build a digital twin per subject using all available modalities.

**Structural connectome:**
- FreeSurfer recon-all → cortical surfaces, parcellation
- DWI preprocessing (topup + eddy) → bedpostx → probtrackx2 → Cmat (streamline counts) + Lmat (fiber lengths)
- sbi4dwi AxCaliber/SANDI → axon diameter maps → ConnectomeMapper → microstructure-informed Dmat (conduction delays)

**Functional dynamics:**
- Resting MEG → source reconstruction → parcellated timeseries
- TDE + PCA (or TuckerTDE) → HMM/DyNeMo → baseline brain states
- State spectra → per-state power maps + coherence networks
- Summary statistics → fractional occupancy, lifetime, switching rate

**Metabolic context:**
- MRS → GABA and glutamate concentrations
- QMT → bound pool fraction (myelin content)
- T1w/T2w ratio → myelination proxy (validated against QMT)

**Status:** neurojax code complete (loaders, models, analysis). Processing scripts ready. Need to run on WAND data.

### Phase 2: TMS-EEG Fitting
Fit whole-brain model to each subject's TMS-evoked potentials.

- TMSProtocol → stimulus injection into JR/RWW neural mass model
- Balloon-Windkessel BOLD hemodynamics
- Leadfield forward projection → sensor-space TEP
- Multi-modal loss: FC + FCD + sensor MSE + TEP waveform + GFP
- Gradient-based optimization (optax Adam through full simulation)
- Per-region heterogeneous parameters (RegionalParameterSpace)

**Fitted parameters per subject:** global coupling G, per-region excitability, conduction velocity scaling

**Status:** Code complete. Need WAND TMS-EEG loader and subject-specific BEM.

### Phase 3: Virtual Lesion Analysis
Decompose TMS-evoked responses into local vs. network contributions (Momi et al. 2023).

- For each region: zero out connectome connections → re-simulate → measure TEP change
- Contribution matrix: (n_regions, n_timepoints) — time-resolved impact of each region
- Local-to-network transition time: when does network feedback exceed local reverberation?
- Momi's finding: ~100ms post-TMS

**Key question:** Does the transition time vary across subjects? Does it correlate with structural connectivity strength, conduction velocity, or GABA concentration?

**Status:** Code complete (`virtual_lesion.py`). Ready to run on fitted models.

### Phase 4: Predictive Modeling
Cross-validated prediction of TMS outcomes from pre-TMS features.

**Features (per subject):**
- Structural: connectome density, clustering, small-world index, hub topology
- Microstructural: mean axon diameter, conduction velocity distribution, g-ratio
- Functional: HMM state occupancies, lifetimes, switching rates, state spectra
- Metabolic: MRS GABA, glutamate, QMT myelin

**Targets:**
- TEP amplitude (peak-to-peak GFP)
- TEP complexity (perturbational complexity index)
- Local/network transition time
- Fitted model parameters (G, regional excitability)

**Methods:** Ridge regression, K-fold CV, permutation-based feature importance

**Status:** Code complete (`prediction.py`). Need features from processed data.

### Phase 5: Publication Outputs
- Fitted vs. empirical TEP waveforms (R² across subjects)
- Virtual lesion contribution matrix (group heatmap)
- Local→network transition time vs. structural/metabolic predictors
- Cross-modal prediction accuracy (which features matter?)
- T1w/T2w vs. QMT myelin validation

---

## Additional Research Topics

### 1. Microstructure → Oscillation Frequency
**Hypothesis:** Individual alpha frequency correlates with mean thalamocortical conduction velocity (from AxCaliber diameter).

The Valdes-Sosa et al. (2026, NSR) ξ-αNET paper shows conduction delays shape alpha across the lifespan using T1w/T2w as a myelin proxy. WAND can do this with *actual* microstructure from AxCaliber.

**Data:** ses-02 AxCaliber → diameter → Hursh-Rushton velocity; ses-01 MEG → individual alpha frequency

### 2. MRS-Constrained Neural Mass Models
**Instead of fitting J_I and J_NMDA freely, constrain them from MRS:**
- J_I ∝ [GABA] (from ses-04/05 MRS)
- J_NMDA ∝ [Glutamate] (from ses-04/05 MRS)

Does this improve model fit? Reduce parameter degeneracy?

### 3. T1w/T2w Ratio Validation
**Question:** How good is the T1w/T2w proxy for myelin content?

WAND provides ground truth from QMT (bound pool fraction), multi-echo GRE T2*, and VFA T1 mapping. The comparison script (`08_advanced_freesurfer.sh`) computes the vertex-wise correlation between T1w/T2w and QMT across the cortical surface.

**Expected:** Strong correlation in cortex, significant bias in regions with high iron content (basal ganglia) where T2* confounds the ratio.

### 4. Tucker-TDE vs. Standard TDE+PCA
**Methods contribution:** Compare tensor decomposition (TuckerTDE) against flatten+PCA for HMM/DyNeMo brain state analysis on the same 170 subjects.

### 5. SINDy/Koopman vs. HMM/DyNeMo
**Cross-method comparison:** Do data-driven dynamical systems (SINDy eigenvalues, DMD frequencies, log-signatures) find the same brain states as generative models (HMM/DyNeMo)?

### 6. Disconnectome Simulation
**Clinical translation:** Given a hypothetical lesion, predict the functional deficit by modifying the connectome (using DeepDisconnectome or tractography-based disconnection) and simulating the consequence with neurojax.

### 7. 4-Way Microstructure Comparison
**Benchmarking:** DIPY vs. FSL vs. sbi4dwi vs. DMI.jl on WAND AxCaliber data. Which method best estimates axon diameter on ultra-strong gradient data?

---

## Processing Pipeline

All scripts in `scripts/wand_processing/`:

| Script | Stage | Runtime |
|--------|-------|---------|
| `00_setup_env.sh` | Environment (FSL 6.0.7, FreeSurfer 8.2.0) | — |
| `01_freesurfer_recon.sh` | recon-all + T2 pial refinement | ~6-8h |
| `02_dwi_preproc.sh` | topup + eddy for AxCaliber/CHARMED | ~30-60min |
| `03_bedpostx_xtract.sh` | Crossing fibers + xtract tract segmentation | ~12-24h |
| `04_tracula.sh` | FreeSurfer+FSL tract-specific analysis | ~2-4h |
| `05_connectome.sh` | probtrackx2 → Cmat/Lmat CSV | ~4-8h |
| `06_run_all.sh` | Master script (all above) | ~24-48h |
| `07_microstructure_comparison.sh` | DIPY/FSL/sbi4dwi/DMI.jl AxCaliber | varies |
| `08_advanced_freesurfer.sh` | T1w/T2w, SAMSEG, thalamic nuclei, SynthSeg | ~2-4h |

---

## neurojax Module Inventory (Built This Session)

### Whole-Brain Modeling (`bench/`)
| Module | Purpose |
|--------|---------|
| `models/rww.py` | Reduced Wong-Wang neural mass model |
| `monitors/bold.py` | Balloon-Windkessel BOLD hemodynamics |
| `monitors/leadfield.py` | Source→sensor forward projection |
| `monitors/tep.py` | TEP extraction + waveform/GFP loss |
| `adapters/regional.py` | Per-region heterogeneous parameters |
| `adapters/vbjax_adapter.py` | Integrated adapter (BOLD, leadfield, regional, TMS, multi-modal loss, vmap) |
| `stimuli/tms.py` | TMS protocol + stimulus train generation |
| `optimizers/gradient.py` | Multi-modal gradient optimizer |

### Data Loading (`io/`)
| Module | Purpose |
|--------|---------|
| `connectome.py` | BIDS connectome loader (QSIRecon/dmipy-jax) |
| `wand_meg.py` | WAND CTF MEG loader |

### Analysis (`analysis/`)
| Module | Purpose |
|--------|---------|
| `state_spectra.py` | HMM/DyNeMo state → PSD + coherence (inverse TDE-PCA) |
| `summary_stats.py` | Fractional occupancy, lifetime, switching rate |
| `nnmf.py` | Spectral band separation (alpha/beta/gamma) |
| `multitaper.py` | DPSS tapers, multitaper PSD/CPSD/coherence |
| `regression_spectra.py` | DyNeMo GLM-spectrum approach |
| `static.py` | Time-averaged baseline (is dynamics needed?) |
| `tensor_tde.py` | Tucker decomposition alternative to TDE+PCA |
| `virtual_lesion.py` | Disconnection simulation (Momi 2023) |
| `prediction.py` | Cross-validated multi-modal prediction |
| `recurrence.py` | Recurrence plots + RQA measures |
| `visibility.py` | Visibility graphs (time series → network) |
| `funcnet.py` | Mutual information, graph measures, small-world |
| `surrogates.py` | Phase randomization, AAFT, significance testing |

### Models (`models/`)
| Module | Purpose |
|--------|---------|
| `hmm.py` | Gaussian HMM with Baum-Welch EM (osl-dynamics compatible) |
| `dynemo.py` | DyNeMo VAE (equinox, fully native JAX) |

### Other
| Module | Purpose |
|--------|---------|
| `preprocessing/filter.py` | IIR filter via jax.lax.scan (fixed lfilter bug) |
| `geometry/surface.py` | FreeSurfer surface IO |
| `geometry/source_space.py` | Icosahedron source space decimation |
| `dynamics/windowed.py` | Windowed SINDy/DMD/signatures |

---

## Test Coverage

**Total tests written this session: ~380+**

All modules have TDD test coverage. Key test files:
- `test_rww.py` (23), `test_leadfield.py` (25), `test_regional.py` (23)
- `test_integration.py` (34 — wired-up adapter tests)
- `test_tms.py` (27), `test_hmm.py` (22), `test_dynemo.py` (41)
- `test_windowed.py` (28), `test_connectome.py` (22), `test_filter.py` (10)
- `test_state_spectra.py` (23), `test_summary_stats.py` (21), `test_nnmf.py` (11)
- `test_multitaper.py` (15), `test_regression_spectra.py` (7), `test_static.py` (10)
- `test_tensor_tde.py` (13), `test_virtual_lesion.py` (17), `test_prediction.py` (21)
- `test_recurrence.py` (21), `test_visibility.py` (14), `test_funcnet.py` (17), `test_surrogates.py` (15)
- `test_surface.py` (8), `test_source_space.py` (11)

**Bugs found and fixed: ~12** (RWW gamma units, Viterbi T=1, filter lfilter, BOLD normalization, static connectivity normalization, Kronecker covariance symmetry, etc.)

---

## Key References

1. Momi D et al. (2023). TMS-evoked responses are driven by recurrent large-scale network dynamics. *eLife* 12:e83232.
2. Valdes-Sosa PA et al. (2026). Lifespan development of EEG alpha and aperiodic component sources is shaped by the connectome and axonal delays. *National Science Review*.
3. Matsulevits A et al. (2024). Deep learning disconnectomes to accelerate long-term post-stroke predictions. *Brain Communications* 6(5):fcae338.
4. Griffiths JD et al. (2022). Whole-brain modelling: past, present, and future. *Computational Modelling of the Brain*.
5. Gohil C et al. (2024). osl-dynamics: A toolbox for modelling fast dynamic brain activity. *eLife* 13:e91949.
